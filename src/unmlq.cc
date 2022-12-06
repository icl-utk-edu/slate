// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/Tile_tpmlqt.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel multiply by Q from LQ factorization.
/// Generic implementation for any target.
/// @ingroup gelqf_impl
///
template <Target target, typename scalar_t>
void unmlq(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts )
{
    // trace::Block trace_block("unmlq");
    // const int priority_one = 1;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);
    int64_t C_mt = C.mt();
    int64_t C_nt = C.nt();

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    // Reserve workspace
    auto W = C.emptyLike();

    if (target == Target::Devices) {
        W.allocateBatchArrays();
        // todo: this is demanding too much device workspace memory
        // only one tile-col of matrix W per MPI process is going to be used,
        // but W with size of whole C is being allocated
        // thus limiting the matrix size that can be processed
        //W.reserveDeviceWorkspace();
    }

    assert(T.size() == 2);
    auto Tlocal  = T[0];
    auto Treduce = T[1];

    // LQ tracks dependencies by block-row.
    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > block_vector(A_mt);
    uint8_t* block = block_vector.data();

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        int64_t k_begin, k_end, k_step;
        if ((side == Side::Left) != (op == Op::NoTrans)) {
            // Left, (Conj)Trans: multiply Q^H C = Q1^H ... QK^H C, or
            // Right, NoTrans:    multiply C Q   = C QK ... Q1,
            // i.e., in reverse order of how Qk's were created.
            k_begin = A_min_mtnt-1;
            k_end   = -1;
            k_step  = -1;
        }
        else {
            // Left,  NoTrans:     multiply Q C   = QK ... Q1 C, or
            // Right, (Conj)Trans: multiply C Q^H = C Q1^H ... QK^H,
            // i.e., in same order as Qk's were created.
            k_begin = 0;
            k_end   = A_min_mtnt;
            k_step  = 1;
        }

        // for k = k_begin, lastk = k_begin (no previous col to depend on);
        // otherwise,       lastk = k - k_step (previous col).
        int64_t lastk = k_begin;
        // OpenMP uses lastk; compiler doesn't, so warns it is unused.
        SLATE_UNUSED(lastk);
        for (int64_t k = k_begin; k != k_end; k += k_step) {

            auto A_panel = A.sub(k, k, k, A_nt-1);

            // Find ranks in this row.
            std::set<int> ranks_set;
            A_panel.getRanks(&ranks_set);
            assert(ranks_set.size() > 0);

            // Find each rank's first (left-most) col in this panel,
            // where the triangular tile resulting from local gelqf
            // panel will reside.
            std::vector< int64_t > first_indices;
            first_indices.reserve(ranks_set.size());
            for (int r: ranks_set) {
                for (int64_t j = 0; j < A_panel.nt(); ++j) {
                    if (A_panel.tileRank(0, j) == r) {
                        first_indices.push_back(j+k);
                        break;
                    }
                }
            }

            #pragma omp task depend(inout:block[k]) \
                             depend(in:block[lastk])
            {
                // Indices for row or col of C.
                int64_t i0 = -1, i1 = -1, j0 = -1, j1 = -1;
                if (side == Side::Left) {
                    j0 = 0;
                    j1 = C_nt-1;
                }
                else {
                    i0 = 0;
                    i1 = C_mt-1;
                }

                // Send V(j) across row C(j, 0:nt-1) or col C(0:mt-1, j),
                // for side = left or right, respectively.
                BcastList bcast_list_V_top;
                BcastList bcast_list_V;
                for (int64_t j = k; j < A_nt; ++j) {
                    if (side == Side::Left) {
                        i0 = j;
                        i1 = j;
                    }
                    else {
                        j0 = j;
                        j1 = j;
                    }
                    // Vs in first_indices (except the left-most one)
                    // need three lives
                    if (j > k && std::find(first_indices.begin(), first_indices.end(), j) != first_indices.end()) {
                        bcast_list_V_top.push_back(
                            {k, j, {C.sub(i0, i1, j0, j1)}});
                    }
                    else {
                        bcast_list_V.push_back(
                            {k, j, {C.sub(i0, i1, j0, j1)}});
                    }
                }
                A.template listBcast(bcast_list_V_top, layout, 0, 3);
                A.template listBcast(bcast_list_V, layout, 0, 2);

                // Send Tlocal(j) across row C(j, 0:nt-1) or col C(0:mt-1, j).
                if (first_indices.size() > 0) {
                    BcastList bcast_list_T;
                    for (int64_t j : first_indices) {
                        if (side == Side::Left) {
                            i0 = j;
                            i1 = j;
                        }
                        else {
                            j0 = j;
                            j1 = j;
                        }
                        bcast_list_T.push_back(
                            {k, j, {C.sub(i0, i1, j0, j1)}});
                    }
                    Tlocal.template listBcast(bcast_list_T, layout);
                }

                // Send Treduce(j) across row C(j, 0:nt-1) or col C(0:mt-1, j).
                if (first_indices.size() > 1) {
                    BcastList bcast_list_T;
                    for (int64_t j : first_indices) {
                        if (side == Side::Left) {
                            i0 = j;
                            i1 = j;
                        }
                        else {
                            j0 = j;
                            j1 = j;
                        }
                        // Exclude first col of this panel,
                        // which doesn't have Treduce tile.
                        if (j > k) {
                            bcast_list_T.push_back(
                                {k, j, {C.sub(i0, i1, j0, j1)}});
                        }
                    }
                    Treduce.template listBcast(bcast_list_T, layout);
                }

                Matrix<scalar_t> C_trail, W_trail;
                if (side == Side::Left) {
                    C_trail = C.sub(k, C_mt-1, 0, C_nt-1);
                    W_trail = W.sub(k, C_mt-1, 0, C_nt-1);
                }
                else {
                    C_trail = C.sub(0, C_mt-1, k, C_nt-1);
                    W_trail = W.sub(0, C_mt-1, k, C_nt-1);
                }

                // Left,  (Conj)Trans: Qi^H C = Qi_local^H Qi_reduce^H C, or
                // Right, NoTrans:     C Qi   = C Qi_reduce Qi_local,
                // do ttmqr then unmqr.
                if ((side == Side::Left) != (op == Op::NoTrans)) {
                    // Apply triangle-triangle reduction reflectors.
                    internal::ttmlq<Target::HostTask>(
                                    side, op,
                                    std::move(A_panel),
                                    Treduce.sub(k, k, k, A_nt-1),
                                    std::move(C_trail));
                }

                // Apply local reflectors.
                internal::unmlq<target>(
                                side, op,
                                std::move(A_panel),
                                Tlocal.sub(k, k, k, A_nt-1),
                                std::move(C_trail),
                                std::move(W_trail));

                // Left,  NoTrans:     Qi C   = Qi_reduce Qi_local C, or
                // Right, (Conj)Trans: C Qi^H = C Qi_local^H Qi_reduce^H,
                // do unmqr then ttmqr.
                if ((side == Side::Left) == (op == Op::NoTrans)) {
                    // Apply triangle-triangle reduction reflectors.
                    internal::ttmlq<Target::HostTask>(
                                    side, op,
                                    std::move(A_panel),
                                    Treduce.sub(k, k, k, A_nt-1),
                                    std::move(C_trail));
                }
            }

            lastk = k;
        }

        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }
    C.clearWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel multiply by $Q$ from LQ factorization.
///
/// Multiplies the general m-by-n matrix $C$ by $Q$ from LQ factorization,
/// according to:
///
/// op              |  side = Left  |  side = Right
/// --------------- | ------------- | --------------
/// op = NoTrans    |  $Q C  $      |  $C Q  $
/// op = ConjTrans  |  $Q^H C$      |  $C Q^H$
///
/// where $Q$ is a unitary matrix defined as the product of k
/// elementary reflectors
/// \[
///     Q = H(1) H(2) . . . H(k)
/// \]
/// as returned by gelqf. $Q$ is of order m if side = Left
/// and of order n if side = Right.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///     - Side::Left:  apply $Q$ or $Q^H$ from the left;
///     - Side::Right: apply $Q$ or $Q^H$ from the right.
///
/// @param[in] op
///     - Op::NoTrans    apply $Q$;
///     - Op::ConjTrans: apply $Q^H$;
///     - Op::Trans:     apply $Q^T$ (only if real).
///       In the real case, Op::Trans is equivalent to Op::ConjTrans.
///       In the complex case, Op::Trans is not allowed.
///
/// @param[in] A
///     Details of the LQ factorization of the original matrix $A$ as returned
///     by gelqf.
///
/// @param[in] T
///     Triangular matrices of the block reflectors as returned by gelqf.
///
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, $C$ is overwritten by $Q C$, $Q^H C$, $C Q$, or $C Q^H$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup gelqf_computational
///
template <typename scalar_t>
void unmlq(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::unmlq<Target::HostTask>( side, op, A, T, C, opts  );
            break;

        case Target::HostNest:
            impl::unmlq<Target::HostNest>( side, op, A, T, C, opts  );
            break;

        case Target::HostBatch:
            impl::unmlq<Target::HostBatch>( side, op, A, T, C, opts  );
            break;

        case Target::Devices:
            impl::unmlq<Target::Devices>( side, op, A, T, C, opts  );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmlq<float>(
    Side side, Op op,
    Matrix<float>& A,
    TriangularFactors<float>& T,
    Matrix<float>& C,
    Options const& opts);

template
void unmlq<double>(
    Side side, Op op,
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Matrix<double>& C,
    Options const& opts);

template
void unmlq< std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Matrix< std::complex<float> >& C,
    Options const& opts);

template
void unmlq< std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
