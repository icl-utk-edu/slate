// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/HermitianMatrix.hh"
#include "internal/internal.hh"
#include "work/work.hh"
#include "../test/print_matrix.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::he2hb from internal::specialization::he2hb
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage Hermitian eigenvalue
/// decomposition.
/// Generic implementation for any target.
/// Panel computed on host using Host OpenMP task.
///
/// ColMajor layout is assumed
///
/// @ingroup heev_specialization
///
template <Target target, typename scalar_t>
void he2hb(slate::internal::TargetType<target>,
           HermitianMatrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           int64_t ib, int max_panel_threads)
{
    using BcastList = typename HermitianMatrix<scalar_t>::BcastList;
    using blas::real;

    assert(A.uplo() == Uplo::Lower);  // for now

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    const scalar_t zero = 0;
    const scalar_t one  = 1;
    int64_t lookahead = 0;
    const scalar_t half = 0.5;

    int64_t nt = A.nt();

    // todo: TriangularFactors takes Matrix, not BaseMatrix or HermitianMatrix.
    // This abuses the "conversion sub-matrix" constructor to get a Matrix.
    T.clear();
    auto empty = A.emptyLike();
    auto Tlocal = Matrix<scalar_t>( empty, 0, nt-1, 0, nt-1 );
    auto Treduce = Tlocal.emptyLike(ib, 0);
    T.push_back( Tlocal );
    T.push_back( Treduce );

    // workspace
    auto W = A.emptyLike();
    auto Wtmp = A.emptyLike();
    //auto TVAVT = A.emptyLike();
    auto Asave = A.emptyLike();

    // Use W(0, 0) for TVAVT, since W(0, 0) is never used otherwise.
    W.tileInsert(0, 0);
    // use TVAVT = W.sub(0, 0, 0, 0)
    auto TVAVT = W(0, 0);
    TVAVT.uplo(Uplo::General);

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays();
        W.reserveDeviceWorkspace();
    }

    // No lookahead is possible, so no need to track dependencies --
    // just execute tasks in order. Also, priority isn't needed.

    int my_rank = A.mpiRank();

    // tracks dependencies by block-column.
    std::vector< uint8_t > block_vector(nt);
    uint8_t* block = block_vector.data();
    uint8_t* row = block_vector.data();

    // bool debug = true;

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        for (int64_t k = 0; k < nt-1; ++k) {

            //----------------------------------------
            // Q panel and update.
            auto       A_panel =       A.sub(k+1, nt-1, k, k);
            auto  Tlocal_panel =  Tlocal.sub(k+1, nt-1, k, k);
            auto Treduce_panel = Treduce.sub(k+1, nt-1, k, k);

            // Find ranks in this column.
            std::set<int> panel_ranks;
            A_panel.getRanks(&panel_ranks);
            assert(panel_ranks.size() > 0);

            // Find each rank's first (top-most) row in this panel,
            // where the triangular tile resulting from local geqrf panel
            // will reside.
            std::vector< int64_t > first_indices;
            first_indices.reserve(panel_ranks.size());
            for (int r: panel_ranks) {
                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                    if (A_panel.tileRank(i, 0) == r) {
                        // todo: way to get index in parent matrix, to avoid manually adding k+1?
                        first_indices.push_back(i+k+1);
                        break;
                    }
                }
            }
            std::vector<int64_t> indices;
            std::vector<int64_t> indices2;
            int64_t i0;

            //--------------------
            // QR of panel
            // local panel factorization
            internal::geqrf<Target::HostTask>(
                            std::move(A_panel),
                            std::move(Tlocal_panel),
                            ib, max_panel_threads);

            // triangle-triangle reductions
            // ttqrt handles tile transfers internally
            internal::ttqrt<Target::HostTask>(
                            std::move(A_panel),
                            std::move(Treduce_panel));

            // if a trailing matrix exists.
            if (k < nt-1) {

                // Send V across row i & col i for trailing matrix update.
                BcastList bcast_list_V_first;
                BcastList bcast_list_V;
                for (int64_t i = k; i < nt; ++i) {
                    // Vs need 2 lives for unmqr (I - VTV^H).
                    // Vs in first_indices (except top-most one, i == k+1)
                    // need 3 lives: 2 for unmqr, 1 for ttmqr.
                    if (i > k+1 && std::find(first_indices.begin(), first_indices.end(), i) != first_indices.end()) {
                        bcast_list_V_first.push_back(
                            {i, k, {A.sub(i, i, k+1, i)}});
                        bcast_list_V_first.push_back(
                             {i, k, {A.sub(i+1, nt-1, i, i)}});
                    }
                    else {
                        bcast_list_V.push_back(
                            {i, k, {A.sub(i, i, k+1, i)}});
                        bcast_list_V.push_back(
                            {i, k, {A.sub(i+1, nt-1, i, i)}});
                    }
                }
                A.template listBcast(bcast_list_V_first, layout, 0, 3);
                A.template listBcast(bcast_list_V, layout, 0, 2);

                // Send Treduce across row i & col i for trailing matrix update
                if (first_indices.size() > 1) {
                    BcastList bcast_list_T;
                    for (int64_t i : first_indices) {
                        // Exclude first row of this panel,
                        // which doesn't have Treduce tile.
                        if (i > k+1) {
                            bcast_list_T.push_back(
                                {i, k, {Treduce.sub(i, i, k+1, i)}});
                            bcast_list_T.push_back(
                                {i, k, {Treduce.sub(i+1, nt-1, i, i)}});
                        }
                    }
                    Treduce.template listBcast(bcast_list_T, layout);
                }

                //TODO: keep only this indices loop
                for (int panel_rank: panel_ranks) {
                    // Find local indices for panel_rank.
                    indices.clear();
                    for (int64_t i = 0; i < A_panel.mt(); ++i) {
                        if (A_panel.tileRank(i, 0) == panel_rank) {
                            // todo: global index
                            indices.push_back(i+k+1);
                        }
                    }
                    i0 = indices[0];

                    // Send Tlocal across row i & col i for trailing matrix update
                    // todo: I think this sets Tlocal with too many lives
                    // -- needs only 1 life per rank, not # tiles.
                    BcastList bcast_list_T;
                    for (int64_t i : indices) {
                        bcast_list_T.push_back(
                            {i0, k, {Tlocal.sub(i, i, k+1, i)}});
                        bcast_list_T.push_back(
                            {i0, k, {Tlocal.sub(i+1, nt-1, i, i)}});
                    }
                    Tlocal.template listBcast(bcast_list_T, layout);
                }
            }

            //----------------------------------------
            // QR update trailing submatrix.
            if (k < nt-1) {

                for (int64_t i = k+1; i < nt; ++i) {
                    W.tileInsert(i, k);
                    W(i, k).set(zero);
                }

                int64_t i0;
                for (int panel_rank: panel_ranks) {
                    // Find local indices for panel_rank.
                    indices.clear();
                    indices2.clear();
                    for (int64_t i = 0; i < A_panel.mt(); ++i) {
                        if (A_panel.tileRank(i, 0) == panel_rank) {
                            // todo: global index
                            indices.push_back(i+k+1);
                            indices2.push_back(i);
                        }
                    }
                    i0 = indices[0];

                    if (A.tileExists(i0, k)) {
                        // Save V0 and set upper(V0) to identity, to avoid trmm's.
                        Asave.tileInsert(i0, k);
                        auto Aik = A(i0, k);
                        gecopy(std::move(Aik), Asave(i0, k));
                        Aik.uplo(Uplo::Upper);
                        Aik.set(zero, one);
                    }
                    #pragma omp taskwait
                    //--------------------
                    // Apply local reflectors.
                    // Compute Wi = (sum_j Aij Vj) T, for i = k+1, ..., nt-1.
                    internal::he2hb_hemm<target>(
                            A.sub(k+1, nt-1),
                            A.sub(k+1, nt-1, k, k),
                            W.sub(k+1, nt-1, k, k),
                            indices2, &row[k+1]);

                    int rank_lower = -1;
                    int rank_upper = -1;

                    // At most 2 ranks contribute to each Wi; if I am one,
                    // exchange partial sum with neighbor and both ranks sum Wi.
                    for (int64_t i = k+1; i < nt; ++i) {
                        #pragma omp task depend(inout:row[i])
                        {

                            for (int64_t j: indices) {
                                if (i >= j) { // lower
                                    rank_lower = A.tileRank(i, j);
                                }
                                else { // upper
                                    rank_upper = A.tileRank(j, i);
                                }
                            }
                            int neighbor = -1;
                            if (rank_lower == my_rank)
                                neighbor = rank_upper;
                            else if (rank_upper == my_rank)
                                neighbor = rank_lower;
                            if (neighbor != -1 && neighbor != my_rank) {
                                Wtmp.tileInsert(i, k);
                                int tag_i = i;
                                int tag_i_ = i+1;
                                if (neighbor < my_rank) {
                                    W.tileGetForWriting(i, k, W.hostNum(), LayoutConvert(layout));
                                    W   .tileSend(i, k, neighbor, tag_i);
                                    Wtmp.tileRecv(i, k, neighbor, layout, tag_i_);
                                }
                                else {
                                    W.tileGetForWriting(i, k, W.hostNum(), LayoutConvert(layout));
                                    Wtmp.tileRecv(i, k, neighbor, layout, tag_i);
                                    W   .tileSend(i, k, neighbor, tag_i_);
                                }
                                {
                                    axpy(one, Wtmp(i, k), W(i, k));
                                    Wtmp.tileErase(i, k);
                                }
                            }
                        }
                    }
                    #pragma omp taskwait

                    internal::he2hb_trmm<target>(
                            A.sub(k+1, nt-1), // todo: needed to get the rank, try replace it with W
                            W.sub(k+1, nt-1, k, k),
                            Tlocal.sub(i0, i0, k, k),
                            indices2, &row[k+1]);
                        auto Wnt1 = W.sub(k+1, nt-1, k, k);
                        Wnt1.tileUpdateAllOrigin();

                    if (A.tileIsLocal(i0, i0)) {
                        //--------------------
                        // This rank has diagonal tiles to update.
                        // Do 2-sided Hermitian update:
                        // A = Q^H A Q
                        //   = (I - V T^H V^H) A (I - V T V^H)
                        //   = A - V W^H - W V^H
                        // where
                        // W = A V T - 0.5 V (T^H V^H (A V T)).

                        // 1a. W = AVT from above.
                        // 1b. TVAVT = V^H (AVT) = V^H W.
                        // Call internal::geset
                        W.tileGetForWriting(0, 0, W.hostNum(), LayoutConvert(layout));
                        TVAVT.set(zero);

                        auto AT = conjTranspose(A.sub(k+1, nt-1, k, k));
                        internal::he2hb_gemm<target>(
                                        one,  std::move(AT),
                                              W.sub(k+1, nt-1, k, k),
                                        zero, W.sub(0, 0, 0, 0),
                                        panel_rank,
                                        &row[k+1], &block[0]);

                        // 1c. TVAVT = T^H (V^H AVT)
                        auto T0    = Tlocal.sub(i0, i0, k, k);
                        auto TVAVT0  = W.sub(0, 0, 0, 0);
                        //T0(0, 0).set(zero);
                        //TVAVT0(0, 0).set(zero);

                        int64_t mb = T0.tileMb(0);
                        int64_t nb = T0.tileNb(0);
                        bool trapezoid = (mb < nb);

                        if (trapezoid) {
                            T0     = T0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                            TVAVT0 = TVAVT0.slice(0, mb-1, 0, nb-1); // first mb-by-nb part
                        }

                        //auto W0 = W.sub(0, 0, 0, 0);
                        // todo: try to call internal::trmm
                        auto Tk0 = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
                        Tk0.tileGetForReading(0, 0, Tk0.hostNum(), LayoutConvert(layout));
                        TVAVT0.tileGetForWriting(0, 0, TVAVT0.hostNum(), LayoutConvert(layout));
                        #pragma omp task depend(in:block[k]) depend(inout:block[0])
                        trmm(Side::Left, Diag::NonUnit,
                             one, conjTranspose(Tk0(0, 0)), std::move(TVAVT0(0, 0)));
                        #pragma omp taskwait
                        //W00.tileUpdateAllOrigin();
                        //TVAVT0.tileUpdateOrigin(0, 0);
                        //W.tileGetForReading(0, 0, W.hostNum(), LayoutConvert(layout));


                        //#pragma omp task depend(in:block[k]) depend(inout:block[0])
                        //internal::trmm<Target::HostTask>(
                        //    Side::Left,
                        //    one, conjTranspose(Tk0),
                        //    std::move(TVAVT0));

                        // 1d. W = W - 0.5 V TVAVT.
                        // Technically, could do a hemm here since TVAVT is Hermitian.
                        //todo: use Debug class to check
                        internal::he2hb_gemm<target>(
                                        -half, A.sub(k+1, nt-1, k, k),
                                                  W.sub(0, 0, 0, 0),
                                        one,      W.sub(k+1, nt-1, k, k),
                                        panel_rank,
                                        &block[0], &row[k+1]);

                        // 2. Update trailing matrix.
                        // todo: use debug class to check why //A.tileTick(i, 0) is needed?
                        internal::her2k<target>(
                                        -one,  A.sub(k+1, nt-1, k, k),
                                               W.sub(k+1, nt-1, k, k),
                                        1.0,   A.sub(k+1, nt-1));
                    }
                    else { // off-diag
                        //--------------------
                        // This rank has only off-diagonal tiles to update (if any).
                        // Update from left:
                        // A = Q^H A
                        //   = (I - V T^H V^H) A = A - V T^H V^H A
                        //   = A - V W^H
                        // where
                        // W = A^H V T = A V T.

                        auto W_panel  = W.sub(k+1, nt-1, k, k);
                        //todo: only for this one pass row and block dep
                        internal::he2hb_gemm_outer<target>(
                                        -one, A.sub(k+1, nt-1, k, k),
                                              std::move(W_panel),
                                        one,  A.sub(k+1, nt-1),
                                        indices2, &row[k+1], &block[k+1]);
                    }

                    if (A.tileExists(i0, k)) {
                        // Restore V0.
                        #pragma omp task depend(inout:block[k])
                        {
                            gecopy(Asave(i0, k), A(i0, k));
                            //internal::copy<target>(Asave.sub(i0, i0, k, k), A.sub(i0, i0, k, k));
                            Asave.tileErase(i0, k);
                        }
                    }
                }

                internal::hettmqr<Target::HostTask>(
                    Op::ConjTrans, std::move(A_panel), std::move(Treduce_panel),
                    A.sub(k+1, nt-1));
            } //if (k < nt-1)
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
    W.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup heev_specialization
///
template <Target target, typename scalar_t>
void he2hb(HermitianMatrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           Options const& opts)
{
    int64_t ib;
    try {
        ib = opts.at(Option::InnerBlocking).i_;
        assert(ib >= 0);
    }
    catch (std::out_of_range&) {
        ib = 16;
    }

    int64_t max_panel_threads;
    try {
        max_panel_threads = opts.at(Option::MaxPanelThreads).i_;
        assert(max_panel_threads >= 1);
    }
    catch (std::out_of_range&) {
        max_panel_threads = std::max(omp_get_max_threads()/2, 1);
    }

    internal::specialization::he2hb(internal::TargetType<target>(),
                                    A, T,
                                    ib, max_panel_threads);
}

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage SVD.
///
/// Reduces an n-by-n Hermitian matrix $A$ to band form using unitary
/// transformations. The factorization has the form
/// \[
///     A = Q B Q^H
/// \]
/// where $Q$ is unitary and $B$ is Hermitian band
/// with nb sub and superdiagonals.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix $A$.
///     On exit:
///     - [upper is not yet implemented]
///       If A is upper, the elements Aij for j = i, ..., i+nb,
///       represent the Hermitian band matrix B.
///       The elements above the nb-th superdiagonal, along with T, represent
///       the unitary matrix $Q$ as a product of elementary reflectors.
///     - If A is lower, the elements Aij for j = i-nb, ..., i,
///       represent the Hermitian band matrix B.
///       The elements below the nb-th subdiagonal, along with T, represent
///       the unitary matrix $Q$ as a product of elementary reflectors.
///
/// @param[out] T
///     On exit, triangular matrices of the block reflectors for Q.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///     Note a lookahead is not possible with he2hb due to dependencies from
///     updating on both left and right sides.
///
/// @ingroup heev_computational
///
template <typename scalar_t>
void he2hb(HermitianMatrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           Options const& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            he2hb<Target::HostTask>(A, T, opts);
            break;
        //case Target::HostNest:
        //    he2hb<Target::HostNest>(A, T, opts);
        //    break;
        //case Target::HostBatch:
        //    he2hb<Target::HostBatch>(A, T, opts);
        //    break;
        case Target::Devices:
            he2hb<Target::Devices>(A, T, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void he2hb<float>(
    HermitianMatrix<float>& A,
    TriangularFactors<float>& T,
    Options const& opts);

template
void he2hb<double>(
    HermitianMatrix<double>& A,
    TriangularFactors<double>& T,
    Options const& opts);

template
void he2hb< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Options const& opts);

template
void he2hb< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Options const& opts);

} // namespace slate
