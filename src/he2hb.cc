// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/HermitianMatrix.hh"
#include "internal/internal.hh"

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
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;
    using blas::real;

    assert(A.uplo() == Uplo::Lower);  // for now

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    const scalar_t zero = 0;
    const scalar_t one  = 1;
    const scalar_t half = 0.5;
    const int priority_one = 1;

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
    auto Wtmp = A.emptyLike();
    auto Asave = A.emptyLike();
    const bool set_hold =  1;  // Do tileGetAndHold in the bcast

    slate::HermitianMatrix<scalar_t> W;
    int64_t n = A.n();
    int64_t nb_A = A.tileNb(0);
    GridOrder order;
    int nprow, npcol, myrow, mycol;
    // For the workspace matrix (W) change the GPU distribution to row cyclic,
    // by doing that, all the GPUs will be busy simultaneously,
    // in particular when we call he2hb_hemm,
    // which will improve the performance.
    A.gridinfo( &order, &nprow, &npcol, &myrow, &mycol );
    std::function<int64_t (int64_t j)> tileNb = [n, nb_A] (int64_t j) {
        return (j + 1)*nb_A > n ? n%nb_A : nb_A;
    };
    std::function<int (std::tuple<int64_t, int64_t> ij)>
    tileRank = [nprow, npcol](std::tuple<int64_t, int64_t> ij) {
        int64_t i = std::get<0>(ij);
        int64_t j = std::get<1>(ij);
        return int(i%nprow + (j%npcol)*nprow);
    };
    int num_devices = blas::get_device_count();
    std::function<int (std::tuple<int64_t, int64_t> ij)>
    tileDevice = [nprow, num_devices](std::tuple<int64_t, int64_t> ij) {
        int64_t i = std::get<0>(ij);
        return int(i/nprow)%num_devices;
    };
    W = slate::HermitianMatrix<scalar_t>(
            Uplo::Lower, n, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);

    W.tileInsert(0, 0);
    auto TVAVT = W.sub(0, 0, 0, 0);
    int num_queues = 10;

    int my_rank = A.mpiRank();

    // tracks dependencies by block-column.
    std::vector< uint8_t > block_vector(nt);
    uint8_t* block = block_vector.data();
    std::vector< uint8_t > alloc_workspace_vector(1);
    uint8_t* alloc_workspace = alloc_workspace_vector.data();
    std::vector< uint8_t > alloc_trailing_vector(1);
    uint8_t* alloc_trailing = alloc_trailing_vector.data();

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
                        first_indices.push_back(i+k+1);
                        break;
                    }
                }
            }
            std::vector<int64_t> indices;
            std::vector<int64_t> indices_local_panel;
            int64_t i0;

            if (k == 0) {
                #pragma omp task depend(inout:alloc_workspace[0])
                {
                    if (target == Target::Devices) {
                        A.allocateBatchArrays();
                        A.reserveDeviceWorkspace();
                        W.allocateBatchArrays(0, num_queues);
                        W.reserveDeviceWorkspace();
                    }
                }
            }

            //--------------------
            // QR of panel
            // local panel factorization
            #pragma omp task depend(inout:block[k])
            {
                internal::geqrf<Target::HostTask>(
                    std::move(A_panel),
                    std::move(Tlocal_panel),
                    ib, max_panel_threads, priority_one);

                // triangle-triangle reductions
                // ttqrt handles tile transfers internally
                internal::ttqrt<Target::HostTask>(
                    std::move(A_panel),
                    std::move(Treduce_panel));
            }

            #pragma omp task depend(in:block[k]) \
                             depend(in:alloc_workspace[0])
            {
                // if a trailing matrix exists.
                if (k < nt-1) {

                    // Send V across row i & col i for trailing matrix update.
                    BcastListTag bcast_list_V_first;
                    BcastListTag bcast_list_V;
                    for (int64_t i = k; i < nt; ++i) {
                        // Vs need 6 lives.
                        // Vs in first_indices (except top-most one, i == k+1)
                        // need 3 lives: for her2k, gemm_outer, and for hettmqr.
                        if (i > k+1 && std::find(first_indices.begin(), first_indices.end(), i) != first_indices.end()) {
                            bcast_list_V_first.push_back(
                                {i, k, {A.sub(i, i, k+1, i),
                                        A.sub(i+1, nt-1, i, i)},
                                        i});
                        }
                        else {
                            bcast_list_V.push_back(
                                {i, k, {A.sub(i, i, k+1, i),
                                        A.sub(i+1, nt-1, i, i)},
                                        i});
                        }
                    }
                    A.template listBcastMT<target>(bcast_list_V_first, layout, 5, set_hold);
                    A.template listBcastMT<target>(bcast_list_V, layout, 6, set_hold);

                    if (first_indices.size() > 1) {
                        //BcastList bcast_list_T;
                        BcastListTag bcast_list_T;
                        for (int64_t i : first_indices) {
                            // Exclude first row of this panel,
                            // which doesn't have Treduce tile.
                            if (i > k+1) {
                                bcast_list_T.push_back(
                                    {i, k, {Treduce.sub(i, i, k+1, i),
                                            Treduce.sub(i+1, nt-1, i, i)},
                                            i});
                            }
                        }
                        Treduce.template listBcastMT(bcast_list_T, layout);
                    }

                    for (int panel_rank: panel_ranks) {
                        // Find local indices for panel_rank.
                        indices.clear();
                        for (int64_t i = 0; i < A_panel.mt(); ++i) {
                            if (A_panel.tileRank(i, 0) == panel_rank) {
                                // global index
                                indices.push_back(i+k+1);
                            }
                        }
                        i0 = indices[0];

                        // Send Tlocal across row i & col i for trailing matrix update
                        // todo: I think this sets Tlocal with too many lives
                        // -- needs only 1 life per rank, not # tiles.
                        //BcastList bcast_list_T;
                        BcastListTag bcast_list_T;
                        for (int64_t i : indices) {
                            bcast_list_T.push_back(
                                {i0, k, {Tlocal.sub(i, i, k+1, i),
                                        Tlocal.sub(i+1, nt-1, i, i)},
                                        i});
                        }
                        Tlocal.template listBcastMT<target>(bcast_list_T, layout, 1, set_hold);
                    }
                }
            }
            // computation and data-movement overlap: Fetching data to be used in he2hb_hemm
            #pragma omp task depend(in:alloc_workspace[0]) \
                             depend(inout:alloc_trailing[0])
            {

                for (int64_t i = k+1; i < nt; ++i) {
                    W.tileInsert(i, k);
                    W(i, k).set(zero);
                }

		if (target == Target::Devices) {
                    std::vector<int64_t> indices_copy;
                    auto  W_panel = W.sub(k+1, nt-1, k, k);
                    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
                    if (k < nt-1) {
                        for (int panel_rank: panel_ranks) {

                            // Find local indices for panel_rank.
                            indices_copy.clear();
                            for (int64_t i = 0; i < A_panel.mt(); ++i) {
                                if (A_panel.tileRank(i, 0) == panel_rank) {
                                    // global index
                                    indices_copy.push_back(i+k+1);
                                }
                            }
                            for (int device = 0; device < W_panel.num_devices(); ++device) {
                                #pragma omp task shared(A, W)
                                {
                                    std::set<ij_tuple> A_tiles_set, A_panel_tiles_set, W_tiles_set;
                                    for (int64_t j: indices_copy) {
                                        for (int64_t i = k+1; i < nt; ++i) {
                                            if (i >= j) { // lower or diagonal
                                                if (A.tileIsLocal(i, j)) {
                                                    if (device == W.tileDevice(i, k)) {
                                                        A_tiles_set.insert({i, j});
                                                        W_tiles_set.insert({i, k});
                                                    }
                                                }
                                            }
                                            else { // upper
                                                if (A.tileIsLocal(j, i)) {
                                                    if (device == W.tileDevice(i, k)) {
                                                        A_tiles_set.insert({j, i});
                                                        W_tiles_set.insert({i, k});
                                                    }
                                                }
                                            }
                                        }

                                        #pragma omp task default(shared)
                                        {
                                            A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                                        }
                                        #pragma omp task default(shared)
                                        {
                                            W.tileGetForWriting(W_tiles_set, device, LayoutConvert(layout));
                                        }
                                        #pragma omp taskwait
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma omp taskwait
            }

            //----------------------------------------
            // QR update trailing submatrix.
            if (k < nt-1) {

                for (int panel_rank: panel_ranks) {
                    // Find local indices for panel_rank.
                    indices.clear();
                    indices_local_panel.clear();
                    for (int64_t i = 0; i < A_panel.mt(); ++i) {
                        if (A_panel.tileRank(i, 0) == panel_rank) {
                            // global index
                            indices.push_back(i+k+1);
                            // local index: indeces within the panel,
                            // to be used in the subsequent calls
                            indices_local_panel.push_back(i);
                        }
                    }
                    i0 = indices[0];

                    #pragma omp task depend(inout:block[k])
                    {
                        if (A.tileExists(i0, k)) {
                            A.tileGetForWriting(i0, k, A.hostNum(),
                                                LayoutConvert(layout));
                            // Save V0 and set upper(V0) to identity, to avoid trmm's.
                            Asave.tileInsert(i0, k);
                            auto Aik = A(i0, k);
                            gecopy(std::move(Aik), Asave(i0, k));
                            Aik.uplo(Uplo::Upper);
                            Aik.set(zero, one);
                        }
                    }
                    //--------------------
                    // Apply local reflectors.
                    // Compute Wi = (sum_j Aij Vj) T, for i = k+1, ..., nt-1.
                    #pragma omp task depend(inout:block[k]) \
                                     depend(inout:alloc_trailing[0])
                    {
                        internal::he2hb_hemm<target>(
                            A.sub(k+1, nt-1),
                            A.sub(k+1, nt-1, k, k),
                            W.sub(k+1, nt-1, k, k),
                            indices_local_panel);
                    }

                    int rank_lower = -1;
                    int rank_upper = -1;

                    // At most 2 ranks contribute to each Wi; if I am one,
                    // exchange partial sum with neighbor and both ranks sum Wi.
                    #pragma omp task depend(inout:block[k]) \
                                     depend(inout:alloc_trailing[0])
                    {
                        for (int64_t i = k+1; i < nt-1; ++i) {
                            #pragma omp task
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
                                        W.tileGetForWriting(i, k, W.hostNum(),
                                                            LayoutConvert(layout));
                                        W   .tileSend(i, k, neighbor, tag_i);
                                        Wtmp.tileRecv(i, k, neighbor, layout, tag_i_);
                                    }
                                    else {
                                        W.tileGetForWriting(i, k, W.hostNum(),
                                                            LayoutConvert(layout));
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
                    }

                    #pragma omp task depend(inout:block[k]) \
                                     depend(in:alloc_trailing[0])
                    {
                        internal::he2hb_trmm<target>(
                            A.sub(k+1, nt-1), // Needed to get the rank
                            W.sub(k+1, nt-1, k, k),
                            Tlocal.sub(i0, i0, k, k),
                            indices_local_panel);
                    }

                    if (A.tileIsLocal(i0, i0)) {
                        //--------------------
                        // This rank has diagonal tiles to update.
                        // Do 2-sided Hermitian update:
                        // A = Q^H A Q
                        //   = (I - V T^H V^H) A (I - V T V^H)
                        //   = A - V W^H - W V^H
                        // where
                        // W = A V T - 0.5 V (T^H V^H (A V T)).

                        // 1a. W = A VT from above.

                        #pragma omp task depend(in:block[k]) \
                                         depend(inout:block[0]) \
                                         depend(inout:block[k+1]) \
                                         depend(inout:block[nt-1])
                        {
                            // 1b. TVAVT = V^H (AVT) = V^H W.
                            auto A_panelT = conjTranspose(A.sub(k+1, nt-1, k, k));
                            W.tileGetForWriting(0, 0, W.hostNum(),
                                                LayoutConvert(layout));
                            TVAVT(0, 0).set(zero);
                            internal::he2hb_gemm<target>(
                                one,  std::move(A_panelT),
                                W.sub(k+1, nt-1, k, k),
                                zero, W.sub(0, 0, 0, 0),
                                panel_rank,
                                &block[0]);

                            // 1c. TVAVT = T^H (V^H AVT)
                            auto T0    = Tlocal.sub(i0, i0, k, k);
                            auto TVAVT0  = W.sub(0, 0, 0, 0);

                            int64_t mb = T0.tileMb(0);
                            int64_t nb = T0.tileNb(0);
                            bool trapezoid = (mb < nb);

                            if (trapezoid) {
                                T0     = T0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                                TVAVT0 = TVAVT0.slice(0, mb-1, 0, nb-1); // first mb-by-nb part
                            }

                            auto Tk0 = TriangularMatrix<scalar_t>(Uplo::Upper,
                                                                  Diag::NonUnit, T0);
                            Tk0.tileGetForReading(0, 0, Tk0.hostNum(),
                                                  LayoutConvert(layout));
                            TVAVT0.tileGetForWriting(0, 0, TVAVT0.hostNum(),
                                                     LayoutConvert(layout));
                            trmm(Side::Left, Diag::NonUnit,
                                 one, conjTranspose(Tk0(0, 0)),
                                 std::move(TVAVT0(0, 0)));

                            // 1d. W = W - 0.5 V TVAVT.
                            internal::he2hb_gemm<target>(
                                -half, A.sub(k+1, nt-1, k, k),
                                W.sub(0, 0, 0, 0),
                                one,   W.sub(k+1, nt-1, k, k),
                                panel_rank,
                                &block[k+1]);

                            // 2. Update trailing matrix.
                            //  A = -V WT -W VT + A
                            internal::her2k<target>(
                                -one,  A.sub(k+1, nt-1, k, k),
                                W.sub(k+1, nt-1, k, k),
                                1.0,   A.sub(k+1, nt-1));
                        }
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

                        #pragma omp task depend(in:block[k]) \
                                         depend(inout:block[k+1]) \
                                         depend(inout:block[nt-1]) \
                                         depend(inout:alloc_trailing[0])
                        {
                            internal::he2hb_gemm_outer<target>(
                                -one, A.sub(k+1, nt-1, k, k),
                                W.sub(k+1, nt-1, k, k),
                                one,  A.sub(k+1, nt-1),
                                indices_local_panel, &block[k+1]);
                        }
                    }

                    #pragma omp task depend(inout:block[k])
                    {
                        if (A.tileExists(i0, k)) {
                            // Restore V0.
                            #pragma omp task
                            {
                                gecopy(Asave(i0, k), A(i0, k));
                                Asave.tileErase(i0, k);
                            }
                            #pragma omp taskwait
                        }
                    }
                }

                #pragma omp task depend(in:block[k]) \
                                 depend(inout:block[k+1]) \
                                 depend(inout:block[nt-1]) \
                                 depend(inout:alloc_trailing[0])
                {
                    internal::hettmqr<Target::HostTask>(
                        Op::ConjTrans,
                        std::move(A_panel),
                        std::move(Treduce_panel),
                        A.sub(k+1, nt-1));
                }

                // Unhold and release tiles in A_panel and Tlocal
                if (target == Target::Devices) {
                    if ( k < nt-1) {
                        #pragma omp task depend(in:block[k])
                        {
                            for (int64_t i = k; i < nt; ++i) {
                                if (A.tileIsLocal(i, k)) {
                                    A.tileUpdateOrigin(i, k);

                                    std::set<int> dev_set;
                                    A.sub(k+1, nt-1).getLocalDevices(&dev_set);

                                    for (auto device : dev_set) {
                                        A.tileUnsetHold(i, k, device);
                                        A.tileRelease(i, k, device);
                                    }
                                }
                            }

                            for (int panel_rank: panel_ranks) {
                                // Find local indices for panel_rank.
                                indices.clear();
                                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                                    if (A_panel.tileRank(i, 0) == panel_rank) {
                                        // global index
                                        indices.push_back(i+k+1);
                                    }
                                }
                                i0 = indices[0];
                                if (indices.size() > 0) {
                                    for (int64_t row : indices) {
                                        if (Tlocal.tileIsLocal(row, k)) {
                                            //Tlocal.tileUpdateOrigin(row, k);

                                            std::set<int> dev_set;
                                            Tlocal.sub(row, row, k+1, nt-1).getLocalDevices(&dev_set);

                                            for (auto device : dev_set) {
                                                Tlocal.tileUnsetHold(i0, k, device);
                                                Tlocal.tileRelease(i0, k, device);
                                            }

                                            std::set<int> dev_set2;
                                            Tlocal.sub(k+1, nt-1, row, row).getLocalDevices(&dev_set);

                                            for (auto device : dev_set2) {
                                                Tlocal.tileUnsetHold(i0, k, device);
                                                Tlocal.tileRelease(i0, k, device);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
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
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );

    int64_t max_panel_threads  = std::max(omp_get_max_threads()/2, 1);
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                        max_panel_threads );

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
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            he2hb<Target::HostTask>(A, T, opts);
            break;
        case Target::HostNest:
            he2hb<Target::HostNest>(A, T, opts);
            break;
        case Target::HostBatch:
            he2hb<Target::HostBatch>(A, T, opts);
            break;
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
