// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 2-stage SVD.
/// Generic implementation for any target.
/// Panel computed on host using Host OpenMP task.
///
/// ColMajor layout is assumed
///
/// @ingroup svd_impl
///
template <Target target, typename scalar_t>
void ge2tb(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& TU,
    TriangularFactors<scalar_t>& TV,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using blas::real;
    using device_info_t = lapack::device_info_int;

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const int queue_0 = 0;

    // Options
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    int64_t max_panel_threads  = std::max(omp_get_max_threads()/2, 1);
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);

    TU.clear();
    TU.push_back(A.emptyLike());
    TU.push_back(A.emptyLike(ib, 0));
    auto TUlocal  = TU[0];
    auto TUreduce = TU[1];

    // Make TVlocal have fixed, square nb-by-nb tiles,
    // and TVreduce have fixed, rectangular ib-by-nb tiles.
    // Otherwise, edge tiles are the wrong size: mb-by-nb instead of nb-by-mb.
    int64_t nb = A.tileNb(0);
    int64_t mb = A.tileMb(0);
    TV.clear();
    TV.push_back(A.emptyLike(nb, nb));
    TV.push_back(A.emptyLike(ib, nb));
    auto TVlocal  = TV[0];
    auto TVreduce = TV[1];
    auto TVlocalT = A.emptyLike(nb, nb, Op::ConjTrans);

    // workspace
    auto W = A.emptyLike();

    int64_t num_devices  = A.num_devices();
    int     panel_device = -1;
    size_t  work_size    = 0;
    std::vector< scalar_t* > dwork_array( num_devices, nullptr );

    int     VTpanel_device = -1;
    std::vector< scalar_t* > VTdwork_array( num_devices, nullptr );

    int panel_VTpanel_device = -1;

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays();
        // todo: this is demanding too much device workspace memory
        // only one tile-row and one tile-col of matrix W per MPI process
        // is going to be used, but W with size of whole A is being allocated
        // thus limiting the matrix size that can be processed
        // For now, allocate workspace tiles 1-by-1.
        //W.reserveDeviceWorkspace();

        // Find largest panel size and device for copying to
        // contiguous memory within internal geqrf routine
        int64_t mlocal = 0;
        int64_t first_panel_seen = -1;
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.mt(); ++i) {
                if (A.tileIsLocal(i, j)) {
                    if (first_panel_seen < 0) {
                        first_panel_seen = j;
                    }
                    if (first_panel_seen == j) {
                        if (panel_device < 0) {
                            panel_device = A.tileDevice(i, j);
                        }
                        mlocal += A.tileMb(i);
                        // Asserting 1-D distribution for device
                        assert( panel_device == A.tileDevice(i, j) );
                        panel_VTpanel_device = panel_device;
                    }
                }
            }
            if (first_panel_seen >= 0) {
                break;
            }
        }

        // Find largest panel size and device for copying to
        // contiguous memory within internal geqrf routine to factorize VT_panel
        int64_t nlocal = 0;
        int64_t first_VTpanel_seen = -1;
        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = i+1; j < A.nt(); ++j) {
                if (A.tileIsLocal(i, j)) {
                    if (first_VTpanel_seen < 0) {
                        first_VTpanel_seen = i;
                    }
                    if (first_VTpanel_seen == i) {
                        if (VTpanel_device < 0) {
                            VTpanel_device = A.tileDevice(i, j);
                        }
                        nlocal += A.tileNb(j);
                        // Asserting 1-D distribution for device
                        //assert( VTpanel_device == A.tileDevice(i, j) );
                        panel_VTpanel_device = VTpanel_device;
                    }
                }
            }
            if (first_VTpanel_seen >= 0) {
                break;
            }
        }

        // Allocate memory to factorize V and VT
        if (panel_VTpanel_device >= 0) {

            lapack::Queue* queue = A.compute_queue( panel_VTpanel_device, queue_0 );

            // Find the max needed allocation
            int64_t mlocal_max = std::max(nlocal, mlocal);
            int64_t nb_max = std::max(nb, mb);
            size_t  size_tau = (size_t) std::min( mlocal_max, nb_max );
            size_t  size_A   = (size_t) blas::max( 1, mlocal_max ) * nb_max;
            size_t  hsize, dsize;

            // Find size of the workspace needed
            lapack::geqrf_work_size_bytes( mlocal_max, nb_max, dwork_array[0], mlocal_max,
                                           &dsize, &hsize, *queue );

            // Size of dA, dtau, dwork and dinfo
            work_size = size_A + size_tau + ceildiv(dsize, sizeof(scalar_t))
                        + ceildiv(sizeof(device_info_t), sizeof(scalar_t));

            for (int64_t dev = 0; dev < num_devices; ++dev) {
                blas::Queue* dev_queue = A.compute_queue( dev, queue_0 );
                dwork_array[dev] =
                  blas::device_malloc<scalar_t>( work_size, *dev_queue );
            }
        }
    }

    // Workspace for transposed panels needs one column of tiles.
    auto AT = A.emptyLike(0, 0, Op::ConjTrans);

    // No lookahead is possible, so no need to track dependencies --
    // just execute tasks in order. Also, priority isn't needed.

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < A_min_mtnt; ++k) {
            //----------------------------------------
            // U panel and update.
            auto U_panel   =        A.sub(k, A_mt-1, k, k);
            auto TUl_panel =  TUlocal.sub(k, A_mt-1, k, k);
            auto TUr_panel = TUreduce.sub(k, A_mt-1, k, k);

            // Find each rank's first (top-most) row in this panel,
            // where the triangular tile resulting from local geqrf panel
            // will reside.
            std::vector< int64_t > first_indices
                            = internal::geqrf_compute_first_indices(U_panel, k);

            //--------------------
            // QR of U panel
            // local panel factorization
            internal::geqrf<target>(
                            std::move(U_panel),
                            std::move(TUl_panel),
                            dwork_array, work_size,
                            ib, max_panel_threads );

            // triangle-triangle reductions
            // ttqrt handles tile transfers internally
            internal::ttqrt<Target::HostTask>(
                            std::move(U_panel),
                            std::move(TUr_panel) );

            //--------------------
            // QR update trailing submatrix.
            if (k+1 < A_nt) {

                // bcast V across row for trailing matrix update
                if (k < A_mt) {
                    BcastList bcast_list_V;
                    for (int64_t i = k; i < A_mt; ++i) {
                        // send A(i, k) across row A(i, k+1:nt-1)
                        bcast_list_V.push_back(
                            {i, k, {A.sub(i, i, k+1, A_nt-1)}});
                    }
                    A.template listBcast<target>(bcast_list_V, layout, 0);
                }

                // bcast TUlocal across row for trailing matrix update
                if (first_indices.size() > 0) {
                    BcastList bcast_list_T;
                    for (int64_t row : first_indices) {
                        bcast_list_T.push_back(
                            {row, k, {TUlocal.sub(row, row, k+1, A_nt-1)}});
                    }
                    TUlocal.template listBcast<>( bcast_list_T, layout );
                }

                // bcast TUreduce across row for trailing matrix update
                if (first_indices.size() > 1) {
                    BcastList bcast_list_T;
                    for (int64_t row : first_indices) {
                        // Exclude first row of this panel,
                        // which doesn't have TUreduce tile.
                        if (row > k) {
                            bcast_list_T.push_back(
                                {row, k, {TUreduce.sub(row, row, k+1, A_nt-1)}});
                        }
                    }
                    TUreduce.template listBcast<>( bcast_list_T, layout );
                }

                int64_t j = k+1;
                auto A_trail_j = A.sub(k, A_mt-1, j, A_nt-1);

                // Apply local reflectors
                internal::unmqr<target>(
                                Side::Left, Op::ConjTrans,
                                std::move(U_panel),
                                std::move(TUl_panel),
                                std::move(A_trail_j),
                                W.sub(k, A_mt-1, j, A_nt-1) );

                // Apply triangle-triangle reduction reflectors
                // ttmqr handles the tile broadcasting internally
                internal::ttmqr<Target::HostTask>(
                                Side::Left, Op::ConjTrans,
                                std::move(U_panel),
                                std::move(TUr_panel),
                                std::move(A_trail_j),
                                j );
            }

            // Can release tiles parallel to the main execution
            #pragma omp task
            {
                // Ensure the origin is up to date, then remove the panel's workspace
                U_panel.tileUpdateAllOrigin();
                U_panel.releaseLocalWorkspace();
                U_panel.releaseRemoteWorkspace();

                for (int64_t i : first_indices) {
                    if (TUlocal.tileIsLocal( i, k )) {
                        // TUlocal and TUreduce have the same process distribution
                        TUlocal.tileUpdateOrigin( i, k );
                        TUlocal.releaseLocalWorkspaceTile( i, k );
                        if (i != k) {
                            // i == k is the root of the reduction tree
                            // TUreduce( k, k ) isn't allocated
                            TUreduce.tileUpdateOrigin( i, k );
                            TUreduce.releaseLocalWorkspaceTile( i, k );
                        }
                    }
                    else {
                        TUlocal.releaseRemoteWorkspaceTile( i, k );
                        TUreduce.releaseRemoteWorkspaceTile( i, k );
                    }
                }
            }

            //----------------------------------------
            // V panel and update.
            if (k+1 < A_nt) {
                auto   V_panel =        A.sub(k, k, k+1, A_nt-1);
                auto TVl_panel =  TVlocal.sub(k, k, k+1, A_nt-1);
                auto TVr_panel = TVreduce.sub(k, k, k+1, A_nt-1);
                // Transposed panels.
                auto   VT_panel =       AT.sub(k+1, A_nt-1, k, k);
                auto TVlT_panel = TVlocalT.sub(k+1, A_nt-1, k, k);

                first_indices = internal::gelqf_compute_first_indices(V_panel, k+1);

                //--------------------
                // LQ of V panel
                //----------
                // Instead of doing LQ of panel, we do QR of the transposed
                // panel, so that the panel is computed in column-major for
                // much better cache efficiency.
                for (int64_t j = 0; j < V_panel.nt(); ++j) {
                    if (V_panel.tileIsLocal(0, j)) {
                        V_panel.tileGetForReading( 0, j, HostNum, LayoutConvert(layout) );
                        VT_panel.tileInsert( j, 0 );
                        VT_panel.tileModified( j, 0, HostNum );
                        tile::deepConjTranspose( V_panel(0, j), VT_panel(j, 0) );
                    }
                }

                // local panel factorization
                internal::geqrf<target>(
                                std::move(VT_panel),
                                std::move(TVlT_panel),
                                dwork_array, work_size,
                                ib, max_panel_threads );

                // Find first local tile, which is triangular factor
                // (T in I - V T^H V^H), and copy it to TVlocal.
                for (int64_t i = 0; i < TVlT_panel.mt(); ++i) {
                    if (TVl_panel.tileIsLocal(0, i)) {
                        TVlT_panel.tileGetForReading( i, 0, HostNum, LayoutConvert(layout) );
                        TVl_panel.tileInsert(0, i);
                        TVl_panel.tileModified( 0, i, HostNum );
                        tile::gecopy( TVlT_panel(i, 0), TVl_panel(0, i) );
                        break;
                    }
                }

                // Copy result back.
                for (int64_t j = 0; j < V_panel.nt(); ++j) {
                    if (V_panel.tileIsLocal(0, j)) {
                        VT_panel.tileGetForReading( j, 0, HostNum, LayoutConvert(layout) );
                        V_panel.tileGetForWriting( 0, j, HostNum, LayoutConvert(layout) );
                        tile::deepConjTranspose( VT_panel(j, 0), V_panel(0, j) );
                        VT_panel.tileErase(j, 0, AllDevices);
                    }
                }
                //----------

                // triangle-triangle reductions
                // ttlqt handles tile transfers internally
                internal::ttlqt<Target::HostTask>(
                                std::move(V_panel),
                                std::move(TVr_panel) );

                //--------------------
                // LQ update trailing submatrix
                if (k+1 < A_mt) {

                    // bcast V down col for trailing matrix update
                    if (k+1 < A_nt) {
                        BcastList bcast_list_V;
                        for (int64_t j = k+1; j < A_nt; ++j) {
                            bcast_list_V.push_back(
                                {k, j, {A.sub(k+1, A_mt-1, j, j)}});
                        }
                        A.template listBcast<target>(bcast_list_V, layout, 0);
                    }

                    // bcast TVlocal down col for trailing matrix update
                    if (first_indices.size() > 0) {
                        BcastList bcast_list_T;
                        for (int64_t col : first_indices) {
                            bcast_list_T.push_back(
                                {k, col, {TVlocal.sub(k+1, A_mt-1, col, col)}});
                        }
                        TVlocal.template listBcast<>( bcast_list_T, layout );
                    }

                    // bcast TVreduce down col for trailing matrix update
                    if (first_indices.size() > 1) {
                        BcastList bcast_list_T;
                        for (int64_t col : first_indices) {
                            // Exclude first col of this panel,
                            // which doesn't have TVreduce tile.
                            if (col > k+1) {
                                bcast_list_T.push_back(
                                    {k, col, {TVreduce.sub(k+1, A_mt-1, col, col)}});
                            }
                        }
                        TVreduce.template listBcast<>( bcast_list_T, layout );
                    }

                    int64_t i = k+1;
                    auto A_trail_i = A.sub(i, A_mt-1, k+1, A_nt-1);

                    // Apply local reflectors
                    internal::unmlq<target>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(V_panel),
                                    std::move(TVl_panel),
                                    std::move(A_trail_i),
                                    W.sub(i, A_mt-1, k+1, A_nt-1) );

                    // Apply triangle-triangle reduction reflectors
                    // ttmlq handles the tile broadcasting internally
                    internal::ttmlq<Target::HostTask>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(V_panel),
                                    std::move(TVr_panel),
                                    std::move(A_trail_i),
                                    i );
                }

                // Can release tiles parallel to the main execution
                #pragma omp task
                {
                    // Ensure the origin is up to date, then remove the panel's workspace
                    V_panel.tileUpdateAllOrigin();
                    V_panel.releaseLocalWorkspace();
                    V_panel.releaseRemoteWorkspace();

                    for (int64_t j : first_indices) {
                        if (TVlocal.tileIsLocal( k, j )) {
                            // TVlocal and TVreduce have the same process distribution
                            TVlocal.tileUpdateOrigin( k, j );
                            TVlocal.releaseLocalWorkspaceTile( k, j );
                            if (j != k+1) {
                                // j == k+1 is the root of the reduction tree
                                // TVreduce( k, k+1 ) isn't allocated
                                TVreduce.tileUpdateOrigin( k, j );
                                TVreduce.releaseLocalWorkspaceTile( k, j );
                            }
                        }
                        else {
                            TVlocal.releaseRemoteWorkspaceTile( k, j );
                            TVreduce.releaseRemoteWorkspaceTile( k, j );
                        }
                    }
                }
            }
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();

    if (target == Target::Devices) {
        for (int64_t dev = 0; dev < num_devices; ++dev) {
            blas::Queue* queue = A.compute_queue( dev, queue_0 );

            blas::device_free( dwork_array[dev], *queue );
            dwork_array[dev] = nullptr;

            blas::device_free( VTdwork_array[dev], *queue );
            VTdwork_array[dev] = nullptr;
        }
    }
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 2-stage SVD.
///
/// Reduces an m-by-n matrix $A$ to band form using unitary transformations.
/// The factorization has the form
/// \[
///     A = U B V^H
/// \]
/// where $U$ and $V$ are unitary.
/// If m >= n, $B$ is upper triangular band with nb superdiagonals;
/// if m <  n, $B$ is lower triangular band with nb subdiagonals.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the m-by-n matrix $A$.
///     On exit:
///     - If m >= n, the elements Aij for j = i, ..., i+nb,
///       represent the upper triangular band matrix B.
///       The elements below the main diagonal, along with TU, represent
///       the unitary matrix $U$ as a product of elementary reflectors.
///       The elements above the nb-th superdiagonal, along with TV, represent
///       the unitary matrix $V$ as a product of elementary reflectors.
///     - If m < n, the elements Aij for j = i-nb, ..., i,
///       represent the lower triangular band matrix B.
///       The elements below the nb-th subdiagonal, along with TU, represent
///       the unitary matrix $U$ as a product of elementary reflectors.
///       The elements above the main diagonal, along with TV, represent
///       the unitary matrix $V$ as a product of elementary reflectors.
///
/// @param[out] TU
///     On exit, triangular matrices of the block reflectors for U.
///
/// @param[out] TV
///     On exit, triangular matrices of the block reflectors for V.
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
///     Note a lookahead is not possible with ge2tb due to dependencies from
///     updating on both left and right sides.
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void ge2tb(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& TU,
    TriangularFactors<scalar_t>& TV,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::ge2tb<Target::HostTask>( A, TU, TV, opts );
            break;

        case Target::HostNest:
            impl::ge2tb<Target::HostNest>( A, TU, TV, opts );
            break;

        case Target::HostBatch:
            impl::ge2tb<Target::HostBatch>( A, TU, TV, opts );
            break;

        case Target::Devices:
            impl::ge2tb<Target::Devices>( A, TU, TV, opts );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void ge2tb<float>(
    Matrix<float>& A,
    TriangularFactors<float>& TU,
    TriangularFactors<float>& TV,
    Options const& opts);

template
void ge2tb<double>(
    Matrix<double>& A,
    TriangularFactors<double>& TU,
    TriangularFactors<double>& TV,
    Options const& opts);

template
void ge2tb< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& TU,
    TriangularFactors< std::complex<float> >& TV,
    Options const& opts);

template
void ge2tb< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& TU,
    TriangularFactors< std::complex<double> >& TV,
    Options const& opts);

} // namespace slate
