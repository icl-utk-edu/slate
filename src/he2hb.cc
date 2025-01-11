// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/HermitianMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 2-stage Hermitian eigenvalue
/// decomposition.
/// Generic implementation for any target.
/// Panel computed on host using Host OpenMP task.
///
/// ColMajor layout is assumed
///
/// @ingroup heev_impl
///
template <Target target, typename scalar_t>
void he2hb(
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts )
{
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    slate_assert( A.uplo() == Uplo::Lower );  // for now

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    const scalar_t half = 0.5;
    const real_t r_one  = 1.0;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int batch_size_default = 0;
    const int num_queues = 10;
    const int queue_0 = 0;
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layoutc = LayoutConvert( layout );

    // Options
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );

    int64_t max_panel_threads = std::max( omp_get_max_threads()/2, 1 );
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    int64_t nt = A.nt();
    int mpi_rank = A.mpiRank();

    // todo: TriangularFactors takes Matrix, not BaseMatrix or HermitianMatrix.
    // This abuses the "conversion sub-matrix" constructor to get a Matrix.
    T.clear();
    auto empty = A.emptyLike();
    auto Tlocal = Matrix<scalar_t>( empty, 0, nt-1, 0, nt-1 );
    auto Treduce = Tlocal.emptyLike( ib, 0 );
    T.push_back( Tlocal );
    T.push_back( Treduce );

    // workspace
    auto Wtmp = A.emptyLike();
    auto Asave = A.emptyLike();

    int64_t n = A.n();
    GridOrder grid_order;
    int nprow, npcol, myrow, mycol;
    // For the workspace matrix (W) change the GPU distribution to row cyclic,
    // by doing that, all the GPUs will be busy simultaneously,
    // in particular when we call he2hb_hemm,
    // which will improve the performance.
    A.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    assert( grid_order == GridOrder::Col );  // todo: update for Row

    auto tileNb = A.tileNbFunc();
    auto tileRank = A.tileRankFunc();
    int num_devices = blas::get_device_count();
    auto tileDevice = slate::func::device_1d_grid( GridOrder::Col,
                                                           nprow, num_devices );

    // W is like A, but within node the GPU distribution is row-cyclic.
    slate::HermitianMatrix<scalar_t> W(
            Uplo::Lower, n, tileNb, tileRank, tileDevice, A.mpiComm() );

    // Since W( 0, 0 ) is otherwise unused, store TVAVT there.
    W.tileInsert( 0, 0 );
    auto TVAVT = W.sub( 0, 0, 0, 0 );
    int     panel_device = -1;
    std::vector< scalar_t* > dwork_array( num_devices, nullptr );
    size_t  work_size    = 0;
    using device_info_t = lapack::device_info_int;

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays( batch_size_default, num_queues );
        W.reserveDeviceWorkspace();

        // Find largest panel size and device for copying to
        // contiguous memory within internal geqrf routine
        int64_t mlocal = 0;
        int64_t first_panel_seen = -1;
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j+1; i < A.mt(); ++i) {
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
                    }
                }
            }
            if (first_panel_seen >= 0) {
                break;
            }
        }

        if (panel_device >= 0) {

            lapack::Queue* queue = A.compute_queue( panel_device, queue_0 );

            int64_t nb       = func::max_blocksize(A.nt(), A.tileNbFunc());
            size_t  size_tau = (size_t) std::min( mlocal, nb );
            size_t  size_A   = (size_t) blas::max( 1, mlocal ) * nb;
            size_t  hsize, dsize;

            // Find size of the workspace needed
            lapack::geqrf_work_size_bytes( mlocal, nb, dwork_array[0], mlocal,
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

    // tracks dependencies by block-column.
    std::vector< uint8_t > block_vector( nt + 2 );
    uint8_t* block = block_vector.data();
    uint8_t* fetch_trailing = &block_vector[ nt+1 ];
    SLATE_UNUSED( block ); // Used only by OpenMP
    SLATE_UNUSED( fetch_trailing ); // Used only by OpenMP

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < nt-1; ++k) {
            //----------------------------------------
            // Q panel and update.
            auto       A_panel =       A.sub( k+1, nt-1, k, k );
            auto  Tlocal_panel =  Tlocal.sub( k+1, nt-1, k, k );
            auto Treduce_panel = Treduce.sub( k+1, nt-1, k, k );

            // Find ranks in this column.
            std::set<int> panel_ranks;
            A_panel.getRanks( &panel_ranks );
            assert( panel_ranks.size() > 0 );

            // Find each rank's first (top-most) row in this panel,
            // where the triangular tile resulting from local geqrf panel
            // will reside.
            std::vector< int64_t > first_indices
                          = internal::geqrf_compute_first_indices(A_panel, k+1);

            //--------------------
            // QR of panel
            // local panel factorization
            #pragma omp task slate_omp_default_none \
                depend( inout:block[ k ] ) \
                shared( dwork_array ) \
                firstprivate(  A_panel, Tlocal_panel, Treduce_panel, ib, \
                               max_panel_threads, work_size, priority_1 )
            {
                internal::geqrf<target>(
                    std::move( A_panel ),
                    std::move( Tlocal_panel ),
                    dwork_array, work_size,
                    ib, max_panel_threads, priority_1 );

                // triangle-triangle reductions
                // ttqrt handles tile transfers internally
                internal::ttqrt<Target::HostTask>(
                    std::move( A_panel ),
                    std::move( Treduce_panel ) );
            }

            // if trailing matrix exists.
            if (k < nt-1) {
                //--------------------
                // Bcast V, Treduce, Tlocal.
                #pragma omp task slate_omp_default_none \
                    depend( inout:block[ k ] ) \
                    shared( A, Treduce, Tlocal ) \
                    firstprivate( A_panel, k, nt, panel_ranks, first_indices, \
                                  layout )
                {
                    // Send V across row i & col i for trailing matrix update.
                    BcastListTag bcast_list_V;
                    for (int64_t i = k; i < nt; ++i) {
                        bcast_list_V.push_back(
                            { i, k, { A.sub( i, i, k+1, i ),
                                      A.sub( i+1, nt-1, i, i ) }, i } );
                    }
                    A.template listBcastMT<target>(
                        bcast_list_V, layout );

                    if (first_indices.size() > 1) {
                        //BcastList bcast_list_T;
                        BcastListTag bcast_list_T;
                        for (int64_t i : first_indices) {
                            // Exclude first row of this panel,
                            // which doesn't have Treduce tile.
                            if (i > k+1) {
                                bcast_list_T.push_back(
                                    { i, k, { Treduce.sub( i, i, k+1, i ),
                                              Treduce.sub( i+1, nt-1, i, i ) },
                                      i } );
                            }
                        }
                        Treduce.template listBcastMT<>( bcast_list_T, layout );
                    }

                    std::vector<int64_t> panel_rank_rows;
                    for (int panel_rank : panel_ranks) {
                        // Find local row indices for panel_rank.
                        panel_rank_rows.clear();
                        for (int64_t i = 0; i < A_panel.mt(); ++i) {
                            if (A_panel.tileRank( i, 0 ) == panel_rank) {
                                // global index
                                panel_rank_rows.push_back( i+k+1 );
                            }
                        }
                        int64_t i0 = panel_rank_rows[ 0 ];

                        // Send Tlocal across row i & col i for trailing matrix update
                        // TODO review that this is the right set of processes to send to
                        BcastListTag bcast_list_T;
                        for (int64_t i : panel_rank_rows) {
                            bcast_list_T.push_back(
                                { i0, k, { Tlocal.sub( i, i, k+1, i ),
                                           Tlocal.sub( i+1, nt-1, i, i ) }, i } );
                        }
                        Tlocal.template listBcastMT<target>(
                            bcast_list_T, layout );
                    }
                } // task

                //--------------------
                // Computation and data movement overlap:
                // fetch data to be used in he2hb_hemm
                #pragma omp task slate_omp_default_none \
                    depend( inout:fetch_trailing[ 0 ] ) \
                    shared( A, W ) \
                    firstprivate( zero, A_panel, k, nt, panel_ranks, layoutc )
                {
                    // todo: insert and set on device?
                    // todo: do we need entire column, or subset? needs W[ my_rows + my_cols ].
                    // todo: is this getting erased anywhere?
                    for (int64_t i = k+1; i < nt; ++i) {
                        W.tileInsert( i, k );
                        W( i, k ).set( zero );
                    }

                    if (target == Target::Devices) {
                        std::vector<int64_t> panel_rank_rows;
                        auto  W_panel = W.sub( k+1, nt-1, k, k );
                        for (int panel_rank : panel_ranks) {
                            // Find local row indices for panel_rank.
                            panel_rank_rows.clear();
                            for (int64_t i = 0; i < A_panel.mt(); ++i) {
                                if (A_panel.tileRank( i, 0 ) == panel_rank) {
                                    // global index
                                    panel_rank_rows.push_back( i+k+1 );
                                }
                            }
                            // Move lower( A( :, panel_rank_rows ) )
                            // and  lower( A( panel_rank_rows, : ) )
                            // and  W( :, k ) to GPU.
                            for (int device = 0; device < W_panel.num_devices(); ++device) {
                                #pragma omp task slate_omp_default_none \
                                    shared( A, W ) \
                                    firstprivate( k, nt, device, panel_rank_rows, \
                                                  layoutc )
                                {
                                    std::set<ij_tuple> A_tiles_set, A_panel_tiles_set, W_tiles_set;
                                    for (int64_t j : panel_rank_rows) {
                                        for (int64_t i = k+1; i < nt; ++i) {
                                            if (i >= j) { // lower or diagonal
                                                // todo: should this check device == A.tileDevice( i, j )?
                                                if (A.tileIsLocal( i, j )
                                                    && device == W.tileDevice( i, k )) {
                                                    A_tiles_set.insert( { i, j } );
                                                    W_tiles_set.insert( { i, k } );
                                                }
                                            }
                                            else { // upper
                                                // todo: should this check device == A.tileDevice( j, i )?
                                                if (A.tileIsLocal( j, i )
                                                    && device == W.tileDevice( i, k )) {
                                                    A_tiles_set.insert( { j, i } );
                                                    W_tiles_set.insert( { i, k } );
                                                }
                                            }
                                        }

                                        #pragma omp task slate_omp_default_none \
                                            shared( A, A_tiles_set ) \
                                            firstprivate( device, layoutc )
                                        {
                                            A.tileGetForReading( A_tiles_set, device, layoutc );
                                        }
                                        #pragma omp task slate_omp_default_none \
                                            shared( W, W_tiles_set ) \
                                            firstprivate( device, layoutc )
                                        {
                                            W.tileGetForWriting( W_tiles_set, device, layoutc );
                                        }
                                        #pragma omp taskwait
                                    }
                                } // task
                            } // for device
                        } // for panel_rank
                    } // if devices
                    #pragma omp taskwait
                } // task

                //----------------------------------------
                // QR update trailing submatrix.
                std::vector<int64_t> panel_rank_rows, panel_rank_rows_sub;
                for (int panel_rank : panel_ranks) {
                    // Find local row indices for panel_rank.
                    panel_rank_rows.clear();
                    panel_rank_rows_sub.clear();
                    for (int64_t i = 0; i < A_panel.mt(); ++i) {
                        if (A_panel.tileRank( i, 0 ) == panel_rank) {
                            // global index
                            panel_rank_rows.push_back( i+k+1 );
                            // index relative to panel sub-matrix.
                            panel_rank_rows_sub.push_back( i );
                        }
                    }
                    int64_t i0 = panel_rank_rows[ 0 ];

                    #pragma omp task slate_omp_default_none \
                        depend( inout:block[ k ] ) \
                        shared( A, Asave ) \
                        firstprivate( zero, one, i0, k, layoutc )
                    {
                        if (A.tileExists( i0, k, AnyDevice )) {
                            A.tileGetForWriting( i0, k, HostNum, layoutc );
                            // Save V0 and set upper(V0) to identity, to avoid trmm's.
                            Asave.tileInsert( i0, k );
                            auto Aik = A( i0, k );
                            tile::gecopy( std::move( Aik ), Asave( i0, k ) );
                            Aik.uplo( Uplo::Upper );
                            Aik.set( zero, one );
                        }
                    }

                    //--------------------
                    // Update trailing submatrix.
                    #pragma omp task slate_omp_default_none \
                        depend( inout:block[ k ] ) \
                        depend( inout:block[ k+1 ] ) \
                        depend( inout:block[ nt-1 ] ) \
                        depend( inout:fetch_trailing[ 0 ] ) \
                        shared( A, W, Wtmp, Tlocal, TVAVT ) \
                        firstprivate( zero, half, one, r_one, i0, k, nt, \
                                      panel_rank, panel_rank_rows, \
                                      panel_rank_rows_sub, mpi_rank, \
                                      layout, layoutc, priority_0, queue_0 )
                    {
                        // Compute W = A V T.
                        // 1a. Wi_part = sum_j Aij Vj, local partial sum,
                        // for i = k+1, ..., nt-1 and j = panel_rank_rows.
                        internal::he2hb_hemm<target>(
                            A.sub( k+1, nt-1 ),
                            A.sub( k+1, nt-1, k, k ),
                            W.sub( k+1, nt-1, k, k ),
                            panel_rank_rows_sub );

                        // 1b. Wi = Wi_part1 + Wi_part2.
                        // At most 2 ranks contribute to each Wi; if I am one,
                        // exchange partial sum with neighbor and both ranks sum Wi.
                        #pragma omp taskgroup
                        for (int64_t i = k+1; i < nt-1; ++i) {
                            #pragma omp task slate_omp_default_none \
                                shared( A, W, Wtmp, panel_rank_rows ) \
                                firstprivate( one, i, k, mpi_rank, layout, layoutc )
                            {
                                int rank_lower = -1;
                                int rank_upper = -1;
                                for (int64_t j : panel_rank_rows) {
                                    if (i >= j) { // lower
                                        rank_lower = A.tileRank( i, j );
                                    }
                                    else { // upper
                                        rank_upper = A.tileRank( j, i );
                                    }
                                }
                                int neighbor = -1;
                                if (rank_lower == mpi_rank)
                                    neighbor = rank_upper;
                                else if (rank_upper == mpi_rank)
                                    neighbor = rank_lower;
                                if (neighbor != -1 && neighbor != mpi_rank) {
                                    Wtmp.tileInsert( i, k );
                                    int tag_i = i;
                                    int tag_i1 = i+1;
                                    W.tileGetForWriting( i, k, HostNum, layoutc );
                                    MPI_Request req;
                                    if (neighbor < mpi_rank) {
                                        W  .tileIsend( i, k, neighbor, tag_i, &req );
                                        Wtmp.tileRecv( i, k, neighbor, layout, tag_i1 );
                                    }
                                    else {
                                        W  .tileIsend( i, k, neighbor, tag_i1, &req );
                                        Wtmp.tileRecv( i, k, neighbor, layout, tag_i );
                                    }
                                    MPI_Wait( &req, MPI_STATUS_IGNORE );
                                    auto Wtmp_ik = Wtmp( i, k );
                                    auto W_ik = W( i, k );
                                    blas::axpy( W_ik.nb()*W_ik.nb(),
                                                one, Wtmp_ik.data(), 1,
                                                        W_ik.data(), 1 );
                                    Wtmp.tileErase( i, k );
                                }
                            }
                        }

                        // 1c. Compute Wi = Wi T, for i = k+1, ..., nt-1.
                        internal::he2hb_trmm<target>(
                            A.sub( k+1, nt-1 ), // Needed to get the rank
                            Tlocal.sub( i0, i0, k, k ),
                            W.sub( k+1, nt-1, k, k ),
                            panel_rank_rows_sub );

                        if (A.tileIsLocal( i0, i0 )) {
                            //--------------------
                            // This rank has diagonal tiles to update.
                            // Do 2-sided Hermitian update:
                            // A = Q^H A Q
                            //   = (I - V T^H V^H) A (I - V T V^H)
                            //   = A - V Y^H - Y V^H
                            // where
                            // Y = A V T - 0.5 V (T^H V^H (A V T))
                            //   = W - 0.5 V (T^H V^H W),
                            // W = A V T from above.

                            // 1d. TVAVT = V^H (A V T) = V^H W.
                            // todo: potentially do gemm+reduce here (block inner-product)
                            internal::he2hb_gemm<target>(
                                one,  conj_transpose( A.sub( k+1, nt-1, k, k ) ),
                                      W.sub( k+1, nt-1, k, k ),
                                zero, std::move( TVAVT ),
                                panel_rank );

                            // 1e. TVAVT = T^H (V^H A V T).
                            auto T0     = Tlocal.sub( i0, i0, k, k );
                            auto TVAVT0 = TVAVT;

                            int64_t mb = T0.tileMb( 0 );
                            int64_t nb = T0.tileNb( 0 );
                            bool trapezoid = (mb < nb);
                            if (trapezoid) {
                                // first mb-by-mb part
                                T0 = T0.slice( 0, mb-1, 0, mb-1 );
                                // first mb-by-nb part
                                TVAVT0 = TVAVT0.slice( 0, mb-1, 0, nb-1 );
                            }

                            // todo: move to GPU
                            auto Tk0 = TriangularMatrix<scalar_t>(
                                Uplo::Upper, Diag::NonUnit, T0 );
                            Tk0.tileGetForReading( 0, 0, HostNum, layoutc );
                            TVAVT0.tileGetForWriting( 0, 0, HostNum, layoutc );
                            tile::trmm( Side::Left, Diag::NonUnit,
                                        one, conj_transpose( Tk0( 0, 0 ) ),
                                             std::move( TVAVT0( 0, 0 ) ) );

                            // 1f. Y = W - 0.5 V TVAVT, with Y in W.
                            internal::he2hb_gemm<target>(
                                -half, A.sub( k+1, nt-1, k, k ),
                                       std::move( TVAVT ),
                                one,   W.sub( k+1, nt-1, k, k ),
                                panel_rank );

                            // 2. Update trailing matrix.
                            // A = A - V Y^H - Y V^H, with Y in W.
                            internal::her2k<target>(
                                -one,  A.sub( k+1, nt-1, k, k ),
                                       W.sub( k+1, nt-1, k, k ),
                                r_one, A.sub( k+1, nt-1 ),
                                priority_0, queue_0, layout );
                        }
                        else { // off-diag
                            //--------------------
                            // This rank has only off-diagonal tiles to update (if any).
                            // Update from left tiles in lower( A( panel_rank_rows, : ) ):
                            // A = Q^H A
                            //   = (I - V T^H V^H) A = A - V T^H V^H A
                            //   = A - V W^H
                            // Update from right tiles in lower( A( :, panel_rank_rows ) ):
                            // A = A Q
                            //   = A (I - V T V^H)   = A - A V T V^H
                            //   = A - W V^H
                            // where
                            // W = A V T from above.

                            // 2. Update trailing matrix.
                            // A = A - V W^H or A - W V^H.
                            internal::he2hb_her2k_offdiag_ranks<target>(
                                -one, A.sub( k+1, nt-1, k, k ),
                                      W.sub( k+1, nt-1, k, k ),
                                one,  A.sub( k+1, nt-1 ),
                                panel_rank_rows_sub );
                        }
                    }

                    // Restore V0.
                    #pragma omp task slate_omp_default_none \
                        depend( inout:block[ k ] ) \
                        shared( A, Asave ) \
                        firstprivate( i0, k, layoutc )
                    {
                        if (A.tileExists( i0, k, AnyDevice )) {
                            A.tileGetForWriting( i0, k, layoutc );
                            tile::gecopy( Asave( i0, k ), A( i0, k ) );
                            Asave.tileErase( i0, k );
                        }
                    }
                } // for panel_rank

                //--------------------
                // Update trailing matrix from triangle reductions.
                #pragma omp task slate_omp_default_none \
                    depend( in:block[ k ] ) \
                    depend( inout:block[ k+1 ] ) \
                    depend( inout:block[ nt-1 ] ) \
                    depend( inout:fetch_trailing[ 0 ] ) \
                    shared( A ) \
                    firstprivate( A_panel, Treduce_panel, k, nt )
                {
                    int tag_base = A.mt()*A.mt();
                    // Do 2-sided Hermitian update:
                    // 3. A = Q^H A Q
                    internal::hettmqr<Target::HostTask>(
                        Op::ConjTrans,
                        std::move( A_panel ),
                        std::move( Treduce_panel ),
                        A.sub( k+1, nt-1 ),
                        tag_base );
                }
            }

            // Release workspace tiles
            #pragma omp task slate_omp_default_none \
                depend( inout:block[ k ] ) \
                firstprivate( A_panel, Tlocal, Treduce ) \
                firstprivate( k, nt, first_indices )
            {
                // Ensure the origin is up to date, then remove the panel's workspace
                A_panel.tileUpdateAllOrigin();
                A_panel.releaseLocalWorkspace();
                A_panel.releaseRemoteWorkspace();

                for (int64_t i : first_indices) {
                    if (Tlocal.tileIsLocal( i, k )) {
                        // Tlocal and Treduce have the same process distribution
                        Tlocal.tileUpdateOrigin( i, k );
                        Tlocal.releaseLocalWorkspaceTile( i, k );
                        if (i != k+1) {
                            // i == k is the root of the reduction tree
                            // Treduce( k, k ) isn't allocated
                            Treduce.tileUpdateOrigin( i, k );
                            Treduce.releaseLocalWorkspaceTile( i, k );
                        }
                    }
                    else {
                        Tlocal.releaseRemoteWorkspaceTile( i, k );
                        Treduce.releaseRemoteWorkspaceTile( i, k );
                    }
                }
            }
        } // for k

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    } // parallel, master

    A.releaseWorkspace();
    W.releaseWorkspace();

    if (target == Target::Devices) {
        for (int64_t dev = 0; dev < num_devices; ++dev) {
            blas::Queue* queue = A.compute_queue( dev, queue_0 );

            blas::device_free( dwork_array[dev], *queue );
            dwork_array[dev] = nullptr;
        }
    }
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 2-stage SVD.
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
///       - HostNest:  not implemented.
///       - HostBatch: not implemented.
///       - Devices:   batched BLAS on GPU device.
///     Note a lookahead is not possible with he2hb due to dependencies from
///     updating on both left and right sides.
///
/// @ingroup heev_computational
///
template <typename scalar_t>
void he2hb(
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    // HostNest and HostBatch not implemented; use HostTask.
    switch (target) {
        case Target::Host:
        case Target::HostTask:
        case Target::HostNest:
        case Target::HostBatch:
            impl::he2hb<Target::HostTask>( A, T, opts );
            break;

        case Target::Devices:
            impl::he2hb<Target::Devices>( A, T, opts );
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
