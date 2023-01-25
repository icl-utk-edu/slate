// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/HermitianMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage Hermitian eigenvalue
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

    assert( A.uplo() == Uplo::Lower );  // for now

    // Constants
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    const int priority_one = 1;
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    const scalar_t half = 0.5;
    const real_t r_one  = 1.0;

    // Options
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );

    int64_t max_panel_threads = std::max( omp_get_max_threads()/2, 1 );
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    int64_t nt = A.nt();

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
    const bool set_hold =  1;  // Do tileGetAndHold in the bcast

    int64_t n = A.n();
    int64_t nb_A = A.tileNb( 0 );
    GridOrder grid_order;
    int nprow, npcol, myrow, mycol;
    // For the workspace matrix (W) change the GPU distribution to row cyclic,
    // by doing that, all the GPUs will be busy simultaneously,
    // in particular when we call he2hb_hemm,
    // which will improve the performance.
    A.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    assert( grid_order == GridOrder::Col );  // todo: update for Row

    std::function<int64_t (int64_t j)>
        tileNb = [n, nb_A] (int64_t j) {
            return (j + 1)*nb_A > n ? n%nb_A : nb_A;
        };

    std::function<int (std::tuple<int64_t, int64_t> ij)>
        tileRank = [nprow, npcol]( std::tuple<int64_t, int64_t> ij ) {
            int64_t i = std::get<0>( ij );
            int64_t j = std::get<1>( ij );
            return int( i%nprow + (j%npcol)*nprow );
        };

    int num_devices = blas::get_device_count();
    std::function<int (std::tuple<int64_t, int64_t> ij)>
        tileDevice = [nprow, num_devices]( std::tuple<int64_t, int64_t> ij ) {
            int64_t i = std::get<0>( ij );
            return int( i/nprow )%num_devices;
        };

    // W is like A, but within node the GPU distribution is row-cyclic.
    slate::HermitianMatrix<scalar_t> W(
            Uplo::Lower, n, tileNb, tileRank, tileDevice, A.mpiComm() );

    // Since W( 0, 0 ) is otherwise unused, store TVAVT there.
    W.tileInsert( 0, 0 );
    auto TVAVT = W.sub( 0, 0, 0, 0 );
    int num_queues = 10;

    int my_rank = A.mpiRank();

    // tracks dependencies by block-column.
    std::vector< uint8_t > block_vector( nt + 2 );
    uint8_t* block = block_vector.data();
    uint8_t* alloc_workspace = &block_vector[ nt ];
    uint8_t* fetch_trailing  = &block_vector[ nt+1 ];

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
            std::vector< int64_t > first_indices;
            first_indices.reserve( panel_ranks.size() );
            for (int r : panel_ranks) {
                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                    if (A_panel.tileRank( i, 0 ) == r) {
                        first_indices.push_back( i+k+1 );
                        break;
                    }
                }
            }

            if (k == 0) {
                #pragma omp task depend( inout:alloc_workspace[ 0 ] )
                {
                    if (target == Target::Devices) {
                        A.allocateBatchArrays();
                        A.reserveDeviceWorkspace();
                        W.allocateBatchArrays( 0, num_queues );
                        W.reserveDeviceWorkspace();
                    }
                }
            }

            //--------------------
            // QR of panel
            // local panel factorization
            #pragma omp task depend( inout:block[ k ] )
            {
                internal::geqrf<Target::HostTask>(
                    std::move( A_panel ),
                    std::move( Tlocal_panel ),
                    ib, max_panel_threads, priority_one);

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
                #pragma omp task depend( inout:block[ k ] ) \
                                 depend( in:alloc_workspace[ 0 ] )
                {
                    // Send V across row i & col i for trailing matrix update.
                    BcastListTag bcast_list_V_first;
                    BcastListTag bcast_list_V;
                    for (int64_t i = k; i < nt; ++i) {
                        // Vs need 6 lives.
                        // Vs in first_indices (except top-most one, i == k+1)
                        // need 3 lives: for her2k, gemm_outer, and for hettmqr.
                        if (i > k+1 && std::find( first_indices.begin(), first_indices.end(), i ) != first_indices.end()) {
                            bcast_list_V_first.push_back(
                                { i, k, { A.sub( i, i, k+1, i ),
                                          A.sub( i+1, nt-1, i, i ) }, i } );
                        }
                        else {
                            bcast_list_V.push_back(
                                { i, k, { A.sub( i, i, k+1, i ),
                                          A.sub( i+1, nt-1, i, i ) }, i } );
                        }
                    }
                    A.template listBcastMT<target>( bcast_list_V_first, layout, 5, set_hold );
                    A.template listBcastMT<target>( bcast_list_V, layout, 6, set_hold );

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
                        Treduce.template listBcastMT( bcast_list_T, layout );
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
                        // todo: I think this sets Tlocal with too many lives
                        // -- needs only 1 life per rank, not # tiles.
                        //BcastList bcast_list_T;
                        BcastListTag bcast_list_T;
                        for (int64_t i : panel_rank_rows) {
                            bcast_list_T.push_back(
                                { i0, k, { Tlocal.sub( i, i, k+1, i ),
                                           Tlocal.sub( i+1, nt-1, i, i ) }, i } );
                        }
                        Tlocal.template listBcastMT<target>( bcast_list_T, layout, 1, set_hold );
                    }
                } // task

                //--------------------
                // Computation and data movement overlap:
                // fetch data to be used in he2hb_hemm
                #pragma omp task depend( in:alloc_workspace[ 0 ] ) \
                                 depend( inout:fetch_trailing[ 0 ] )
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
                                #pragma omp task shared( A, W )
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

                                        #pragma omp task default( shared )
                                        {
                                            A.tileGetForReading( A_tiles_set, device, layout_conv );
                                        }
                                        #pragma omp task default( shared )
                                        {
                                            W.tileGetForWriting( W_tiles_set, device, layout_conv );
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

                    #pragma omp task depend( inout:block[ k ] )
                    {
                        if (A.tileExists( i0, k )) {
                            A.tileGetForWriting( i0, k, HostNum, layout_conv );
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
                    #pragma omp task depend( inout:block[ k ] ) \
                                     depend( inout:block[ k+1 ] ) \
                                     depend( inout:block[ nt-1 ] ) \
                                     depend( inout:fetch_trailing[ 0 ] )
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
                            #pragma omp task
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
                                if (rank_lower == my_rank)
                                    neighbor = rank_upper;
                                else if (rank_upper == my_rank)
                                    neighbor = rank_lower;
                                if (neighbor != -1 && neighbor != my_rank) {
                                    Wtmp.tileInsert( i, k );
                                    int tag_i = i;
                                    int tag_i1 = i+1;
                                    if (neighbor < my_rank) {
                                        W.tileGetForWriting( i, k, HostNum,
                                                             layout_conv );
                                        W   .tileSend( i, k, neighbor, tag_i );
                                        Wtmp.tileRecv( i, k, neighbor, layout, tag_i1 );
                                    }
                                    else {
                                        W.tileGetForWriting( i, k, HostNum,
                                                             layout_conv );
                                        Wtmp.tileRecv( i, k, neighbor, layout, tag_i );
                                        W   .tileSend( i, k, neighbor, tag_i1 );
                                    }
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
                            // todo: shouldn't need to set TVAVT = 0 since beta = 0.
                            // todo: on GPU
                            TVAVT.tileGetForWriting( 0, 0, HostNum, layout_conv );
                            TVAVT( 0, 0 ).set( zero );
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
                            Tk0.tileGetForReading( 0, 0, HostNum, layout_conv );
                            TVAVT0.tileGetForWriting( 0, 0, HostNum, layout_conv );
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
                                r_one, A.sub( k+1, nt-1 ));
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
                    #pragma omp task depend( inout:block[ k ] )
                    {
                        if (A.tileExists( i0, k )) {
                            A.tileGetForWriting( i0, k, layout_conv );
                            tile::gecopy( Asave( i0, k ), A( i0, k ) );
                            Asave.tileErase( i0, k );
                        }
                    }
                } // for panel_rank

                //--------------------
                // Update trailing matrix from triangle reductions.
                #pragma omp task depend( in:block[ k ] ) \
                                 depend( inout:block[ k+1 ] ) \
                                 depend( inout:block[ nt-1 ] ) \
                                 depend( inout:fetch_trailing[ 0 ] )
                {
                    // Do 2-sided Hermitian update:
                    // 3. A = Q^H A Q
                    internal::hettmqr<Target::HostTask>(
                        Op::ConjTrans,
                        std::move( A_panel ),
                        std::move( Treduce_panel ),
                        A.sub( k+1, nt-1 ) );
                }

                // Unhold and release tiles in A_panel and Tlocal.
                if (target == Target::Devices) {
                    // todo: inout, right? what prevents this from executing during previous update?
                    #pragma omp task depend( inout:block[ k ] )
                    {
                        for (int64_t i = k; i < nt; ++i) {
                            if (A.tileIsLocal( i, k )) {
                                A.tileUpdateOrigin( i, k );

                                std::set<int> dev_set;
                                A.sub( k+1, nt-1 ).getLocalDevices( &dev_set );

                                for (auto device : dev_set) {
                                    A.tileUnsetHold( i, k, device );
                                    A.tileRelease( i, k, device );
                                }
                            }
                        }

                        for (int panel_rank : panel_ranks) {
                            // Find local row indices for panel_rank.
                            panel_rank_rows.clear();
                            for (int64_t i = 0; i < A_panel.mt(); ++i) {
                                if (A_panel.tileRank( i, 0 ) == panel_rank) {
                                    // global index
                                    panel_rank_rows.push_back( i+k+1 );
                                }
                            }
                            if (panel_rank_rows.size() > 0) {
                                int64_t i0 = panel_rank_rows[ 0 ];
                                for (int64_t i : panel_rank_rows) {
                                    if (Tlocal.tileIsLocal( i, k )) {
                                        //Tlocal.tileUpdateOrigin( i, k );

                                        std::set<int> dev_set;
                                        Tlocal.sub( i, i, k+1, nt-1 ).getLocalDevices( &dev_set );

                                        for (auto device : dev_set) {
                                            Tlocal.tileUnsetHold( i0, k, device );
                                            Tlocal.tileRelease( i0, k, device );
                                        }

                                        std::set<int> dev_set2;
                                        Tlocal.sub( k+1, nt-1, i, i ).getLocalDevices( &dev_set );

                                        for (auto device : dev_set2) {
                                            Tlocal.tileUnsetHold( i0, k, device );
                                            Tlocal.tileRelease( i0, k, device );
                                        }
                                    }
                                }
                            }
                        } // for panel_rank
                    } // task
                } // if devices
            } // if (k < nt-1)
        } // for k

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    } // parallel, master

    A.releaseWorkspace();
    W.releaseWorkspace();
}

} // namespace impl

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
void he2hb(
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::he2hb<Target::HostTask>( A, T, opts );
            break;

        case Target::HostNest:
            impl::he2hb<Target::HostNest>( A, T, opts );
            break;

        case Target::HostBatch:
            impl::he2hb<Target::HostBatch>( A, T, opts );
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
