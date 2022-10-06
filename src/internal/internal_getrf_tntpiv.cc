// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_getrf.hh"
#include "internal/Tile_getrf_tntpiv.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
/// Convert pivot rows (i.e., permutation of 0, ..., m-1) to sequence of
/// row-swaps to be applied to a matrix (i.e., LAPACK-style sequential
/// pivots), for m = mt * mb rows.
///
/// @param[in,out] aux_pivot
///     On entry, permutation of 0, ..., m-1 formed by diag_len swaps
///     during panel factorization, in (tile index, offset) format.
///     Actually, only first diag_len entries are accessed.
///     On exit, first diag_len entries are LAPACK-style sequential
///     pivots, in (tile index, offset) format.
///
/// @param[in] diag_len
///     Length of the diagonal, min( mb, nb ) of diagonal tile.
///
/// @param[in] mt
///     Number of block-rows in panel.
///
/// @param[in] mb
///     Number of rows in each tile.
///
template <typename scalar_t>
void permutation_to_sequential_pivot(
    std::vector< AuxPivot< scalar_t > >& aux_pivot,
    int64_t diag_len, int mt, int64_t mb)
{
    struct TileOffset {
        int64_t index;  // tile index
        int64_t offset; // offset within tile
    };
    std::vector< TileOffset > pivot_list;
    pivot_list.reserve( mt * mb );

    // pivot_list = [
    //      ( 0,    0 ), ..., ( 0,    diag_len - 1 ),   // first tile
    //      ...
    //      ( mt-1, 0 ), ..., ( mt-1, diag_len - 1 )    // last tile
    // ]
    for (int64_t i = 0; i < mt; ++i) {
        for (int64_t ii = 0; ii < diag_len; ++ii) {
            pivot_list.push_back( { i, ii } );
        }
    }

    // Search for where row ii was pivoted to as row iip. Often, iip
    // will simply be aux_pivot[ ii ], which is why we start at k = ii.
    // Otherwise, it's at a previous pivot, so count down k = ii, ..., 0.
    // Example:
    //
    // aux_pivot with mt = 2, nb = 4
    // on entry    on exit
    // ( 1, 1 )    ( 1, 1 )
    // ( 0, 3 )    ( 0, 3 )
    // ( 0, 0 )    ( 1, 1 )
    // ( 0, 2 )    ( 1, 1 )
    //
    // ( 1, 0 )    ( .... )  unchanged and ignored
    // ( 0, 1 )    ( .... )
    // ( 1, 2 )    ( .... )
    // ( 1, 3 )    ( .... )
    //
    // iip  pivot_list  ii = 0      ii = 1      ii = 2      ii = 3
    // 0    ( 0, 0 )-.  ( .... )    ( .... )    ( .... )    ( .... )
    // 1    ( 0, 1 ) |  ( 0, 1 )-.  ( .... )    ( .... )    ( .... )
    // 2    ( 0, 2 ) |  ( 0, 2 ) |  ( 0, 2 )-.  ( .... )    ( .... )
    // 3    ( 0, 3 ) |  ( 0, 3 )==> ( 0, 1 ) |  ( 0, 1 )-.  ( .... )
    //               |                       |           |
    // 4    ( 1, 0 ) |  ( 1, 0 )    ( 1, 0 ) |  ( 1, 0 ) |  ( 1, 0 )
    // 5    ( 1, 1 )==> ( 0, 0 )    ( 0, 0 )==> ( 0, 2 )==> ( 0, 1 )
    // 6    ( 1, 2 )    ( 1, 2 )    ( 1, 2 )    ( 1, 2 )    ( 1, 2 )
    // 7    ( 1, 3 )    ( 1, 3 )    ( 1, 3 )    ( 1, 3 )    ( 1, 3 )
    //
    // ii = 0: found aux_pivot[ 0 ] = ( 1, 1 ) at k = 0, iip = 5 =~ ( 1, 1 )
    // ii = 1: found aux_pivot[ 1 ] = ( 0, 3 ) at k = 1, iip = 3 =~ ( 0, 3 )
    // ii = 2: found aux_pivot[ 2 ] = ( 0, 0 ) at k = 0, iip = 5 =~ ( 1, 1 )
    // ii = 3: found aux_pivot[ 3 ] = ( 0, 1 ) at k = 2, iip = 5 =~ ( 1, 1 )

    for (int64_t ii = 0; ii < diag_len; ++ii) {
        int64_t iip = -1;
        for (int k = ii; k >= 0; --k) {
            int64_t iip_ = aux_pivot[ k ].tileIndex() * mb
                         + aux_pivot[ k ].elementOffset();
            // Pivot row iip_ must be at or below row ii.
            if (iip_ >= ii
                && pivot_list[ iip_ ].index  == aux_pivot[ ii ].tileIndex()
                && pivot_list[ iip_ ].offset == aux_pivot[ ii ].elementOffset()) {
                iip = iip_;
                break;
            }
        }
        assert( iip >= 0 );

        // Save iip in (tile index, offset) format as the sequential pivot.
        aux_pivot[ ii ].set_tileIndex(     iip / mb );
        aux_pivot[ ii ].set_elementOffset( iip % mb );

        // Copy pivot from ii to iip.
        // (Could swap, but pivot_list[ ii ] is never accessed again.)
        pivot_list[ iip ] = pivot_list[ ii ];
    }
}

//------------------------------------------------------------------------------
/// Multi-threaded LU factorization of local tiles.
///
/// @params[in,out] tiles
///     List of tiles to factor.
///
/// @params[in] diag_len
///     Length of diagonal, min( mb, nb ) of diagonal tile.
///
/// @params[in] ib
///     Inner blocking.
///
/// @params[in] stage
///     Stage = 0 is initial local tiles,
///     stage = 1 is subsequent tournament.
///
/// @params[in] nb
///     Block size. (todo: mb, nb?)
///
/// @params[in] tile_indices
///     Block row indices of tiles in tiles array.
///
/// @params[in] mpi_rank
///     MPI rank of this process.
///
/// @params[in] max_panel_threads
///     Maximum number of threads to launch for local panel.
///
/// @params[in] priority
///     OpenMP priority.
///     todo: unused. Should it be on taskloop?
///
/// @ingroup gesv_internal
///
template <typename scalar_t>
void getrf_tntpiv_local(
    std::vector< Tile< scalar_t > >& tiles,
    int64_t diag_len, int64_t ib, int stage,
    int64_t nb, std::vector<int64_t>& tile_indices,
    std::vector< std::vector< AuxPivot< scalar_t > > >& aux_pivot,
    int mpi_rank, int max_panel_threads, int priority)
{
    // Launch the panel tasks.
    int thread_size = max_panel_threads;
    if (int(tiles.size()) < max_panel_threads)
        thread_size = tiles.size();

    ThreadBarrier thread_barrier;
    std::vector<scalar_t> max_value( thread_size );
    std::vector<int64_t>  max_index( thread_size );
    std::vector<int64_t>  max_offset( thread_size );
    std::vector<scalar_t> top_block( ib * nb );

    #if 1
        omp_set_nested(1);
        // Launching new threads for the panel guarantees progression.
        // This should never deadlock, but may be detrimental to performance.
        #pragma omp parallel for \
                    num_threads( thread_size ) \
                    shared( thread_barrier, max_value, max_index, max_offset, \
                            top_block, aux_pivot )
    #else
        // Issuing panel operation as tasks may cause a deadlock.
        #pragma omp taskloop \
                    num_tasks( thread_size ) \
                    shared( thread_barrier, max_value, max_index, max_offset, \
                            top_block, aux_pivot )
    #endif
    for (int thread_id = 0; thread_id < thread_size; ++thread_id) {
        // Factor the local panel in parallel.
        tile::getrf_tntpiv_local(
            diag_len, ib, stage,
            tiles, tile_indices,
            aux_pivot,
            mpi_rank,
            thread_id, thread_size,
            thread_barrier,
            max_value, max_index, max_offset, top_block);
    }
    #pragma omp taskwait

}

//------------------------------------------------------------------------------
/// LU factorization of a column of tiles, host implementation.
/// @ingroup gesv_internal
///
template <typename scalar_t>
void getrf_tntpiv_panel(
    internal::TargetType<Target::HostTask>,
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority)
    // todo: missing tag
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    const Layout layout = Layout::ColMajor;

    assert( A.nt() == 1 );

    internal::copy<Target::HostTask>( std::move( A ), std::move( Awork ) );

    // Move the panel to the host.
    std::set<ij_tuple> A_tiles_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, 0 )) {
            A_tiles_set.insert( { i, 0 } );
        }
    }
    A.tileGetForWriting( A_tiles_set, LayoutConvert::ColMajor );

    // lists of local tiles, indices, and offsets
    std::vector< Tile<scalar_t> > tiles, tiles_copy_poriginal;
    std::vector<int64_t> tile_indices;

    // Build the set of ranks in the panel.
    // Build lists of local tiles and their indices.
    std::set<int> ranks_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        ranks_set.insert( A.tileRank( i, 0 ) );
        if (A.tileIsLocal( i, 0 )) {
            tiles.push_back( Awork( i, 0 ) );
            tile_indices.push_back( i );
        }
    }
    // Find each rank's first (top-most) row in this panel.
    std::vector< std::pair<int, int64_t> > rank_rows;
    rank_rows.reserve( ranks_set.size() );
    for (int r : ranks_set) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileRank( i, 0 ) == r) {
                rank_rows.push_back( { r, i } );
                break;
            }
        }
    }

    // Sort rank_rows by row.
    std::sort( rank_rows.begin(), rank_rows.end(), compareSecond<int, int64_t> );

    // Find index of first tile on this rank.
    int index;
    for (index = 0; index < int(rank_rows.size()); ++index) {
        if (rank_rows[ index ].first == A.mpiRank())
            break;
    }

    // If participating in the panel factorization.
    if (index < int(rank_rows.size())) {

        int nranks = rank_rows.size();
        int nlevels = int( ceil( log2( nranks ) ) );

        std::vector< std::vector< AuxPivot< scalar_t > > > aux_pivot( 2 );
        aux_pivot[ 0 ].resize( A.tileMb( 0 ) );
        aux_pivot[ 1 ].resize( A.tileMb( 0 ) );

        int piv_len = std::min(tiles[0].mb(), tiles[0].nb());

        // Factor the panel locally in parallel, for stage = 0.
        getrf_tntpiv_local(
            tiles, piv_len, ib, 0,
                     A.tileNb(0), tile_indices, aux_pivot,
                     A.mpiRank(), max_panel_threads, priority);

        if (nranks > 1) {

            internal::copy<Target::HostTask>( std::move( A ), std::move( Awork ) );

            std::vector< std::vector<std::pair<int, int64_t>> > global_tracking(tile_indices.size());

            for (int i=0; i < int(tile_indices.size()); i++) {
                global_tracking[i].reserve(A.tileMb(0));

                for (int64_t j = 0; j < A.tileMb(0); ++j) {
                    global_tracking[i].push_back({tile_indices[i], j});
                }
            }

            std::pair<int, int64_t> global_pair;

            for (int j=0; j < piv_len; ++j) {
                if (aux_pivot[0][j].localTileIndex() > 0 ||
                    aux_pivot[0][j].localOffset() > j) {

                    swapLocalRow(
                        0, A.tileNb(0),
                        tiles[0], j,
                        tiles[aux_pivot[0][j].localTileIndex()],
                        aux_pivot[0][j].localOffset());

                    int index2 = aux_pivot[0][j].localTileIndex();
                    int offset = aux_pivot[0][j].localOffset();

                    global_pair = global_tracking[0][j];
                    global_tracking[0][j] = global_tracking[index2][offset];
                    global_tracking[index2][offset]=global_pair;
                }
            }

            for (int j=0; j < piv_len; ++j) {
                aux_pivot[0][j].set_tileIndex(global_tracking[0][j].first);
                aux_pivot[0][j].set_elementOffset(global_tracking[0][j].second);
            }

            int step =1;
            int src, dst;
            int64_t i_src, i_dst, i_current;

            for (int level = 0; level < nlevels; ++level) {

                if (index % (2*step) == 0) {
                    if (index + step < nranks) {

                        src       = rank_rows[index+step].first;
                        i_dst     = rank_rows[index+step].second;
                        i_current = rank_rows[index].second;

                        Awork.tileRecv(i_dst, 0, src, layout);
                        Awork.tileGetForWriting(i_current, 0, LayoutConvert(layout));

                        MPI_Status status;
                        MPI_Recv(aux_pivot.at(1).data(),
                                 sizeof(AuxPivot<scalar_t>)*aux_pivot.at(1).size(),
                                 MPI_BYTE, src, 0, A.mpiComm(),  &status);

                        //Alocate workspace to copy tiles in the tree reduction.
                        std::vector< Tile<scalar_t> > local_tiles;
                        std::vector<scalar_t> data1( Awork.tileMb(i_current) * Awork.tileNb(0) );
                        std::vector<scalar_t> data2( Awork.tileMb(i_dst) * Awork.tileNb(0) );

                        Tile<scalar_t> tile1( Awork.tileMb(i_current), Awork.tileNb(0),
                                              &data1[0], Awork.tileMb(i_current), slate::HostNum, TileKind::Workspace );
                        Tile<scalar_t> tile2( Awork.tileMb(i_dst), Awork.tileNb(0),
                                              &data2[0], Awork.tileMb(i_dst), slate::HostNum, TileKind::Workspace );

                        local_tiles.push_back( tile1 );
                        local_tiles.push_back( tile2 );

                        Awork(i_current, 0).copyData( &local_tiles[0]);
                        Awork(i_dst, 0).copyData( &local_tiles[1]);

                        piv_len = std::min(local_tiles[0].mb(), local_tiles[0].nb());

                        // Factor the panel locally in parallel.
                        getrf_tntpiv_local(
                            local_tiles, piv_len, ib, 1,
                                     A.tileNb(0), tile_indices, aux_pivot,
                                     A.mpiRank(), max_panel_threads, priority);

                        std::vector< Tile<scalar_t> > ptiles;
                        ptiles.push_back(Awork(i_current, 0));
                        ptiles.push_back(Awork(i_dst, 0));

                        for (int j=0; j < piv_len; ++j) {
                            if (aux_pivot[0][j].localTileIndex() > 0 ||
                                aux_pivot[0][j].localOffset() > j) {

                                swapLocalRow(
                                    0, A.tileNb(0),
                                    ptiles[0], j,
                                    ptiles[aux_pivot[0][j].localTileIndex()],
                                    aux_pivot[0][j].localOffset());
                            }
                        }
                        if (level == nlevels-1) {
                            // Copy the last factorization back to panel tile
                            local_tiles[ 0 ].copyData( &ptiles[ 0 ] );
                            permutation_to_sequential_pivot(
                                aux_pivot[ 0 ], diag_len, A.mt(), A.tileMb( 0 ) );
                        }

                        Awork.tileTick(i_dst, 0);
                    }
                }
                else {
                    dst   = rank_rows[ index - step ].first;
                    i_src = rank_rows[ index ].second;
                    Awork.tileSend(i_src, 0, dst);

                    MPI_Send(aux_pivot.at(0).data(),
                             sizeof(AuxPivot<scalar_t>)*aux_pivot.at(0).size(),
                             MPI_BYTE, dst, 0, A.mpiComm());
                    break;
                }
                step *= 2;
            } // for loop over levels
        }

        // Copy pivot information from aux_pivot to pivot.
        for (int64_t i = 0; i < diag_len; ++i) {
            pivot[ i ] = Pivot( aux_pivot[ 0 ][ i ].tileIndex(),
                                aux_pivot[ 0 ][ i ].elementOffset() );
        }
    }
}

//------------------------------------------------------------------------------
/// LU factorization of a column of tiles.
/// Dispatches to target implementations.
/// @ingroup gesv_internal
///
template <Target target, typename scalar_t>
void getrf_tntpiv_panel(
    Matrix<scalar_t>&& A,
    Matrix<scalar_t>&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority)
{
    getrf_tntpiv_panel(
        internal::TargetType<target>(),
        A, Awork, diag_len, ib, pivot, max_panel_threads, priority );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostTask, float>(
    Matrix<float>&& A,
    Matrix<float>&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostTask, double>(
    Matrix<double>&& A,
    Matrix<double>&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

} // namespace internal

} // namespace slate
