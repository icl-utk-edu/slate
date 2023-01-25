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
#include "lapack.hh"
#include "lapack/device.hh"
#include "blas/device.hh"

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
/// @params[in] target
///     Target for dispacth to correct implementation.
///
/// @params[in,out] tiles
///     List of tiles to factor on the CPU.
///
/// @params[in,out] dwork_array
///     List of tiles to factor on the GPU,
///     includes workspace pivots and info.
///
/// @params[in] work_size
///    Total size in dwork_array available.
///
/// @params[in] mlocal
///    Number of rows in dwork_array.
///
/// @params[in] device
///    Device performing factorization,
///    needed for pointing to correct memory in dwork_array.
///    Device == HostNum for CPU implementation.
///
/// @params[in] queue
///    Queue associated to input device.
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
/// @params[in] mb
///     Tile row block size.
///
/// @params[in] nb
///     Tile column block size.
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
    internal::TargetType<Target::HostTask>,
    std::vector< Tile< scalar_t > >& tiles,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int mlocal, int device, lapack::Queue* queue,
    int64_t diag_len, int64_t ib, int stage, int64_t mb,
    int64_t nb, std::vector<int64_t>& tile_indices,
    std::vector< std::vector< AuxPivot< scalar_t > > >& aux_pivot,
    int mpi_rank, int max_panel_threads, int priority)
{
    // Launch the panel tasks.
    int thread_size = std::min( max_panel_threads, int( tiles.size() ) );

    ThreadBarrier thread_barrier;
    std::vector<scalar_t> max_value( thread_size );
    std::vector<int64_t>  max_index( thread_size );
    std::vector<int64_t>  max_offset( thread_size );
    std::vector<scalar_t> top_block( ib * nb );

    #if 1
        // todo: this can be just `omp parallel` (no for). cf. internal_geqrf.cc
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
}

template <typename scalar_t>
void getrf_tntpiv_local(
    internal::TargetType<Target::HostBatch>,
    std::vector< Tile< scalar_t > >& tiles,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int mlocal, int device, lapack::Queue* queue,
    int64_t diag_len, int64_t ib, int stage, int64_t mb,
    int64_t nb, std::vector<int64_t>& tile_indices,
    std::vector< std::vector< AuxPivot< scalar_t > > >& aux_pivot,
    int mpi_rank, int max_panel_threads, int priority)
{
    getrf_tntpiv_local(
        internal::TargetType<Target::HostTask>(),
        tiles, dwork_array, work_size, mlocal, device, queue,
        diag_len, ib, stage, mb, nb, tile_indices, aux_pivot,
        mpi_rank, max_panel_threads, priority );
}

template <typename scalar_t>
void getrf_tntpiv_local(
    internal::TargetType<Target::HostNest>,
    std::vector< Tile< scalar_t > >& tiles,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int mlocal, int device, lapack::Queue* queue,
    int64_t diag_len, int64_t ib, int stage, int64_t mb,
    int64_t nb, std::vector<int64_t>& tile_indices,
    std::vector< std::vector< AuxPivot< scalar_t > > >& aux_pivot,
    int mpi_rank, int max_panel_threads, int priority)
{
    getrf_tntpiv_local(
        internal::TargetType<Target::HostTask>(),
        tiles, dwork_array, work_size, mlocal, device, queue,
        diag_len, ib, stage, mb, nb, tile_indices, aux_pivot,
        mpi_rank, max_panel_threads, priority );
}

template <typename scalar_t>
void getrf_tntpiv_local(
    internal::TargetType<Target::Devices>,
    std::vector< Tile< scalar_t > >& tiles,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int mlocal, int device, lapack::Queue* queue,
    int64_t diag_len, int64_t ib, int stage, int64_t mb,
    int64_t nb, std::vector<int64_t>& tile_indices,
    std::vector< std::vector< AuxPivot< scalar_t > > >& aux_pivot,
    int mpi_rank, int max_panel_threads, int priority)
{
    using lapack::device_info_int;
    using lapack::device_pivot_int;

    int64_t size_A = mlocal * nb;
    size_t size_ipiv = (size_t) diag_len;
    size_t dsize, hsize;
    char* hwork = nullptr;

    // Device contiguous memory for lapack::getrf call
    scalar_t* work = dwork_array[ device ];
    scalar_t* dA   = work;

    // Find workspace size, used for reference that input memory is large enough.
    lapack::getrf_work_size_bytes( mlocal, nb,
                                   dA, mlocal, &dsize, &hsize, *queue );

    size_t tot_size = size_A + diag_len + ceildiv(dsize, sizeof(scalar_t))
                      + ceildiv( sizeof(device_info_int), sizeof(scalar_t));

    assert(hsize == 0);
    assert( tot_size <= work_size);

    device_pivot_int* dipiv = (device_pivot_int*) &work[ size_A ];
    void*             dwork = &work[ size_A + diag_len ];
    device_info_int*  dinfo = (device_info_int*) &work[ size_A + diag_len
                                        + ceildiv( dsize, sizeof(scalar_t) )];

    std::vector<char> hwork_vector( hsize );
    hwork = hwork_vector.data();

    std::vector< device_pivot_int > hipiv( size_ipiv );
    std::vector< scalar_t > hdiagu( diag_len );

    // Factor the panel locally in parallel on decice.
    lapack::getrf( mlocal, nb, dA, mlocal, dipiv,
                   dwork, dsize, hwork, hsize, dinfo, *queue );

    blas::device_memcpy<device_pivot_int>( &hipiv[0], dipiv, size_ipiv,
                                   blas::MemcpyKind::Default, *queue);

    device_copy_vector( diag_len, dA, mlocal + 1,
                                  &hdiagu[ 0 ], 1, *queue );

    // Convert device sequential pivots to aux pivots for stage 0
    for (int64_t i = 0; i < diag_len; ++i) {
        aux_pivot[ 0 ][ i ].set_localTileIndex( ( hipiv[ i ] - 1 ) / mb );
        aux_pivot[ 0 ][ i ].set_localOffset( (  hipiv[ i ] - 1 ) % mb );
        aux_pivot[ 0 ][ i ].set_tileIndex(      tile_indices[ aux_pivot[ 0 ][ i ].localTileIndex() ] );
        aux_pivot[ 0 ][ i ].set_elementOffset(  aux_pivot[ 0 ][ i ].localOffset() );
        aux_pivot[ 0 ][ i ].set_value(          hdiagu[ i ] );
    }
}

//------------------------------------------------------------------------------
/// LU factorization of a column of tiles.
/// @ingroup gesv_internal
///
template <Target target, typename scalar_t>
void getrf_tntpiv_panel(
    internal::TargetType<target>,
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& Awork,
    std::vector< scalar_t* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority)
{
    assert( A.nt() == 1 );

    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    const Layout layout = Layout::ColMajor;

    int device = slate::HostNum;
    int64_t nb = A.tileNb( 0 );
    int64_t mb = A.tileMb( 0 );

    // lists of local tiles, indices, and offsets
    std::vector< Tile<scalar_t> > original_tiles;
    std::vector< Tile<scalar_t> > tiles;
    std::vector<int64_t> tile_indices;
    std::set<int> ranks_set;

    int64_t tile_index_zero = -1;
    int64_t mlocal = 0;
    if (target == Target::Devices) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, 0 )) {
                if (tile_index_zero < 0) {
                    tile_index_zero = i;
                    device = A.tileDevice( i, 0 );
                }
                else {
                    // Assuming devices have a 1-D distribution.
                    assert( device == A.tileDevice( i, 0 ) );
                }
                mlocal += A.tileMb( i );
            }
        }
    }

    // Build the set of ranks in the panel.
    // Build lists of local tiles and their indices.
    std::set<ij_tuple> A_tiles_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, 0 )) {
            A_tiles_set.insert( { i, 0 } );
        }
    }
    internal::copy<Target::HostTask>( std::move( A ), std::move( Awork ) );
    A.tileGetForReading( A_tiles_set, device, LayoutConvert::ColMajor );

    // Device contiguous memory for lapack::getrf call
    scalar_t* work = dwork_array[ device ];
    scalar_t* dA   = work;
    int64_t temp_loc = 0;

    lapack::Queue* queue = nullptr;
    if (target == Target::Devices) {
        if (device < 0) {
           return;
        }
        assert(device >= 0);
        queue = A.compute_queue( device, 0 );

        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, 0 )) {
                // Copy A in dA
                Tile Ai0 = A( i, 0, device );
                blas::device_memcpy_2d<scalar_t>(
                        &dA[ temp_loc ], mlocal,
                        Ai0.data(), Ai0.stride(),
                        Ai0.mb(), nb,
                        blas::MemcpyKind::Default, *queue );
                temp_loc += Ai0.mb();
            }
        }
    }

    // Build the set of ranks in the panel.
    // Build lists of local tiles and their indices.
    for (int64_t i = 0; i < A.mt(); ++i) {
        ranks_set.insert( A.tileRank( i, 0 ) );
        if (A.tileIsLocal( i, 0 )) {
            A_tiles_set.insert( { i, 0 } );
            original_tiles.push_back( A( i, 0 ) );
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
    int nranks = rank_rows.size();
    int index;
    for (index = 0; index < nranks; ++index) {
        if (rank_rows[ index ].first == A.mpiRank())
            break;
    }

    // Either we participate, or the tiles should be empty.
    assert( index < nranks
            || (tiles.size() == 0
                && A_tiles_set.size() == 0) );

    // If participating in the panel factorization.
    if (index < nranks) {
        int nlevels = int( ceil( log2( nranks ) ) );

        std::vector< std::vector< AuxPivot< scalar_t > > > aux_pivot( 2 );
        aux_pivot[ 0 ].resize( mb );
        aux_pivot[ 1 ].resize( mb );

        // piv_len can be < diag_len, if a rank's only tile is short.
        int64_t piv_len = std::min( tiles[ 0 ].mb(), nb );

        getrf_tntpiv_local(
            internal::TargetType<target>(),
            tiles, dwork_array, work_size, mlocal, device, queue,
            piv_len, ib, 0, mb, nb, tile_indices, aux_pivot,
            A.mpiRank(), max_panel_threads, priority );

        if (nranks > 1) {
            // For s = tile_indices.size(), permute is an s-length vector
            // of mb-length vectors of pairs (tile_index, offset):
            //     permute = [ [ (0, 0) ... (0, mb-1) ],
            //                 ...,
            //                 [ (s-1, 0), ..., (s-1, mb-1) ] ].
            // containing final positions of rows,
            // vs. sequential row swaps in usual pivot vector.
            // The aggregate size is mlocal.
            std::vector< std::vector< std::pair< int64_t, int64_t > > >
                permute( tile_indices.size() );

            for (int64_t i = 0; i < int64_t( tile_indices.size() ); ++i) {
                permute[ i ].reserve( mb );
                for (int64_t ii = 0; ii < mb; ++ii) {
                    permute[ i ].push_back( { tile_indices[ i ], ii } );
                }
            }

            // Apply swaps to tiles in Awork, and to permute.
            // Swap (tile, row) (i=0, ii) with (ip, iip).
            auto Awork00 = Awork( tile_indices[0], 0 );
            scalar_t* Awork00_data = Awork00.data();
            for (int64_t ii = 0; ii < piv_len; ++ii) {
                int64_t ip  = aux_pivot[ 0 ][ ii ].localTileIndex();
                int64_t iip = aux_pivot[ 0 ][ ii ].localOffset();

                if (ip > 0 || iip > ii) {
                    std::swap( permute[ 0  ][ ii  ],
                               permute[ ip ][ iip ] );
                }
                int64_t isp  =  permute[ 0 ][ ii ].first;
                int64_t iisp =  permute[ 0 ][ ii ].second;

                auto Aisp = A( isp, 0 );
                scalar_t* Aisp_data = Aisp.data();

                blas::copy( nb, &Aisp_data[ iisp ], Aisp.stride(),
                             &Awork00_data[ ii ], Awork00.stride() );
            }

            // Copy nb elements of permute to aux_pivot for first block.
            for (int64_t ii = 0; ii < piv_len; ++ii) {
                aux_pivot[ 0 ][ ii ].set_tileIndex(     permute[ 0 ][ ii ].first  );
                aux_pivot[ 0 ][ ii ].set_elementOffset( permute[ 0 ][ ii ].second );
            }

            int64_t step = 1;
            for (int level = 0; level < nlevels; ++level) {
                if (index % (2*step) == 0) {
                    if (index + step < nranks) {
                        // This is the top, rank1, of the pair;
                        // recv tile from bottom, rank2, and do LU factorization.
                        int rank2  = rank_rows[ index+step ].first;
                        int64_t i2 = rank_rows[ index+step ].second;
                        int64_t i1 = rank_rows[ index ].second;

                        Awork.tileRecv( i2, 0, rank2, layout );
                        Awork.tileGetForWriting( i1, 0, LayoutConvert( layout ));

                        MPI_Status status;
                        MPI_Recv( aux_pivot[ 1 ].data(),
                                  sizeof(AuxPivot<scalar_t>) * aux_pivot[ 1 ].size(),
                                  MPI_BYTE, rank2, 0, A.mpiComm(),  &status );

                        // Alocate workspace to copy tiles in the tree reduction.
                        std::vector<scalar_t> data1( Awork.tileMb( i1 ) * nb );
                        std::vector<scalar_t> data2( Awork.tileMb( i2 ) * nb );

                        Tile<scalar_t> tile1( Awork.tileMb( i1 ), nb,
                                              &data1[ 0 ], Awork.tileMb( i1 ),
                                              slate::HostNum, TileKind::Workspace );
                        Tile<scalar_t> tile2( Awork.tileMb( i2 ), nb,
                                              &data2[ 0 ], Awork.tileMb( i2 ),
                                              slate::HostNum, TileKind::Workspace );

                        Awork( i1, 0 ).copyData( &tile1 );
                        Awork( i2, 0 ).copyData( &tile2 );

                        piv_len = std::min( tile1.mb(), nb );

                        std::vector< Tile< scalar_t > > tmp_tiles;
                        tmp_tiles.push_back( tile1 );
                        tmp_tiles.push_back( tile2 );

                        // Factor the panel locally in parallel.
                        getrf_tntpiv_local(
                            internal::TargetType<Target::HostTask>(),
                            tmp_tiles, dwork_array, work_size, mlocal, device,
                            queue, piv_len, ib, 1, mb, nb, tile_indices,
                            aux_pivot, A.mpiRank(), max_panel_threads, priority );

                        std::vector< Tile< scalar_t > > work_tiles;
                        work_tiles.push_back( Awork( i1, 0 ) );
                        work_tiles.push_back( Awork( i2, 0 ) );

                        // Swap rows in tiles in Awork.
                        // Swap (tile, row) (0, ii) and (ip, iip).

                        for (int64_t ii = 0; ii < piv_len; ++ii) {
                            int64_t ip  = aux_pivot[ 0 ][ ii ].localTileIndex();
                            int64_t iip = aux_pivot[ 0 ][ ii ].localOffset();
                            if (ip > 0 || iip > ii) {
                                swapLocalRow(
                                    0, nb,
                                    work_tiles[ 0  ], ii,
                                    work_tiles[ ip ], iip );
                            }
                        }
                        if (level == nlevels-1) {
                            // Copy the last factorization back to panel tile
                            tile1.copyData( &work_tiles[ 0 ] );
                            permutation_to_sequential_pivot(
                                aux_pivot[ 0 ], diag_len, A.mt(), mb );
                        }

                        Awork.tileTick( i2, 0 );
                    }
                }
                else {
                    // This is bottom, rank2, of the pair;
                    // send tile i2 and pivot data to top, rank1.
                    int rank1  = rank_rows[ index - step ].first;
                    int64_t i2 = rank_rows[ index ].second;
                    Awork.tileSend( i2, 0, rank1 );

                    MPI_Send( aux_pivot[ 0 ].data(),
                              sizeof(AuxPivot<scalar_t>) * aux_pivot[ 0 ].size(),
                              MPI_BYTE, rank1, 0, A.mpiComm() );

                    // This rank is done!
                    break;
                }
                step *= 2;
            } // for loop over levels
        }
        else {
            if (target == Target::Devices) {
                // Copy from contiguous memory back into workspace.
                // If no reduction is needed do not redo factorization.
                Awork.tileGetForWriting( A_tiles_set, slate::HostNum, LayoutConvert::ColMajor );
                temp_loc = 0;
                for (int64_t i = 0; i < Awork.mt(); ++i) {
                    if (Awork.tileIsLocal( i, 0 )) {
                        Tile Ai0 = Awork( i, 0 );
                        blas::device_memcpy_2d<scalar_t>(
                                Ai0.data(), Ai0.stride(),
                                &dA[ temp_loc ], mlocal,
                                Ai0.mb(), nb,
                                blas::MemcpyKind::Default, *queue );
                        temp_loc += Ai0.mb();
                    }
                }
            }
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
    std::vector< scalar_t* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority)
{
    getrf_tntpiv_panel(
        internal::TargetType<target>(),
        A, Awork, dwork_array, work_size,
        diag_len, ib, pivot, max_panel_threads, priority );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostTask, float>(
    Matrix<float>&& A,
    Matrix<float>&& Awork,
    std::vector< float* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostTask, double>(
    Matrix<double>&& A,
    Matrix<double>&& Awork,
    std::vector< double* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& Awork,
    std::vector< std::complex<float>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& Awork,
    std::vector< std::complex<double>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostNest, float>(
    Matrix<float>&& A,
    Matrix<float>&& Awork,
    std::vector< float* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostNest, double>(
    Matrix<double>&& A,
    Matrix<double>&& Awork,
    std::vector< double* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostNest, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& Awork,
    std::vector< std::complex<float>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostNest, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& Awork,
    std::vector< std::complex<double>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostBatch, float>(
    Matrix<float>&& A,
    Matrix<float>&& Awork,
    std::vector< float* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostBatch, double>(
    Matrix<double>&& A,
    Matrix<double>&& Awork,
    std::vector< double* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostBatch, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& Awork,
    std::vector< std::complex<float>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::HostBatch, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& Awork,
    std::vector< std::complex<double>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::Devices, float>(
    Matrix<float>&& A,
    Matrix<float>&& Awork,
    std::vector< float* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::Devices, double>(
    Matrix<double>&& A,
    Matrix<double>&& Awork,
    std::vector< double* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::Devices, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& Awork,
    std::vector< std::complex<float>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_tntpiv_panel<Target::Devices, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& Awork,
    std::vector< std::complex<double>* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

} // namespace internal

} // namespace slate
