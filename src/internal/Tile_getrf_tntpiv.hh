// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GETRF_TNTPIV_HH
#define SLATE_TILE_GETRF_TNTPIV_HH

#include "internal/internal.hh"
#include "internal/internal_swap.hh"
#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"
#include "slate/internal/util.hh"

#include <list>

#include <blas.hh>
#include <lapack.hh>

namespace slate {
namespace tile {

//------------------------------------------------------------------------------
/// Compute the LU factorization of a local panel,
/// for use in CALU tournament pivoting.
///
/// @param[in] diag_len
///     length of the panel diagonal
///
/// @param[in] ib
///     internal blocking in the panel
///
/// @param[in] stage
///     stage of the tree reduction
///
/// @param[in,out] tiles
///     local tiles in the panel
///
/// @param[in] tile_indices
///     i indices of the tiles in the panel
///
/// @param[in,out] aux_pivot
///     pivots produced by the panel factorization, of dimension (2, mb).
///
///     For stage == 0,
///     aux_pivot[ 0 ][ 0:mb-1 ] is used.
///
///     For stage == 1,
///     aux_pivot[ 0 ][ 0:mb-1 ] contains pivot info for tile 0,
///     aux_pivot[ 1 ][ 0:mb-1 ] contains pivot info for tile 1.
///
/// @param[in] mpi_rank
///     MPI rank in the panel factorization
///
/// @param[in] thread_id
///     ID of this thread
///
/// @param[in] thread_size
///     number of local threads
///
/// @param[in] thread_barrier
///     barrier for synchronizing local threads
///
/// @param[out] max_value
///     workspace for per-thread pivot value, of length thread_size.
///
/// @param[out] max_index
///     workspace for per-thread pivot index, of length thread_size.
//      (local index of the tile containing the pivot)
///
/// @param[out] max_offset
///     workspace for per-thread pivot offset, of length thread_size.
///     (pivot offset in the tile)
///
/// @param[out] top_block
///     workspace for broadcasting the top row for the geru operation
///     and the top block for the gemm operation.
///
/// @ingroup gesv_tile
///
template <typename scalar_t>
void getrf_tntpiv_local(
    int64_t diag_len, int64_t ib, int stage,
    std::vector< Tile< scalar_t > >& tiles,
    std::vector< int64_t >& tile_indices,
    std::vector< std::vector< internal::AuxPivot< scalar_t > > >& aux_pivot,
    int mpi_rank, int thread_id, int thread_size,
    ThreadBarrier& thread_barrier,
    std::vector< scalar_t >& max_value,
    std::vector< int64_t  >& max_index,
    std::vector< int64_t  >& max_offset,
    std::vector< scalar_t >& top_block)
{
    trace::Block trace_block( "lapack::getrf_tntpiv" );

    using real_t = blas::real_type<scalar_t>;
    using internal::AuxPivot;

    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    assert( int( max_value .size() ) == thread_size );
    assert( int( max_index .size() ) == thread_size );
    assert( int( max_offset.size() ) == thread_size );

    int64_t nb = tiles[ 0 ].nb();

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        //=======================
        // ib panel factorization
        int64_t kb = std::min( diag_len-k, ib );

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+kb; ++j) {

            // Start with diagonal entry.
            max_value [ thread_id ] = tiles[ 0 ]( j, j );
            max_index [ thread_id ] = 0;
            max_offset[ thread_id ] = j;

            //------------------
            // thread max search
            for (int64_t idx = thread_id;
                 idx < int64_t( tiles.size() );
                 idx += thread_size)
            {
                auto tile = tiles[ idx ];

                // For diagonal tile, start at sub-diagonal;
                // otherwise start at tile's row 0.
                int64_t start = (idx == 0 ? j+1 : 0);
                for (int64_t i = start; i < tile.mb(); ++i) {
                    if (cabs1( tile( i, j ) ) > cabs1( max_value[ thread_id ] )) {
                        max_value [ thread_id ] = tile( i, j );
                        max_index [ thread_id ] = idx;
                        max_offset[ thread_id ] = i;
                    }
                }
            }
            thread_barrier.wait( thread_size );

            //------------------------------------
            // global max reduction and pivot swap
            if (thread_id == 0) {
                // threads max reduction
                for (int tid = 1; tid < thread_size; ++tid) {
                    if (cabs1( max_value[ tid ] ) > cabs1( max_value[ 0 ] )) {
                        max_value [ 0 ] = max_value [ tid ];
                        max_index [ 0 ] = max_index [ tid ];
                        max_offset[ 0 ] = max_offset[ tid ];
                    }
                }

                // Stage 0 is first local LU before the tree reduction.
                // Read global index and offset and swap it with j.
                if (stage == 0) {
                    aux_pivot[ 0 ][ j ] = AuxPivot<scalar_t>(
                        tile_indices[ max_index[ 0 ] ], max_offset[ 0 ],
                        max_index[ 0 ], max_offset[ 0 ],
                        max_value[ 0 ], mpi_rank);
                }
                else {
                    assert( max_index[ 0 ] >= 0 && max_index[ 0 ] <= 1 );
                    int64_t global_tile_index
                        = aux_pivot[ max_index[ 0 ] ][ max_offset[ 0 ] ].tileIndex();
                    int64_t global_offset
                        = aux_pivot[ max_index[ 0 ] ][ max_offset[ 0 ] ].elementOffset();

                    aux_pivot[ max_index[ 0 ] ][ max_offset[ 0 ] ] = aux_pivot[ 0 ][ j ];

                    aux_pivot[ 0 ][ j ] = AuxPivot<scalar_t>(
                        global_tile_index, global_offset,
                        max_index [ 0 ], max_offset[ 0 ],
                        max_value [ 0 ], mpi_rank );
                }

                // pivot swap
                // if pivot is not on the diagonal
                if (aux_pivot[ 0 ][ j ].localTileIndex() > 0 ||
                    aux_pivot[ 0 ][ j ].localOffset() > j) {
                    // local swap
                    swapLocalRow( 0, nb,
                                  tiles[ 0 ], j,
                                  tiles[ aux_pivot[ 0 ][ j ].localTileIndex() ],
                                  aux_pivot[ 0 ][ j ].localOffset() );
                }
                // Broadcast the top row for the geru operation.
                if (k+kb > j+1) {
                    auto top_tile = tiles[ 0 ];
                    // todo: make it a tile operation
                    blas::copy( k+kb-j-1,
                                &top_tile.at( j, j+1 ), top_tile.stride(),
                                top_block.data(), 1 );
                }
            }
            thread_barrier.wait( thread_size );

            // column scaling and trailing update
            for (int64_t idx = thread_id;
                 idx < int64_t( tiles.size() );
                 idx += thread_size)
            {
                auto tile = tiles[ idx ];

                // column scaling
                real_t sfmin = std::numeric_limits<real_t>::min();
                if (cabs1( aux_pivot[ 0 ][ j ].value() ) >= sfmin) {
                    // todo: make it a tile operation
                    if (idx == 0) {
                        // diagonal tile
                        scalar_t alpha = one / tile( j, j );
                        int64_t m = tile.mb()-j-1;
                        if (m > 0)
                            blas::scal( tile.mb()-j-1, alpha, &tile.at( j+1, j ), 1 );
                    }
                    else {
                        // off diagonal tile
                        scalar_t alpha = one / aux_pivot[ 0 ][ j ].value();
                        blas::scal( tile.mb(), alpha, &tile.at( 0, j ), 1 );
                    }
                }
                else if (aux_pivot[ 0 ][ j ].value() != zero) {
                    if (idx == 0) {
                        // diagonal tile
                        for (int64_t i = j+1; i < tile.mb(); ++i)
                            tile.at( i, j ) /= tile( j, j );
                    }
                    else {
                        // off diagonal tile
                        for (int64_t i = 0; i < tile.mb(); ++i)
                            tile.at( i, j ) /= aux_pivot[ 0 ][ j ].value();
                    }
                }
                else {
                    // aux_pivot[ 0 ][ j ].value() == 0:
                    // The factorization has been completed
                    // but the factor U is exactly singular
                    // todo: how to handle a zero pivot
                }

                // trailing update
                // todo: make it a tile operation
                if (k+kb > j+1) {
                    if (idx == 0) {
                        blas::geru( Layout::ColMajor,
                                    tile.mb()-j-1, k+kb-j-1,
                                    -one, &tile.at( j+1, j ), 1,
                                          top_block.data(), 1,
                                          &tile.at( j+1, j+1 ), tile.stride() );
                    }
                    else {
                        blas::geru( Layout::ColMajor,
                                    tile.mb(), k+kb-j-1,
                                    -one, &tile.at( 0, j ), 1,
                                          top_block.data(), 1,
                                          &tile.at( 0, j+1 ), tile.stride() );
                    }
                }
            }
            // todo: needed? In tile::getrf, this thread_barrier was removed.
            thread_barrier.wait( thread_size );
        }

        // Trailing submatrix update.
        if (k+kb < nb) {
            //=================
            if (thread_id == 0) {
                // triangular solve
                auto top_tile = tiles[ 0 ];
                blas::trsm( Layout::ColMajor,
                            Side::Left, Uplo::Lower,
                            Op::NoTrans, Diag::Unit,
                            kb, nb-k-kb,
                            one, &top_tile.at( k, k ),    top_tile.stride(),
                                 &top_tile.at( k, k+kb ), top_tile.stride() );
            }
            // todo: supperfluous, since next if is also tid 0?
            // todo: merge these two `if` blocks, as in tile::getrf?
            thread_barrier.wait( thread_size );

            if (thread_id == 0) {
                // Broadcast the top block for gemm.
                auto top_tile = tiles[ 0 ];
                lapack::lacpy( lapack::MatrixType::General,
                               kb, nb-k-kb,
                               &top_tile.at( k, k+kb ), top_tile.stride(),
                               top_block.data(), kb );
            }
            thread_barrier.wait( thread_size );

            //============================
            // rank-ib update to the right
            for (int64_t idx = thread_id;
                 idx < int64_t( tiles.size() );
                 idx += thread_size)
            {
                auto tile = tiles[ idx ];

                if (idx == 0) {
                    if (k+kb < tile.mb()) {
                        blas::gemm( blas::Layout::ColMajor,
                                    Op::NoTrans, Op::NoTrans,
                                    tile.mb()-k-kb, nb-k-kb, kb,
                                    -one, &tile.at( k+kb, k    ), tile.stride(),
                                          &tile.at( k,    k+kb ), tile.stride(),
                                    one,  &tile.at( k+kb, k+kb ), tile.stride() );
                    }
                }
                else {
                    blas::gemm( blas::Layout::ColMajor,
                                Op::NoTrans, Op::NoTrans,
                                tile.mb(), nb-k-kb, kb,
                                -one, &tile.at( 0, k ), tile.stride(),
                                      top_block.data(), kb,
                                one,  &tile.at( 0, k+kb ), tile.stride() );
                }
            }
            thread_barrier.wait( thread_size );
        }
    }
}

} // namespace tile
} // namespace slate

#endif // SLATE_TILE_GETRF_TNTPIV_HH
