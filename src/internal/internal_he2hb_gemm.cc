// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Inner product C = AB to update a single block C,
/// where A and B are single blocks.
/// panel_ranks are the mpi ranks in A block (A_panel.getRanks( &panel_ranks )),
/// panel_rank is in panel_ranks.
/// Loop over the local tiles of A on this panel_rank to update C = AB.
/// Dispatches to target implementations.
/// @ingroup heev_internal
/// todo: add more details
///
template <Target target, typename scalar_t>
void he2hb_gemm(
    scalar_t alpha, Matrix<scalar_t>&& A, Matrix<scalar_t>&& B,
    scalar_t beta,  Matrix<scalar_t>&& C,
    int panel_rank,
    int priority, int64_t queue_index )
{
    he2hb_gemm( internal::TargetType<target>(),
                alpha, A, B, beta, C,
                panel_rank, priority, queue_index );
}

//------------------------------------------------------------------------------
/// Inner product C = AB,
/// Host OpenMP task implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_gemm(
    internal::TargetType<Target::HostTask>,
    scalar_t alpha, Matrix<scalar_t>& A, Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    int panel_rank,
    int priority, int64_t queue_index )
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    assert( A.nt() == B.mt() );

    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        #pragma omp task priority( priority )
        {
            scalar_t beta_ = beta;
            for (int64_t k = 0; k < A.nt(); ++k) {
                if (A.tileRank( i, k ) == panel_rank) {
                    A.tileGetForReading( i, k, layout_conv );
                    B.tileGetForReading( k, 0, layout_conv );
                    C.tileGetForWriting( i, 0, layout_conv );
                    tile::gemm( alpha, A( i, k ), B( k, 0 ),
                                beta_, C( i, 0 ) );
                    A.tileTick( i, k );
                    B.tileTick( k, 0 );
                }
                beta_ = 1.0;
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Inner product C = AB,
/// Device implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_gemm(
    internal::TargetType<Target::Devices>,
    scalar_t alpha, Matrix<scalar_t>& A, Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    int panel_rank,
    int priority, int64_t queue_index)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    using blas::conj;
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // check dimensions
    assert( C.mt() > 0 );
    assert( C.nt() > 0 );
    assert( A.mt() == C.mt() );
    assert( B.nt() == C.nt() );

    assert( C.num_devices() > 0 );

    int err = 0;

    #pragma omp taskgroup
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared( A, B, C, err ) priority( priority )
        {
            Op opA = A.op();
            Op opB = B.op();

            for (int64_t k = 0; k < B.mt(); ++k) {
                std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
                for (int64_t i = 0; i < A.mt(); ++i) {
                    if (A.tileRank( i, k ) == panel_rank
                        && device == C.tileDevice( i, 0 )) {
                        A_tiles_set.insert( { i, k } );
                        B_tiles_set.insert( { k, 0 } );
                        C_tiles_set.insert( { i, 0 } );
                    }
                }
                #pragma omp taskgroup
                {
                    #pragma omp task default( shared )
                    {
                        A.tileGetForReading( A_tiles_set, device, layout_conv );
                    }
                    #pragma omp task default( shared )
                    {
                        B.tileGetForReading( B_tiles_set, device, layout_conv );
                    }
                    #pragma omp task default( shared )
                    {
                        C.tileGetForWriting( C_tiles_set, device, layout_conv );
                    }
                }

                int64_t batch_size = C_tiles_set.size();

                int64_t i_interior = A.mt();
                int64_t i_last = 0;
                int64_t mt = C.mt();

                // check if there are cleanup tiles
                if (C.tileMb( mt-1 ) != C.tileMb( 0 )) {
                    //if (C.m() % C.tileMb(0) != 0) {
                    i_interior = A.mt() - 1;
                    i_last = 1;
                }

                // interior
                std::vector<scalar_t*> a_array00;
                std::vector<scalar_t*> b_array00;
                std::vector<scalar_t*> c_array00;
                a_array00.reserve( batch_size );
                b_array00.reserve( batch_size );
                c_array00.reserve( batch_size );

                int64_t lda00 = 0;
                int64_t ldb00 = 0;
                int64_t ldc00 = 0;
                int64_t mb00 = C.tileMb( 0 );
                int64_t nb00 = C.tileNb( 0 );
                int64_t kb   = A.tileNb( 0 );
                for (int64_t i = 0; i < i_interior; ++i) {
                    if (A.tileRank( i, k ) == panel_rank
                        && device == C.tileDevice( i, 0 )) {
                        a_array00.push_back( A( i, k, device ).data() );
                        b_array00.push_back( B( k, 0, device ).data() );
                        c_array00.push_back( C( i, 0, device ).data() );
                        lda00 = A( i, k, device ).stride();
                        ldb00 = B( k, 0, device ).stride();
                        ldc00 = C( i, 0, device ).stride();
                    }
                }

                // if mod( n, nb ) != 0, this is for the last tile
                std::vector<scalar_t*> a_array11;
                std::vector<scalar_t*> b_array11;
                std::vector<scalar_t*> c_array11;
                a_array11.reserve( batch_size );
                b_array11.reserve( batch_size );
                c_array11.reserve( batch_size );

                int64_t lda11 = 0;
                int64_t ldb11 = 0;
                int64_t ldc11 = 0;
                int64_t mb11 = C.tileMb( C.mt()-1 );
                int64_t nb11 = C.tileNb( C.nt()-1 );
                // same kb as above
                {
                    int i = C.mt()-1;
                    if ((A.tileRank( i, k ) == panel_rank) && (i_last == 1)) {
                        if (device == C.tileDevice( i, 0 )) {
                            a_array11.push_back( A( i, k, device ).data() );
                            b_array11.push_back( B( k, 0, device ).data() );
                            c_array11.push_back( C( i, 0, device ).data() );
                            lda11 = A( i, k, device ).stride();
                            ldb11 = B( k, 0, device ).stride();
                            ldc11 = C( i, 0, device ).stride();
                        }
                    }
                }

                if (C.op() != Op::NoTrans) {
                    // swap A <=> B; swap m <=> n
                    swap( opA, opB );
                    swap( a_array00, b_array00 );
                    swap( lda00, ldb00 );
                    swap( mb00, nb00 );
                }

                {
                    trace::Block trace_block( "blas::batch::he2hb_gemm" );

                    std::vector<Op> opA_( 1, opA );
                    std::vector<Op> opB_( 1, opB );
                    std::vector<scalar_t> alpha_( 1, alpha );
                    std::vector<scalar_t> beta_( 1, beta );
                    std::vector<int64_t> kb_( 1, kb );
                    // info size 0 disables slow checks in batched BLAS++.
                    std::vector<int64_t> info;

                    blas::Queue* queue = C.compute_queue( device, queue_index );
                    assert( queue != nullptr );

                    if (c_array00.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb00 );
                        std::vector<int64_t>    n( 1,  nb00 );
                        std::vector<int64_t> ldda( 1, lda00 );
                        std::vector<int64_t> lddb( 1, ldb00 );
                        std::vector<int64_t> lddc( 1, ldc00 );
                        blas::batch::gemm(
                            layout, opA_, opB_,
                            //m, n, kb_,
                            m, n, lddb,
                            alpha_, a_array00, ldda,
                                    b_array00, lddb,
                            beta_,  c_array00, lddc,
                            c_array00.size(), info, *queue );
                    }

                    if (c_array11.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb11 );
                        std::vector<int64_t>    n( 1,  nb11 );
                        std::vector<int64_t> ldda( 1, lda11 );
                        std::vector<int64_t> lddb( 1, ldb11 );
                        std::vector<int64_t> lddc( 1, ldc11 );

                        blas::batch::gemm(
                            layout, opA_, opB_,
                            //m, n, kb_,
                            m, n, lddb,
                            alpha_, a_array11, ldda,
                                    b_array11, lddb,
                            beta_,  c_array11, lddc,
                            c_array11.size(), info, *queue );
                    }
                    queue->sync();
                }

                for (int64_t i = 0; i < A.mt(); ++i) {
                    if (A.tileRank( i, k ) == panel_rank
                        && device == C.tileDevice( i, 0 )) {
                        // erase tmp local and remote device tiles;
                        A.tileRelease( i, k, device );
                        B.tileRelease( k, 0, device );
                        // decrement life for remote tiles
                        A.tileTick( i, k );
                        B.tileTick( k, 0 );
                    }
                }
                beta = 1.0;
            } // for loop (k)
        } // pragma
    } // device

    if (err)
        slate_error( std::to_string( err ) );
}

//------------------------------------------------------------------------------
/// Host nested OpenMP -- not implemented.
template <typename scalar_t>
void he2hb_gemm(
    internal::TargetType<Target::HostNest>,
    scalar_t alpha, Matrix<scalar_t>& A, Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    int panel_rank,
    int priority, int64_t queue_index )
{
    slate_not_implemented( "Target::HostNest isn't yet supported." );
}

//------------------------------------------------------------------------------
/// Host batched -- not implemented.
template <typename scalar_t>
void he2hb_gemm(
    internal::TargetType<Target::HostBatch>,
    scalar_t alpha, Matrix<scalar_t>& A, Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    int panel_rank,
    int priority, int64_t queue_index)
{
    slate_not_implemented( "Target::HostBatch isn't yet supported." );
}

// Explicit instantiations.
// ----------------------------------------
template
void he2hb_gemm<Target::HostTask, float>(
    float alpha, Matrix<float>&& A, Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm<Target::HostTask, double>(
    double alpha, Matrix<double>&& A, Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm<Target::Devices, float>(
    float alpha, Matrix<float>&& A, Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm<Target::Devices, double>(
    double alpha, Matrix<double>&& A, Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm<Target::HostNest, float>(
    float alpha, Matrix<float>&& A, Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm<Target::HostNest, double>(
    double alpha, Matrix<double>&& A, Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A, Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A, Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_gemm< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int panel_rank,
    int priority, int64_t queue_index);

} // namespace internal
} // namespace slate
