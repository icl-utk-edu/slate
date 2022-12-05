// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Apply local reflectors on a Hermitian trailing submatrix. Compute Wi = sum_j Aij Vj.
/// Wher, A is the Hermitian trailing submatrix.
/// B contains the local reflectors Vj from local QR factorization
/// of a panel of A
/// C is the output matrix contains the Wi = sum_j Aij Vj
/// panel_rank_rows contains the local row indices for panel_rank,
/// where the panel_rank is B.tileRank( i, 0 ), i = 0:nt-1.
/// Dispatches to target implementations.
/// @ingroup heev_internal
///
template <Target target, typename scalar_t>
void he2hb_hemm(
    HermitianMatrix<scalar_t>&& A,
    Matrix<scalar_t>&& B,
    Matrix<scalar_t>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    he2hb_hemm( internal::TargetType<target>(),
                A, B, C, panel_rank_rows, priority, queue_index );
}

//------------------------------------------------------------------------------
/// Apply local reflectors.
/// Host OpenMP task implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_hemm(
    internal::TargetType<Target::HostTask>,
    HermitianMatrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    int64_t mt = A.mt();
    const scalar_t one  = 1;

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    #pragma omp taskgroup
    for (int64_t i = 0; i < mt; ++i) {
        #pragma omp task shared( A, B, C )
        {
            for (int64_t j : panel_rank_rows) {
                if (i >= j) { // lower or diagonal
                    if (A.tileIsLocal( i, j )) {
                        A.tileGetForReading( i, j, layout_conv );
                        B.tileGetForReading( j, 0, layout_conv );
                        C.tileGetForWriting( i, 0, layout_conv );
                        if (i == j) {
                            tile::hemm( Side::Left, one, A( i, j ), B( j, 0 ),
                                        one, C( i, 0 ) );
                        }
                        else {
                            // todo: if HeMatrix returned conjTrans tiles,
                            // could merge this with one below.
                            tile::gemm( one, A( i, j ), B( j, 0 ),
                                        one, C( i, 0 ) );
                        }
                        A.tileTick( i, j );
                        B.tileTick( j, 0 );
                    }
                }
                else { // upper
                    if (A.tileIsLocal( j, i )) {
                        A.tileGetForReading( j, i, layout_conv );
                        B.tileGetForReading( j, 0, layout_conv );
                        C.tileGetForWriting( i, 0, layout_conv );
                        tile::gemm( one, conj_transpose( A( j, i ) ), B( j, 0 ),
                                    one, C( i, 0 ) );
                        A.tileTick( j, i );
                        B.tileTick( j, 0 );
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Apply local reflectors.
/// Host nested OpenMP implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_hemm(
    internal::TargetType<Target::HostNest>,
    HermitianMatrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& C,
    std::vector<int64_t> panel_rank_rows,
    int priority, int64_t queue_index )
{
    slate_not_implemented( "Target::HostNest isn't yet supported." );
}

//------------------------------------------------------------------------------
/// Apply local reflectors.
/// Host batched implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_hemm(
    internal::TargetType<Target::HostBatch>,
    HermitianMatrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& C,
    std::vector<int64_t> panel_rank_rows,
    int priority, int64_t queue_index )
{
    slate_not_implemented( "Target::HostBatch isn't yet supported." );
}

#if 1 // device non-batch multi-queue implementation

//------------------------------------------------------------------------------
/// Apply local reflectors.
/// GPU device BLAS implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_hemm(
    internal::TargetType<Target::Devices>,
    HermitianMatrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& C,
    std::vector<int64_t> panel_rank_rows,
    int priority, int64_t queue_index )
{
    int64_t mt = A.mt();
    const scalar_t one  = 1;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    #pragma omp taskgroup
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared( A, B, C ) priority( priority )
        {

            std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
            for (int64_t j : panel_rank_rows) {
                for (int64_t i = 0; i < mt; ++i) {
                    if (i >= j) { // lower or diagonal
                        if (A.tileIsLocal( i, j )) {
                            if (device == C.tileDevice( i, 0 )) {
                                A_tiles_set.insert( { i, j } );
                                B_tiles_set.insert( { j, 0 } );
                                C_tiles_set.insert( { i, 0 } );
                            }
                        }
                    }
                    else { // upper
                        if (A.tileIsLocal( j, i )) {
                            if (device == C.tileDevice( i, 0 )) {
                                A_tiles_set.insert( { j, i } );
                                B_tiles_set.insert( { j, 0 } );
                                C_tiles_set.insert( { i, 0 } );
                            }
                        }
                    }
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
        }
    }

    int num_queues = C.numComputeQueues();

    #pragma omp taskgroup
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared( A, B, C ) priority( priority )
        {
            trace::Block trace_block( "blas::batch::he2hb_hemm" );
            // to have one queue and then fork several streams
            //blas::Queue* queue = C.compute_queue( device, queue_index );
            //assert( queue != nullptr );
            for (int64_t j : panel_rank_rows) {
                //queue->fork(); // to have multiple streams
                for (int64_t i = 0; i < mt; ++i) {
                    // queue per iteration i
                    blas::Queue* queue = C.compute_queue( device, i % num_queues );
                    assert( queue != nullptr );
                    if (i >= j) { // lower or diagonal
                        if (A.tileIsLocal( i, j )
                            && device == C.tileDevice( i, 0 )) {
                            //blas::Queue* queue = C.compute_queue( device, queue_index );
                            //assert( queue != nullptr );

                            auto Aij = A( i, j, device );
                            auto Bj0 = B( j, 0, device );
                            auto Ci0 = C( i, 0, device );

                            if (i == j) {
                                blas::hemm( layout, blas::Side::Left,
                                            blas::Uplo::Lower,
                                            Ci0.mb(), Ci0.nb(),
                                            one, Aij.data(), Aij.stride(),
                                                 Bj0.data(), Bj0.stride(),
                                            one, Ci0.data(), Ci0.stride(),
                                            *queue );
                            }
                            else {
                                // todo: if HeMatrix returned conjTrans tiles,
                                // could merge this with one below.
                                blas::gemm( layout, Op::NoTrans, Op::NoTrans,
                                            Aij.mb(), Bj0.nb(), Aij.nb(),
                                            one, Aij.data(), Aij.stride(),
                                                 Bj0.data(), Bj0.stride(),
                                            one, Ci0.data(), Ci0.stride(),
                                            *queue );
                            }
                        }
                    }
                    else { // upper
                        if (A.tileIsLocal( j, i )
                            && device == C.tileDevice( i, 0 )) {
                            auto Aji = A( j, i, device );
                            auto Bj0 = B( j, 0, device );
                            auto Ci0 = C( i, 0, device );

                            blas::gemm( layout, Op::ConjTrans, Op::NoTrans,
                                        Aji.nb(), Bj0.nb(), Aji.mb(),
                                        one, Aji.data(), Aji.stride(),
                                             Bj0.data(), Bj0.stride(),
                                        one, Ci0.data(), Ci0.stride(),
                                        *queue );
                        }
                    }
                    //queue->revolve(); // new stream
                } // i loop
                //queue->join(); // sync all the streams on this queue
            } // j loop
            //queue->sync(); // sync all the queues/streams

            // sync all the queues (in case of queue per iteration i)
            for (int64_t i = 0; i < num_queues; i++) {
                blas::Queue* queue = C.compute_queue( device, i );
                queue->sync();
            }

            for (int64_t j : panel_rank_rows) {
                for (int64_t i = 0; i < mt; ++i) {
                    if (i >= j) { // lower or diagonal
                        if (A.tileIsLocal( i, j )
                            && device == C.tileDevice( i, 0 )) {
                            C.tileModified( i, 0, device );
                            A.tileRelease( i, j, device );
                            B.tileRelease( j, 0, device );
                            A.tileTick( i, j );
                            B.tileTick( j, 0 );
                        }
                    }
                    else { // upper
                        if (A.tileIsLocal( j, i )
                            && device == C.tileDevice( i, 0 )) {
                            C.tileModified( i, 0, device );
                            A.tileRelease( j, i, device );
                            B.tileRelease( j, 0, device );
                            A.tileTick( j, i );
                            B.tileTick( j, 0 );
                        }
                    }
                } //i loop
            } // j loop
        } // pragma
    } // devices
}

#else // device batch implementation

//------------------------------------------------------------------------------
/// Apply local reflectors.
/// GPU device batched cuBLAS implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_hemm(internal::TargetType<Target::Devices>,
    HermitianMatrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& C,
    std::vector<int64_t> panel_rank_rows,
    int priority, int64_t queue_index)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    int64_t mt = A.mt();

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    assert( C.num_devices() > 0 );
    scalar_t alpha = 1.;
    scalar_t beta = 1.;

    // check if there is a cleanup tile
    int64_t i_interior = mt;
    int64_t i_last = 0;
    if (C.tileMb( mt-1 ) != C.tileMb( 0 )) {
        i_interior = C.mt() - 1;
        i_last = 1;
    }

    Op opA = A.op();
    Op opB = B.op();
    Op opA_upper = Op::ConjTrans;

    int err = 0;
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared( A, B, C, err ) priority( priority )
        {
            std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
            for (int64_t j : panel_rank_rows) {
                for (int64_t i = 0; i < mt; ++i) {
                    if (i >= j) { // lower or diagonal
                        if (A.tileIsLocal( i, j )) {
                            if (device == C.tileDevice( i, 0 )) {
                                A_tiles_set.insert( { i, j } );
                                B_tiles_set.insert( { j, 0 } );
                                C_tiles_set.insert( { i, 0 } );
                            }
                        }
                    }
                    else { // upper
                        if (A.tileIsLocal( j, i )) {
                            if (device == C.tileDevice( i, 0 )) {
                                A_tiles_set.insert( { j, i } );
                                B_tiles_set.insert( { j, 0 } );
                                C_tiles_set.insert( { i, 0 } );
                            }
                        }
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

                // interior
                std::vector<scalar_t*> a_array00_lower;
                std::vector<scalar_t*> b_array00_lower;
                std::vector<scalar_t*> c_array00_lower;
                a_array00_lower.reserve( batch_size );
                b_array00_lower.reserve( batch_size );
                c_array00_lower.reserve( batch_size );

                std::vector<scalar_t*> a_array00_upper;
                std::vector<scalar_t*> b_array00_upper;
                std::vector<scalar_t*> c_array00_upper;
                a_array00_upper.reserve( batch_size );
                b_array00_upper.reserve( batch_size );
                c_array00_upper.reserve( batch_size );

                std::vector<scalar_t*> a_array00_diag;
                std::vector<scalar_t*> b_array00_diag;
                std::vector<scalar_t*> c_array00_diag;
                a_array00_diag.reserve( batch_size );
                b_array00_diag.reserve( batch_size );
                c_array00_diag.reserve( batch_size );

                int64_t lda00_lower = 0;
                int64_t lda00_upper = 0;
                int64_t lda00_diag = 0;
                int64_t ldb00 = 0;
                int64_t ldc00 = 0;
                int64_t mb00 = C.tileMb( 0 );
                int64_t nb00 = C.tileNb( 0 );
                int64_t kb   = B.tileMb( j );

                for (int64_t i = 0; i < i_interior; ++i) {
                    if (i > j) { // lower
                        if (A.tileIsLocal( i, j )) {
                            if (device == C.tileDevice( i, 0 )) {
                                a_array00_lower.push_back( A( i, j, device ).data() );
                                b_array00_lower.push_back( B( j, 0, device ).data() );
                                c_array00_lower.push_back( C( i, 0, device ).data() );
                                lda00_lower = A( i, j, device ).stride();
                                ldb00 = B( j, 0, device ).stride();
                                ldc00 = C( i, 0, device ).stride();
                            }
                        }
                    }
                }

                for (int64_t i = 0; i < i_interior; ++i) {
                    if (i < j) { // upper
                        if (A.tileIsLocal( j, i )) {
                            if (device == C.tileDevice( i, 0 )) {
                                a_array00_upper.push_back( A( j, i, device ).data() );
                                b_array00_upper.push_back( B( j, 0, device ).data() );
                                c_array00_upper.push_back( C( i, 0, device ).data() );
                                lda00_upper = A( j, i, device ).stride();
                                ldb00 = B( j, 0, device ).stride();
                                ldc00 = C( i, 0, device ).stride();
                            }
                        }
                    }
                }

                // todo: no need for two loops
                for (int64_t i = 0; i < i_interior; ++i) {
                    if (i == j) { // diagonal
                        if (A.tileIsLocal( i, j )) {
                            if (device == C.tileDevice( i, 0 )) {
                                a_array00_diag.push_back( A( i, j, device ).data() );
                                b_array00_diag.push_back( B( j, 0, device ).data() );
                                c_array00_diag.push_back( C( i, 0, device ).data() );
                                lda00_diag = A( i, j, device ).stride();
                                ldb00 = B( j, 0, device ).stride();
                                ldc00 = C( i, 0, device ).stride();
                            }
                        }
                    }
                }

                // last row if there is a clean-up tile
                std::vector<scalar_t*> a_array10_lower;
                std::vector<scalar_t*> b_array10_lower;
                std::vector<scalar_t*> c_array10_lower;
                a_array10_lower.reserve( batch_size );
                b_array10_lower.reserve( batch_size );
                c_array10_lower.reserve( batch_size );

                std::vector<scalar_t*> a_array10_upper;
                std::vector<scalar_t*> b_array10_upper;
                std::vector<scalar_t*> c_array10_upper;
                a_array10_upper.reserve( batch_size );
                b_array10_upper.reserve( batch_size );
                c_array10_upper.reserve( batch_size );

                std::vector<scalar_t*> a_array10_diag;
                std::vector<scalar_t*> b_array10_diag;
                std::vector<scalar_t*> c_array10_diag;
                a_array10_diag.reserve( batch_size );
                b_array10_diag.reserve( batch_size );
                c_array10_diag.reserve( batch_size );

                int64_t lda10_lower = 0;
                int64_t lda10_upper = 0;
                int64_t lda10_diag = 0;
                int64_t ldb10 = 0;
                int64_t ldc10 = 0;
                int64_t mb10 = C.tileMb( C.mt()-1 );
                int64_t nb10 = C.tileNb( 0 );

                if (i_last == 1) {
                    int64_t i = C.mt()-1;
                    if (i > j) { // lower
                        if (A.tileIsLocal( i, j )) {
                            if (device == C.tileDevice( i, 0 )) {
                                a_array10_lower.push_back( A( i, j, device ).data() );
                                b_array10_lower.push_back( B( j, 0, device ).data() );
                                c_array10_lower.push_back( C( i, 0, device ).data() );
                                lda10_lower = A( i, j, device ).stride();
                                ldb10 = B( j, 0, device ).stride();
                                ldc10 = C( i, 0, device ).stride();
                            }
                        }
                    }
                    if (i < j) { // upper
                        if (A.tileIsLocal( j, i )) {
                            if (device == C.tileDevice( i, 0 )) {
                                a_array10_upper.push_back( A( j, i, device ).data() );
                                b_array10_upper.push_back( B( j, 0, device ).data() );
                                c_array10_upper.push_back( C( i, 0, device ).data() );
                                lda10_upper = A( j, i, device ).stride();
                                ldb10 = B( j, 0, device ).stride();
                                ldc10 = C( i, 0, device ).stride();
                            }
                        }
                    }
                    if (i == j) { // diag
                        if (A.tileIsLocal( j, i )) {
                            if (device == C.tileDevice( i, 0 )) {
                                a_array10_diag.push_back( A( j, i, device ).data() );
                                b_array10_diag.push_back( B( j, 0, device ).data() );
                                c_array10_diag.push_back( C( i, 0, device ).data() );
                                lda10_diag = A( j, i, device ).stride();
                                ldb10 = B( j, 0, device ).stride();
                                ldc10 = C( i, 0, device ).stride();
                            }
                        }
                    }
                }

                std::vector<Op> opA_( 1, opA );
                std::vector<Op> opB_( 1, opB );
                std::vector<Op> opA_upper_( 1, opA_upper );
                std::vector<scalar_t> alpha_( 1, alpha );
                std::vector<scalar_t> beta_( 1, beta );
                std::vector<int64_t> kb_( 1, kb );
                // info size 0 disables slow checks in batched BLAS++.
                std::vector<int64_t> info;

                blas::Queue* queue = C.compute_queue( device, queue_index );
                assert( queue != nullptr );
                //int cuerror;

                {
                    trace::Block trace_block( "blas::batch::he2hb_hemm_gemm" );
                    if (c_array00_lower.size() > 0) {
                        std::vector<int64_t>    m( 1, mb00 );
                        std::vector<int64_t>    n( 1, nb00 );
                        std::vector<int64_t> ldda( 1, lda00_lower );
                        std::vector<int64_t> lddb( 1, ldb00 );
                        std::vector<int64_t> lddc( 1, ldc00 );

                        //cuerror = cudaHostRegister( a_array00_lower.data(),
                            //(size_t)a_array00_lower.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( b_array00_lower.data(),
                            //(size_t)b_array00_lower.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( c_array00_lower.data(),
                            //(size_t)c_array00_lower.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );

                        blas::batch::gemm(
                            layout, opA_, opB_,
                            m, n, kb_,
                            alpha_, a_array00_lower, ldda,
                                    b_array00_lower, lddb,
                            beta_,  c_array00_lower, lddc,
                            c_array00_lower.size(), info, *queue );

                        //cuerror = cudaHostUnregister( a_array00_lower.data() );
                        //cuerror = cudaHostUnregister( b_array00_lower.data() );
                        //cuerror = cudaHostUnregister( c_array00_lower.data() );

                    }

                    if (c_array00_upper.size() > 0) {
                        std::vector<int64_t>    m( 1, mb00 );
                        std::vector<int64_t>    n( 1, nb00 );
                        std::vector<int64_t> ldda( 1, lda00_upper );
                        std::vector<int64_t> lddb( 1, ldb00 );
                        std::vector<int64_t> lddc( 1, ldc00 );

                        //cuerror = cudaHostRegister( a_array00_upper.data(),
                            //(size_t)a_array00_upper.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( b_array00_upper.data(),
                            //(size_t)b_array00_upper.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( c_array00_upper.data(),
                            //(size_t)c_array00_upper.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );

                        blas::batch::gemm(
                            layout, opA_upper_, opB_,
                            m, n, kb_,
                            alpha_, a_array00_upper, ldda,
                                    b_array00_upper, lddb,
                            beta_,  c_array00_upper, lddc,
                            c_array00_upper.size(), info, *queue );

                        //cuerror = cudaHostUnregister( a_array00_upper.data() );
                        //cuerror = cudaHostUnregister( b_array00_upper.data() );
                        //cuerror = cudaHostUnregister( c_array00_upper.data() );
                    }

                    if (c_array10_lower.size() > 0) {
                        opA = Op::NoTrans;
                        std::vector<int64_t>    m( 1, mb10 );
                        std::vector<int64_t>    n( 1, nb10 );
                        std::vector<int64_t> ldda( 1, lda10_lower );
                        std::vector<int64_t> lddb( 1, ldb10 );
                        std::vector<int64_t> lddc( 1, ldc10 );

                        //cuerror = cudaHostRegister( a_array10_lower.data(),
                            //(size_t)a_array10_lower.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( b_array10_lower.data(),
                            //(size_t)b_array10_lower.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( c_array10_lower.data(),
                            //(size_t)c_array10_lower.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );

                        blas::batch::gemm(
                            layout, opA_, opB_,
                            m, n, kb_,
                            alpha_, a_array10_lower, ldda,
                                    b_array10_lower, lddb,
                            beta_,  c_array10_lower, lddc,
                            c_array10_lower.size(), info, *queue );

                        //cuerror = cudaHostUnregister( a_array10_lower.data() );
                        //cuerror = cudaHostUnregister( b_array10_lower.data() );
                        //cuerror = cudaHostUnregister( c_array10_lower.data() );
                    }

                    if (c_array10_upper.size() > 0) {
                        std::vector<int64_t>    m( 1, mb10 );
                        std::vector<int64_t>    n( 1, nb10 );
                        std::vector<int64_t> ldda( 1, lda10_upper );
                        std::vector<int64_t> lddb( 1, ldb10 );
                        std::vector<int64_t> lddc( 1, ldc10 );

                        //cuerror = cudaHostRegister( a_array10_upper.data(),
                            //(size_t)a_array10_upper.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( b_array10_upper.data(),
                            //(size_t)b_array10_upper.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( c_array10_upper.data(),
                            //(size_t)c_array10_upper.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );

                        blas::batch::gemm(
                            layout, opA_upper_, opB_,
                            m, n, kb_,
                            alpha_, a_array10_upper, ldda,
                                    b_array10_upper, lddb,
                            beta_,  c_array10_upper, lddc,
                            c_array10_upper.size(), info, *queue );

                        //cuerror = cudaHostUnregister( a_array10_upper.data() );
                        //cuerror = cudaHostUnregister( b_array10_upper.data() );
                        //cuerror = cudaHostUnregister( c_array10_upper.data() );
                    }
                }

                {
                    trace::Block trace_block( "blas::batch::he2hb_hemm_hemm" );
                    if (c_array00_diag.size() > 0) {
                        std::vector<int64_t>    m( 1, mb00 );
                        std::vector<int64_t>    n( 1, nb00 );
                        std::vector<int64_t> ldda( 1, lda00_diag );
                        std::vector<int64_t> lddb( 1, ldb00 );
                        std::vector<int64_t> lddc( 1, ldc00 );

                        blas::Side side_ = Side::Left;
                        std::vector<blas::Side> side( 1, side_ );
                        blas::Uplo uplo_ = Uplo::Lower;
                        std::vector<blas::Uplo> uplo( 1, uplo_ );

                        //cuerror = cudaHostRegister( a_array00_diag.data(),
                            //(size_t)a_array00_diag.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( b_array00_diag.data(),
                            //(size_t)b_array00_diag.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( c_array00_diag.data(),
                            //(size_t)c_array00_diag.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );

                        blas::batch::hemm(
                            layout, side, uplo,
                            m, n,
                            alpha_, a_array00_diag, ldda,
                                   b_array00_diag, lddb,
                            beta_, c_array00_diag, lddc,
                            c_array00_diag.size(), info, *queue );

                        //cuerror = cudaHostUnregister( a_array00_diag.data() );
                        //cuerror = cudaHostUnregister( b_array00_diag.data() );
                        //cuerror = cudaHostUnregister( c_array00_diag.data() );
                    }

                    if (c_array10_diag.size() > 0) {
                        std::vector<int64_t>    m( 1, mb10 );
                        std::vector<int64_t>    n( 1, nb10 );
                        std::vector<int64_t> ldda( 1, lda10_diag );
                        std::vector<int64_t> lddb( 1, ldb10 );
                        std::vector<int64_t> lddc( 1, ldc10 );

                        blas::Side side_ = Side::Left;
                        std::vector<blas::Side> side( 1, side_ );
                        blas::Uplo uplo_ = Uplo::Lower;
                        std::vector<blas::Uplo> uplo( 1, uplo_ );

                        //cuerror = cudaHostRegister( a_array10_diag.data(),
                            //(size_t)a_array10_diag.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( b_array10_diag.data(),
                            //(size_t)b_array10_diag.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );
                        //cuerror = cudaHostRegister( c_array10_diag.data(),
                            //(size_t)c_array10_diag.size()*sizeof(scalar_t),
                            //cudaHostRegisterDefault );

                        blas::batch::hemm(
                            layout, side, uplo,
                            m, n,
                            alpha_, a_array10_diag, ldda,
                                   b_array10_diag, lddb,
                            beta_, c_array10_diag, lddc,
                            c_array10_diag.size(), info, *queue );

                        //cuerror = cudaHostUnregister( a_array10_diag.data() );
                        //cuerror = cudaHostUnregister( b_array10_diag.data() );
                        //cuerror = cudaHostUnregister( c_array10_diag.data() );
                    }
                }

                queue->sync();

                for (int64_t i = 0; i < mt; ++i) {
                    if (i >= j) { // lower or diagonal
                        if (A.tileIsLocal( i, j )) {
                            if (device == C.tileDevice( i, 0 )) {
                                A.tileRelease( i, j, device );
                                B.tileRelease( j, 0, device );
                                A.tileTick( i, j );
                                B.tileTick( j, 0 );
                            }
                        }
                    }
                    else { // upper
                        if (A.tileIsLocal( j, i )) {
                            if (device == C.tileDevice( i, 0 )) {
                                A.tileRelease( j, i, device );
                                B.tileRelease( j, 0, device );
                                A.tileTick( j, i );
                                B.tileTick( j, 0 );
                            }
                        }
                    }
                }
            } // j = panel_rank_rows
        } // task
    } // device

    if (err)
        slate_error( std::to_string( err ) );
}

#endif // else device batch implementation

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void he2hb_hemm<Target::HostTask, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& B,
    Matrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm<Target::HostTask, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& B,
    Matrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    Matrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    Matrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm<Target::Devices, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& B,
    Matrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm<Target::Devices, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& B,
    Matrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::Devices, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    Matrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::Devices, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    Matrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm<Target::HostNest, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& B,
    Matrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm<Target::HostNest, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& B,
    Matrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::HostNest, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    Matrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::HostNest, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    Matrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm<Target::HostBatch, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& B,
    Matrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm<Target::HostBatch, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& B,
    Matrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::HostBatch, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    Matrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_hemm< Target::HostBatch, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    Matrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

} // namespace internal
} // namespace slate
