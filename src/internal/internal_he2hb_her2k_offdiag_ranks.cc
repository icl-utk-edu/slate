// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// matrix multiply to update trailing matrix, except the diagonal tiles.
/// where A is a single block column (contains V after doing QR factorization of a panel k).
/// and B is a single block row (contain W = VT)
/// C is a trailing square submatrix
/// Cij -= AikBjk^H, i = k+1:nt-1, j = k+1:nt-1.
/// Dispatches to target implementations.
/// @ingroup heev_internal
///
template <Target target, typename scalar_t>
void he2hb_her2k_offdiag_ranks(
    scalar_t alpha, Matrix<scalar_t>&& A,
                    Matrix<scalar_t>&& B,
    scalar_t beta,  HermitianMatrix<scalar_t>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index )
{
    he2hb_her2k_offdiag_ranks(
        internal::TargetType<target>(),
        alpha, A, B, beta, C,
        panel_rank_rows, priority, queue_index );
}

//------------------------------------------------------------------------------
/// matrix multiply to update trailing matrix, except the diagonal tiles.
/// Host OpenMP task implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_her2k_offdiag_ranks(
    internal::TargetType<Target::HostTask>,
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    int64_t nt = C.nt();

    // try to loop over one tile and do two gemm, similar to her2k
    #pragma omp taskgroup
    for (int64_t j = 0; j < nt; ++j) {
        #pragma omp task
        for (int64_t i : panel_rank_rows) {
            // todo: if HermitianMatrix returned conjTrans
            // tiles, could merge these two.
            if (i > j) {  // lower
                if (C.tileIsLocal( i, j )) {
                    A.tileGetForReading( i, 0, layout_conv );
                    B.tileGetForReading( j, 0, layout_conv );
                    C.tileGetForWriting( i, j, layout_conv );
                    // Aij -= Vik Wjk^H
                    tile::gemm( alpha, A( i, 0 ), conj_transpose( B( j, 0 ) ),
                                beta, C( i, j ) );
                    A.tileTick( i, 0 );
                    B.tileTick( j, 0 );
                }
            }
            else if (i < j) {  // upper
                if (C.tileIsLocal( j, i )) {
                    B.tileGetForReading( j, 0, layout_conv );
                    A.tileGetForReading( i, 0, layout_conv );
                    C.tileGetForWriting( j, i, layout_conv );
                    // Aji -= Wjk Vik^H
                    tile::gemm( alpha, B( j, 0 ), conjTranspose( A( i, 0 ) ),
                                beta, C( j, i ) );
                    B.tileTick( j, 0 );
                    A.tileTick( i, 0 );
                }
            }
            else { // i == j
                // Diagonal tiles dealt with above.
                assert( ! C.tileIsLocal( i, j ) );
            }
        }
    }
}

//------------------------------------------------------------------------------
/// matrix multiply to update trailing matrix, except the diagonal tiles.
/// where A is a single block column and B is a single block row.
/// Device implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_her2k_offdiag_ranks(
    internal::TargetType<Target::Devices>,
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    int64_t nt = C.nt();
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // check dimensions
    assert( C.mt() > 0 );
    assert( C.nt() > 0 );
    assert( A.nt() == 1 );
    assert( B.nt() == 1 );
    assert( A.mt() == C.mt() );

    assert( C.num_devices() > 0 );

    int err = 0;
    #pragma omp taskgroup
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared( A, B, C, err ) priority( priority )
        {
            Op opA = A.op();
            Op opB = B.op();
            opA = Op::NoTrans;
            opB = Op::ConjTrans;

            std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
            for (int64_t j = 0; j < nt; ++j) {
                for (int64_t i : panel_rank_rows) {
                    if (i > j) {
                        if (C.tileIsLocal( i, j )
                            && device == C.tileDevice( i, j )) {
                            A_tiles_set.insert( { i, 0 } );
                            B_tiles_set.insert( { j, 0 } );
                            C_tiles_set.insert( { i, j } );
                        }
                    }
                    else if (i < j) {
                        if (C.tileIsLocal( j, i )
                            && device == C.tileDevice( j, i )) {
                            B_tiles_set.insert( { j, 0 } );
                            A_tiles_set.insert( { i, 0 } );
                            C_tiles_set.insert( { j, i } );
                        }
                    }
                    else { // i == j
                        // Diagonal tiles dealt with above.
                        assert( ! C.tileIsLocal( i, j ) );
                    }
                }
            }
            int64_t batch_size = C_tiles_set.size();

            int64_t j_interior = nt;
            int64_t j_last = 0;
            if (C.tileNb( nt-1 ) != C.tileNb( 0 )) {
                j_interior = C.nt() - 1;
                j_last = 1;
            }

            int64_t m_indices = panel_rank_rows.size();
            int64_t i_interior = m_indices;
            int64_t i_last = 0;
            int64_t i0 = panel_rank_rows[ 0 ];
            int64_t i1 = panel_rank_rows[ m_indices-1 ];
            if (C.tileMb( i0 ) != C.tileMb( i1 )) {
                i_interior = m_indices - 1;
                i_last = 1;
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
            int64_t mb00 = C.tileMb( i0 );
            int64_t nb00 = C.tileNb( 0 );
            int64_t kb   = A.tileNb( 0 );

            for (int64_t j = 0; j < j_interior; ++j) {
                for (int64_t i_ = 0; i_ < i_interior; ++i_) {
                    int64_t i = panel_rank_rows[ i_ ];
                    if (i > j) {
                        if (C.tileIsLocal( i, j )
                            && device == C.tileDevice( i, j )) {
                            a_array00.push_back( A( i, 0, device ).data() );
                            b_array00.push_back( B( j, 0, device ).data() );
                            c_array00.push_back( C( i, j, device ).data() );
                            lda00 = A( i, 0, device ).stride();
                            ldb00 = B( j, 0, device ).stride();
                            ldc00 = C( i, j, device ).stride();
                        }
                    }
                    else if (i < j) {
                        if (C.tileIsLocal( j, i )
                            && device == C.tileDevice( j, i )) {
                            a_array00.push_back( B( j, 0, device ).data() );
                            b_array00.push_back( A( i, 0, device ).data() );
                            c_array00.push_back( C( j, i, device ).data() );
                            lda00 = B( j, 0, device ).stride();
                            ldb00 = A( i, 0, device ).stride();
                            ldc00 = C( j, i, device ).stride();
                        }
                    }
                    else { // i == j
                        // Diagonal tiles dealt with above.
                        assert( ! C.tileIsLocal( i, j ) );
                    }
                }
            }

            // last column if there is a clean-up tile
            std::vector<scalar_t*> a_array01;
            std::vector<scalar_t*> b_array01;
            std::vector<scalar_t*> c_array01;
            a_array01.reserve( batch_size );
            b_array01.reserve( batch_size );
            c_array01.reserve( batch_size );

            int64_t lda01 = 0;
            int64_t ldb01 = 0;
            int64_t ldc01 = 0;
            int64_t mb01 = C.tileMb( i0 );
            int64_t nb01 = C.tileNb( nt-1 );

            if (j_last == 1) {
                //for (int64_t j = 0; j < nt; ++j) {
                int64_t j = C.nt()-1;
                //for (int64_t i : panel_rank_rows) {
                for (int64_t i_ = 0; i_ < i_interior; ++i_) {
                    int64_t i = panel_rank_rows[ i_ ];
                    if (i > j) {
                        if (C.tileIsLocal( i, j )
                            && device == C.tileDevice( i, j )) {
                            a_array01.push_back( A( i, 0, device ).data() );
                            b_array01.push_back( B( j, 0, device ).data() );
                            c_array01.push_back( C( i, j, device ).data() );
                            lda01 = A( i, 0, device ).stride();
                            ldb01 = B( j, 0, device ).stride();
                            ldc01 = C( i, j, device ).stride();
                        }
                    }
                    else if (i < j) {
                        if (C.tileIsLocal( j, i )
                            && device == C.tileDevice( j, i )) {
                            a_array01.push_back( B( j, 0, device ).data() );
                            b_array01.push_back( A( i, 0, device ).data() );
                            c_array01.push_back( C( j, i, device ).data() );
                            mb01 = C.tileNb( nt-1 );
                            nb01 = C.tileMb( i0 );
                            lda01 = B( j, 0, device ).stride();
                            ldb01 = A( i, 0, device ).stride();
                            ldc01 = C( j, i, device ).stride();
                        }
                    }
                    else { // i == j
                        assert( ! C.tileIsLocal( i, j ) );
                    }
                }
            }

            // last row if there is a clean-up tile
            std::vector<scalar_t*> a_array10;
            std::vector<scalar_t*> b_array10;
            std::vector<scalar_t*> c_array10;
            a_array10.reserve( batch_size );
            b_array10.reserve( batch_size );
            c_array10.reserve( batch_size );

            int64_t lda10 = 0;
            int64_t ldb10 = 0;
            int64_t ldc10 = 0;
            int64_t mb10 = C.tileMb( i1 );
            int64_t nb10 = C.tileNb( 0 );

            if (i_last == 1) {
                int64_t i = i1;
                for (int64_t j = 0; j < j_interior; ++j) {
                    if (i > j) {
                        if (C.tileIsLocal( i, j )
                            && device == C.tileDevice( i, j )) {
                            a_array10.push_back( A( i, 0, device ).data() );
                            b_array10.push_back( B( j, 0, device ).data() );
                            c_array10.push_back( C( i, j, device ).data() );
                            lda10 = A( i, 0, device ).stride();
                            ldb10 = B( j, 0, device ).stride();
                            ldc10 = C( i, j, device ).stride();
                        }
                    }
                    else if (i < j) {
                        if (C.tileIsLocal( j, i )
                            && device == C.tileDevice( j, i )) {
                            a_array10.push_back( B( j, 0, device ).data() );
                            b_array10.push_back( A( i, 0, device ).data() );
                            c_array10.push_back( C( j, i, device ).data() );
                            mb10 = C.tileNb( 0 );
                            nb10 = C.tileMb( i1 );
                            lda10 = A( i, 0, device ).stride();
                            ldb10 = B( j, 0, device ).stride();
                            ldc10 = C( j, i, device ).stride();
                        }
                    }
                    else { // i == j
                        assert( ! C.tileIsLocal( i, j ) );
                    }
                }
            }

            // bottom-right corner
            std::vector<scalar_t*> a_array11;
            std::vector<scalar_t*> b_array11;
            std::vector<scalar_t*> c_array11;

            int64_t lda11 = 0;
            int64_t ldb11 = 0;
            int64_t ldc11 = 0;
            int64_t mb11 = C.tileMb( i1 );
            int64_t nb11 = C.tileNb( nt-1 );

            if (i_last == 1 && j_last == 1) {
                int64_t i = i1;
                int64_t j = nt-1;
                if (i > j) {
                    if (C.tileIsLocal( i, j )
                        && device == C.tileDevice( i, j)) {
                        a_array11.push_back( A( i, 0, device ).data() );
                        b_array11.push_back( B( j, 0, device ).data() );
                        c_array11.push_back( C( i, j, device ).data() );
                        lda11 = A( i, 0, device ).stride();
                        ldb11 = B( j, 0, device ).stride();
                        ldc11 = C( i, j, device ).stride();
                    }
                }
                else if (i < j) {
                    if (C.tileIsLocal( j, i )
                        && device == C.tileDevice( j, i )) {
                        a_array11.push_back( B( j, 0, device ).data() );
                        b_array11.push_back( A( i, 0, device ).data() );
                        c_array11.push_back( C( j, i, device ).data() );
                        mb11 = C.tileNb( nt-1 );
                        nb11 = C.tileMb( i1 );
                        lda11 = A( i, 0, device ).stride();
                        ldb11 = B( j, 0, device ).stride();
                        ldc11 = C( j, i, device ).stride();
                    }
                }
                else { // i == j
                    assert( ! C.tileIsLocal( i, j ) );
                }
            }

            {
                trace::Block trace_block( "blas::batch::gemm" );

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
                        m, n, kb_,
                        alpha_, a_array00, ldda,
                                b_array00, lddb,
                        beta_,  c_array00, lddc,
                        c_array00.size(), info, *queue );
                }

                if (c_array01.size() > 0) {
                    std::vector<int64_t>    m( 1,  mb01 );
                    std::vector<int64_t>    n( 1,  nb01 );
                    std::vector<int64_t> ldda( 1, lda01 );
                    std::vector<int64_t> lddb( 1, ldb01 );
                    std::vector<int64_t> lddc( 1, ldc01 );
                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, kb_,
                        alpha_, a_array01, ldda,
                                b_array01, lddb,
                        beta_,  c_array01, lddc,
                        c_array01.size(), info, *queue );
                }

                if (c_array10.size() > 0) {
                    std::vector<int64_t>    m( 1,  mb10 );
                    std::vector<int64_t>    n( 1,  nb10 );
                    std::vector<int64_t> ldda( 1, lda10 );
                    std::vector<int64_t> lddb( 1, ldb10 );
                    std::vector<int64_t> lddc( 1, ldc10 );
                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, kb_,
                        alpha_, a_array10, ldda,
                                b_array10, lddb,
                        beta_,  c_array10, lddc,
                        c_array10.size(), info, *queue );
                }

                if (c_array11.size() > 0) {
                    std::vector<int64_t>    m( 1,  mb11 );
                    std::vector<int64_t>    n( 1,  nb11 );
                    std::vector<int64_t> ldda( 1, lda11 );
                    std::vector<int64_t> lddb( 1, ldb11 );
                    std::vector<int64_t> lddc( 1, ldc11 );
                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, kb_,
                        alpha_, a_array11, ldda,
                                b_array11, lddb,
                        beta_,  c_array11, lddc,
                        c_array11.size(), info, *queue );
                }
                queue->sync();
            }
            for (int64_t j = 0; j < nt; ++j) {
                for (int64_t i : panel_rank_rows) {
                    if (i > j) {
                        if (C.tileIsLocal( i, j )
                            && device == C.tileDevice( i, j )) {
                            // erase tmp local and remote device tiles;
                            A.tileRelease( i, 0, device );
                            B.tileRelease( j, 0, device );
                            // decrement life for remote tiles
                            A.tileTick( i, 0 );
                            B.tileTick( j, 0 );
                        }
                    }
                    else if (i < j) {
                        if (C.tileIsLocal( j, i )
                            && device == C.tileDevice( j, i )) {
                            // erase tmp local and remote device tiles;
                            A.tileRelease( i, 0, device );
                            B.tileRelease( j, 0, device );
                            // decrement life for remote tiles
                            A.tileTick( i, 0 );
                            B.tileTick( j, 0 );
                        }
                    }
                }
            }
        }
    }

    if (err)
        slate_error( std::to_string( err ) );
}

//------------------------------------------------------------------------------
/// matrix multiply to update trailing matrix, except the diagonal tiles.
/// where A is a single block column and B is a single block row.
/// Host nest implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_her2k_offdiag_ranks(
    internal::TargetType<Target::HostNest>,
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index )
{
    slate_not_implemented( "Target::HostNest isn't yet supported." );
}

//------------------------------------------------------------------------------
/// matrix multiply to update trailing matrix, except the diagonal tiles.
/// where A is a single block column and B is a single block row.
/// Host batched implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_her2k_offdiag_ranks(
    internal::TargetType<Target::HostBatch>,
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index )
{
    slate_not_implemented( "Target::HostBatch isn't yet supported." );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  HermitianMatrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  HermitianMatrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  HermitianMatrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  HermitianMatrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  HermitianMatrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  HermitianMatrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  HermitianMatrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  HermitianMatrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  HermitianMatrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  HermitianMatrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  HermitianMatrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  HermitianMatrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  HermitianMatrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  HermitianMatrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  HermitianMatrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  HermitianMatrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

} // namespace internal
} // namespace slate
