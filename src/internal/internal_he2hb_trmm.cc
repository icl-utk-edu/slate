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
/// Triangular matrix multiply. Compute B = B A
/// AH is a Hermitian matrix. It needed here just to check if the rank is an
/// upper or lower rank that contribute to compute Bi, i = 0:mt-1.
/// B is a block cloumn.
/// A contains upper triangular or trapezoid T.
/// indices contains the local indices for panel_rank,
/// If A contains upper triangular T, then call trmm B = B T
/// If A contains trapezoid T, then the slice
/// T = A[ 0:A.mb(), 0:A.mb() ] is upper triangular,
/// Bi = Bi[ 0:B.mb(), 0:A.mb() ]. Call trmm Bi = Bi T.
/// Dispatches to target implementations.
/// @ingroup heev_internal
///
template <Target target, typename scalar_t>
void he2hb_trmm(
    HermitianMatrix<scalar_t>&& AH, Matrix<scalar_t>&& A,
    Matrix<scalar_t>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    he2hb_trmm( internal::TargetType<target>(),
                AH, A, B,
                panel_rank_rows, priority, queue_index );
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host OpenMP task implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_trmm(
    internal::TargetType<Target::HostTask>,
    HermitianMatrix<scalar_t>& AH,
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    const scalar_t one  = 1;
    int my_rank = AH.mpiRank();

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    auto A0 = A.sub( 0, 0, 0, 0 );

    #pragma omp taskgroup
    for (int64_t i = 0; i < B.mt(); ++i) {
        #pragma omp task shared( AH, B )
        {
            int rank_lower = -1;
            int rank_upper = -1;
            for (int64_t j : panel_rank_rows) {
                if (i >= j) { // lower
                    rank_lower = AH.tileRank( i, j );
                }
                else { // upper
                    rank_upper = AH.tileRank( j, i );
                }
            }
            // If I contributed to Bi, multiply by A.
            if (rank_upper == my_rank || rank_lower == my_rank) {
                // Bi = Bi * A
                auto Bi = B.sub( i, i, 0, 0 );

                int64_t mb = A0.tileMb( 0 );
                int64_t nb = A0.tileNb( 0 );
                bool trapezoid = (mb < nb);

                B.tileGetForWriting( i, 0, layout_conv );
                if (trapezoid) {
                    auto B00 = Bi( 0, 0 );
                    int64_t mb1 = B00.mb();
                    A0 = A0.slice( 0, mb-1,  0, mb-1 ); // first mb-by-mb part
                    Bi = Bi.slice( 0, mb1-1, 0, mb-1 ); // first mb1-by-mb part
                }

                auto T = TriangularMatrix<scalar_t>( Uplo::Upper, Diag::NonUnit, A0 );
                tile::trmm( Side::Right, Diag::NonUnit,
                            one, std::move( T( 0, 0 ) ), Bi( 0, 0 ) );

                B.tileTick( i, 0 );
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Device implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_trmm(
    internal::TargetType<Target::Devices>,
    HermitianMatrix<scalar_t>& AH,
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index )
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    int my_rank = AH.mpiRank();

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layout_conv = LayoutConvert( layout );

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared( AH, B ) priority( priority )
        {
            std::set<ij_tuple> B_tiles_set, A0_tiles_set;
            int rank_lower = -1;
            int rank_upper = -1;

            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j : panel_rank_rows) {
                    if (i >= j) { // lower
                        rank_lower = AH.tileRank( i, j );
                    }
                    else { // upper
                        rank_upper = AH.tileRank( j, i );
                    }
                }

                if (rank_upper == my_rank || rank_lower == my_rank) {
                    if (device == B.tileDevice( i, 0 )) {
                        B_tiles_set.insert( { i, 0 } );
                    }
                }
            }

            int64_t i_interior = B.mt();
            int64_t i_last = 0;
            int64_t mt = B.mt();
            if (B.tileMb( mt-1 ) != B.tileMb( 0 )) {
                i_interior = B.mt() - 1;
                i_last = 1;
            }

            int64_t batch_size = B_tiles_set.size();
            if (batch_size > 0) {

                auto A0 = A.sub( 0, 0, 0, 0 );
                A0.tileGetForReading( 0, 0, device, layout_conv );
                B.tileGetForWriting( B_tiles_set, device, layout_conv );

                // interior
                std::vector<scalar_t*> b_array0;
                std::vector<scalar_t*> a_array0;
                a_array0.reserve( batch_size );
                b_array0.reserve( batch_size );

                // bottom-right tile
                std::vector<scalar_t*> a_array1;
                std::vector<scalar_t*> b_array1;

                int64_t lda0 = 0;
                int64_t ldb0 = 0;
                int64_t lda1 = 0;
                int64_t ldb1 = 0;

                int64_t mb0 = B.tileMb( 0 );
                int64_t nb0 = B.tileNb( 0 );
                int64_t mb1 = B.tileMb( B.mt()-1 );
                int64_t nb1 = B.tileNb( B.mt()-1 );

                rank_lower = -1;
                rank_upper = -1;

                for (int64_t i = 0; i < i_interior; ++i) {
                    for (int64_t j : panel_rank_rows) {
                        if (i >= j) { // lower
                            rank_lower = AH.tileRank( i, j );
                        }
                        else { // upper
                            rank_upper = AH.tileRank( j, i );
                        }
                    }
                    A0 = A.sub( 0, 0, 0, 0 );
                    int64_t mb = A0.tileMb( 0 );
                    int64_t nb = A0.tileNb( 0 );
                    auto Bi = B.sub( i, i, 0, 0 );
                    bool trapezoid = (mb < nb);

                    if (trapezoid) {
                        auto B00 = Bi( 0, 0 );
                        mb1 = B00.mb();
                        A0 = A0.slice( 0, mb-1,  0, mb-1 ); // first mb-by-mb part
                        Bi = Bi.slice( 0, mb1-1, 0, mb-1 ); // first mb1-by-mb part
                    }
                    auto T = TriangularMatrix<scalar_t>( Uplo::Upper, Diag::NonUnit, A0 );

                    if (rank_upper == my_rank || rank_lower == my_rank) {
                        if (device == B.tileDevice( i, 0 )) {
                            a_array0.push_back( T( 0, 0, device ).data() );
                            b_array0.push_back( Bi( 0, 0, device ).data() );
                            //b_array0.push_back( B( i, 0, device ).data() );
                            lda0 = A0( 0, 0, device ).stride();
                            ldb0 = Bi( 0, 0, device ).stride();
                            mb0 = Bi.tileMb( 0 );
                            nb0 = Bi.tileNb( 0 );
                        }
                    }
                }

                if (i_last == 1) {
                    int64_t i = B.mt()-1;
                    rank_lower = -1;
                    rank_upper = -1;
                    for (int64_t j : panel_rank_rows) {
                        if (i >= j) { // lower
                            rank_lower = AH.tileRank( i, j );
                        }
                        else { // upper
                            rank_upper = AH.tileRank( j, i );
                        }
                    }
                    A0 = A.sub( 0, 0, 0, 0 );
                    int64_t mb = A0.tileMb( 0 );
                    int64_t nb = A0.tileNb( 0 );
                    auto Bi = B.sub( i, i, 0, 0 );
                    bool trapezoid = (mb < nb);

                    if (trapezoid) {
                        auto B00 = Bi( 0, 0 );
                        mb1 = B00.mb();
                        A0 = A0.slice( 0, mb-1,  0, mb-1 ); // first mb-by-mb part
                        Bi = Bi.slice( 0, mb1-1, 0, mb-1 ); // first mb1-by-mb part
                    }
                    auto T = TriangularMatrix<scalar_t>( Uplo::Upper, Diag::NonUnit, A0 );
                    if (rank_upper == my_rank || rank_lower == my_rank) {
                        if (device == B.tileDevice( i, 0 )) {
                            a_array1.push_back( T( 0, 0, device ).data() );
                            b_array1.push_back( Bi( 0, 0, device ).data() );
                            lda1 = T( 0, 0, device ).stride();
                            ldb1 = Bi( 0, 0, device ).stride();
                            mb1 = Bi.tileMb( 0 );
                            nb1 = Bi.tileNb( 0 );
                        }
                    }
                }

                {
                    trace::Block trace_block( "blas::batch::he2hb_trmm" );
                    blas::Queue* queue = B.compute_queue( device, queue_index );
                    assert( queue != nullptr );

                    Side sideB = Side::Right;
                    Uplo uploB = Uplo::Upper;
                    Op opB = Op::NoTrans;
                    Diag diagB = Diag::NonUnit;
                    scalar_t alpha = 1.;
                    std::vector<Side>      side_( 1, sideB );
                    std::vector<Uplo>      uplo_( 1, uploB );
                    std::vector<Op>         opA_( 1, opB   );
                    std::vector<Diag>      diag_( 1, diagB );
                    std::vector<scalar_t> alpha_( 1, alpha );
                    std::vector<int64_t>   info;

                    if (b_array0.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb0 );
                        std::vector<int64_t>    n( 1,  nb0 );
                        std::vector<int64_t>  ldb( 1, ldb0 );
                        std::vector<int64_t>  lda( 1, lda0 );
                        blas::batch::trmm(
                            layout, side_, uplo_, opA_, diag_,
                            m, n,
                            alpha_, a_array0, lda,
                            b_array0, ldb,
                            a_array0.size(), info, *queue );
                    }

                    if (b_array1.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb1 );
                        std::vector<int64_t>    n( 1,  nb1 );
                        std::vector<int64_t>  lda( 1, lda1 );
                        std::vector<int64_t>  ldb( 1, ldb1 );
                        blas::batch::trmm(
                            layout, side_, uplo_, opA_, diag_,
                            m, n,
                            //m, lda,
                            alpha_, a_array1, lda,
                            b_array1, ldb,
                            a_array1.size(), info, *queue );
                    }

                    queue->sync();
                }

                rank_lower = -1;
                rank_upper = -1;
                for (int64_t i = 0; i < B.mt(); ++i) {
                    for (int64_t j : panel_rank_rows) {
                        if (i >= j) { // lower
                            rank_lower = AH.tileRank( i, j );
                        }
                        else { // upper
                            rank_upper = AH.tileRank( j, i );
                        }
                    }

                    if (rank_upper == my_rank || rank_lower == my_rank) {
                        if (device == B.tileDevice( i, 0 )) {
                            B.tileRelease( i, 0, device );
                            B.tileTick( i, 0 );
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host nested OpenMP implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_trmm(
    internal::TargetType<Target::HostNest>,
    HermitianMatrix<scalar_t>& AH,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& A,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    slate_not_implemented( "Target::HostNest isn't yet supported." );
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host batched OpenMP implementation.
/// @ingroup heev_internal
///
template <typename scalar_t>
void he2hb_trmm(
    internal::TargetType<Target::HostBatch>,
    HermitianMatrix<scalar_t>& AH,
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index)
{
    slate_not_implemented( "Target::HostBatch isn't yet supported." );
}


//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void he2hb_trmm<Target::HostTask, float>(
    HermitianMatrix<float>&& AH,
    Matrix<float>&& A,
    Matrix<float>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostTask, double>(
    HermitianMatrix<double>&& AH,
    Matrix<double>&& A,
    Matrix<double>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& AH,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& AH,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::Devices, float>(
    HermitianMatrix<float>&& AH,
    Matrix<float>&& A,
    Matrix<float>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::Devices, double>(
    HermitianMatrix<double>&& AH,
    Matrix<double>&& A,
    Matrix<double>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::Devices, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& AH,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::Devices, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& AH,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostNest, float>(
    HermitianMatrix<float>&& AH,
    Matrix<float>&& A,
    Matrix<float>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostNest, double>(
    HermitianMatrix<double>&& AH,
    Matrix<double>&& A,
    Matrix<double>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostNest, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& AH,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostNest, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& AH,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostBatch, float>(
    HermitianMatrix<float>&& AH,
    Matrix<float>&& A,
    Matrix<float>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostBatch, double>(
    HermitianMatrix<double>&& AH,
    Matrix<double>&& A,
    Matrix<double>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostBatch, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& AH,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostBatch, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& AH,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index);

} // namespace internal
} // namespace slate
