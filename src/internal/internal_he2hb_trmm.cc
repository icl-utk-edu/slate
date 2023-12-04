// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Determines whether this process contributes to B(i, 0).
/// Specifically, it checks whether there is a j in panel_rank_rows such that
/// AH(i, j) is local (taking into account the symmetric storage.)
///
template <typename scalar_t>
bool need_Bi0(HermitianMatrix<scalar_t> AH,
              int mpi_rank,
              int64_t i,
              std::vector<int64_t>& panel_rank_rows)
{
    for (int64_t j : panel_rank_rows) {
        if (i >= j) { // lower
            if (AH.tileRank( i, j ) == mpi_rank) {
                return true;
            }
        }
        else {
            if (AH.tileRank( j, i ) == mpi_rank) {
                return true;
            }
        }
    }
    return false;
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply. Compute B = B A
/// AH is a Hermitian matrix. It's needed here just to check if the rank is an
/// upper or lower rank that contribute to compute Bi, i = 0:mt-1.
/// B is a block column.
/// A contains upper triangular or trapezoid T.
/// indices contains the local indices for panel_rank,
/// If A contains upper triangular T, then call trmm B = B T
/// If A contains trapezoid T, then the slice
/// T = A[ 0:A.mb(), 0:A.mb() ] is upper triangular,
/// Bi = Bi[ 0:B.mb(), 0:A.mb() ]. Call trmm Bi = Bi T.
/// Dispatches to target implementations.
///
/// panel_rank_rows contains the local row-indices of B
///
/// @ingroup heev_internal
///
template <Target target, typename scalar_t>
void he2hb_trmm(
    HermitianMatrix<scalar_t>&& AH, Matrix<scalar_t>&& A,
    Matrix<scalar_t>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index )
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
    int priority, int64_t queue_index )
{
    const scalar_t one  = 1;
    int mpi_rank = AH.mpiRank();

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layoutc = LayoutConvert( layout );

    if (panel_rank_rows.size() == 0) {
        return;
    }
    auto A0 = A.sub( 0, 0, 0, 0 );
    int64_t mb = A0.tileMb( 0 );
    int64_t nb = A0.tileNb( 0 );
    bool trapezoid = (mb < nb);
    if (trapezoid) {
        A0 = A0.slice( 0, mb-1,  0, mb-1 ); // first mb-by-mb part
    }

    #pragma omp taskgroup
    for (int64_t i = 0; i < B.mt(); ++i) {
        #pragma omp task slate_omp_default_none \
            shared( A0, AH, B, panel_rank_rows ) \
            firstprivate( one, i, mpi_rank, layoutc, mb, trapezoid ) \
            priority( priority )
        {
            // If I contributed to Bi, multiply by A.
            if (need_Bi0( AH, mpi_rank, i, panel_rank_rows )) {
                // Bi = Bi * A
                auto Bi = B.sub( i, i, 0, 0 );

                B.tileGetForWriting( i, 0, layoutc );
                if (trapezoid) {
                    auto B00 = Bi( 0, 0 );
                    int64_t mb1 = B00.mb();
                    Bi = Bi.slice( 0, mb1-1, 0, mb-1 ); // first mb1-by-mb part
                }

                auto T = TriangularMatrix<scalar_t>( Uplo::Upper, Diag::NonUnit, A0 );
                tile::trmm( Side::Right, Diag::NonUnit,
                            one, std::move( T( 0, 0 ) ), Bi( 0, 0 ) );
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

    int mpi_rank = AH.mpiRank();

    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layoutc = LayoutConvert( layout );

    if (panel_rank_rows.size() == 0) {
        return;
    }

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task slate_omp_default_none \
            shared( A, AH, B, panel_rank_rows ) \
            firstprivate( device, queue_index, mpi_rank, layout, layoutc ) \
            priority( priority )
        {
            std::set<ij_tuple> B_tiles_set, A0_tiles_set;

            for (int64_t i = 0; i < B.mt(); ++i) {
                if (need_Bi0( AH, mpi_rank, i, panel_rank_rows )
                    && device == B.tileDevice( i, 0 )) {
                    B_tiles_set.insert( { i, 0 } );
                }
            }

            int64_t batch_size = B_tiles_set.size();
            if (batch_size > 0) {

                auto A0 = A.sub( 0, 0, 0, 0 );
                A0.tileGetForReading( 0, 0, device, layoutc );
                B.tileGetForWriting( B_tiles_set, device, layoutc );

                // interior
                std::vector<scalar_t*> b_array0;
                std::vector<scalar_t*> a_array0;
                a_array0.reserve( batch_size );
                b_array0.reserve( batch_size );

                // bottom-right tile
                std::vector<scalar_t*> a_array1;
                std::vector<scalar_t*> b_array1;

                int64_t mb = A0.tileMb( 0 );
                int64_t nb = A0.tileNb( 0 );
                bool trapezoid = (mb < nb);
                if (trapezoid) {
                    A0 = A0.slice( 0, mb-1,  0, mb-1 ); // first mb-by-mb part
                }
                auto T = TriangularMatrix<scalar_t>( Uplo::Upper, Diag::NonUnit, A0 );

                scalar_t** t_array_host = B.array_host(device, queue_index);
                scalar_t** b_array_host = t_array_host + batch_size;

                // Variant of device_regions_build to handle he2hb_trmm
                using Params = device_regions_params<false, 2>;

                // Find ranges of matching mb's and ranges of matching nb's.
                auto irange = device_regions_range( RowCol::Row, B );

                // loop over regions
                int64_t batch_count = 0;
                std::vector<Params> group_params;
                for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                    // Loop over the tiles in this region,
                    // save any that should be computed on this process & device
                    Params group;
                    group.mb = B.tileMb( irange[ ii ] );
                    group.nb = T.tileMb( 0 );
                    for (int64_t i = irange[ ii ]; i < irange[ ii+1 ]; ++i) {
                        if (need_Bi0( AH, mpi_rank, i, panel_rank_rows )
                            && device == B.tileDevice( i, 0 )) {

                            // Add tiles to current group
                            auto Bi = B.sub( i, i, 0, 0 );
                            if (trapezoid) {
                                auto B00 = Bi( 0, 0 );
                                int64_t mb1 = B00.mb();
                                Bi = Bi.slice( 0, mb1-1, 0, mb-1 ); // first mb1-by-mb part
                            }

                            auto Tij = T( 0, 0, device );
                            t_array_host[ batch_count ] = Tij.data();
                            auto Bij = Bi( 0, 0, device );
                            b_array_host[ batch_count ] = Bij.data();
                            if (group.count == 0) {
                                group.ld[0] = Tij.stride();
                                group.ld[1] = Bij.stride();
                            }
                            else {
                                //assert( group.ld[0] == Tij.stride() );
                                //assert( group.ld[1] == Bij.stride() );
                            }
                            ++group.count;
                            ++batch_count;
                        }
                    } // for i
                    // If any tiles in the region should be computed here, save the group
                    if (group.count > 0) {
                        group_params.push_back( group );
                    }
                } // for ii

                {
                    trace::Block trace_block( "blas::batch::he2hb_trmm" );
                    blas::Queue* queue = B.compute_queue( device, queue_index );
                    // assert conflicts with default(none) in old gcc.
                    //assert( queue != nullptr );

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

                    for (size_t g = 0; g < group_params.size(); ++g) {

                        int64_t group_count = group_params[ g ].count;
                        std::vector<int64_t>    m( 1, group_params[g].mb );
                        std::vector<int64_t>    n( 1, group_params[g].nb );
                        std::vector<int64_t> ldda( 1, group_params[g].ld[0] );
                        std::vector<int64_t> lddb( 1, group_params[g].ld[1] );

                        std::vector<scalar_t*> t_array(t_array_host, t_array_host+group_count);
                        std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);

                        blas::batch::trmm(
                            layout, side_, uplo_, opA_, diag_,
                            m, n,
                            alpha_, t_array, ldda,
                                    b_array, lddb,
                            group_count, info, *queue );

                        t_array_host += group_count;
                        b_array_host += group_count;
                    }

                    queue->sync();
                }
            }
        }
    }
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
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_trmm<Target::HostTask, double>(
    HermitianMatrix<double>&& AH,
    Matrix<double>&& A,
    Matrix<double>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_trmm< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& AH,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_trmm< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& AH,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_trmm<Target::Devices, float>(
    HermitianMatrix<float>&& AH,
    Matrix<float>&& A,
    Matrix<float>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_trmm<Target::Devices, double>(
    HermitianMatrix<double>&& AH,
    Matrix<double>&& A,
    Matrix<double>&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_trmm< Target::Devices, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& AH,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_trmm< Target::Devices, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& AH,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& B,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

} // namespace internal
} // namespace slate
