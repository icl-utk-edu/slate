// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

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
    int priority, int64_t queue_index )
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layoutc = LayoutConvert( layout );

    int64_t nt = C.nt();

    // try to loop over one tile and do two gemm, similar to her2k
    #pragma omp taskgroup
    for (int64_t j = 0; j < nt; ++j) {
        #pragma omp task slate_omp_default_none \
            shared( A, B, C, panel_rank_rows ) \
            firstprivate( alpha, beta, j, layoutc )
        {
            for (int64_t i : panel_rank_rows) {
                // todo: if HermitianMatrix returned conjTrans
                // tiles, could merge these two.
                if (i > j) {  // lower
                    if (C.tileIsLocal( i, j )) {
                        A.tileGetForReading( i, 0, layoutc );
                        B.tileGetForReading( j, 0, layoutc );
                        C.tileGetForWriting( i, j, layoutc );
                        // Aij -= Vik Wjk^H
                        tile::gemm( alpha, A( i, 0 ), conj_transpose( B( j, 0 ) ),
                                    beta, C( i, j ) );
                    }
                }
                else if (i < j) {  // upper
                    if (C.tileIsLocal( j, i )) {
                        B.tileGetForReading( j, 0, layoutc );
                        A.tileGetForReading( i, 0, layoutc );
                        C.tileGetForWriting( j, i, layoutc );
                        // Aji -= Wjk Vik^H
                        tile::gemm( alpha, B( j, 0 ), conj_transpose( A( i, 0 ) ),
                                    beta, C( j, i ) );
                    }
                }
                else { // i == j
                    // Diagonal tiles dealt with above.
                    // assert conflicts with default(none) in old gcc.
                    //assert( ! C.tileIsLocal( i, j ) );
                }
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
    int priority, int64_t queue_index )
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;
    const LayoutConvert layoutc = LayoutConvert( layout );

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
        #pragma omp task slate_omp_default_none \
            shared( A, B, C, err, panel_rank_rows ) \
            firstprivate( alpha, beta, device, nt, layout, layoutc, queue_index ) \
            priority( priority )
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
                        // Diagonal tiles dealt with elsewhere in he2hb.
                        // assert conflicts with default(none) in old gcc.
                        //assert( ! C.tileIsLocal( i, j ) );
                    }
                }
            }
            int64_t batch_size = C_tiles_set.size();

            #pragma omp taskgroup
            {
                #pragma omp task slate_omp_default_none \
                    shared( A, A_tiles_set ) \
                    firstprivate( device, layoutc )
                {
                    A.tileGetForReading( A_tiles_set, device, layoutc );
                }
                #pragma omp task slate_omp_default_none \
                    shared( B, B_tiles_set ) \
                    firstprivate( device, layoutc )
                {
                    B.tileGetForReading( B_tiles_set, device, layoutc );
                }
                #pragma omp task slate_omp_default_none \
                    shared( C, C_tiles_set ) \
                    firstprivate( device, layoutc )
                {
                    C.tileGetForWriting( C_tiles_set, device, layoutc );
                }
            }

            scalar_t** a_array_host = C.array_host(device, queue_index);
            scalar_t** b_array_host = a_array_host + batch_size;
            scalar_t** c_array_host = b_array_host + batch_size;

            using Params = device_regions_params<false, 3>;

            // Find ranges of matching mb's and ranges of matching nb's.
            auto jrange = device_regions_range( RowCol::Col, C );

            std::vector< int64_t > irange;
            int64_t last_ij = -1;
            for (int64_t k = 0; k < int64_t(panel_rank_rows.size()); ++k) {
                int64_t kb = panel_rank_rows[ k ];
                if (kb != last_ij) {
                    last_ij = kb;
                    irange.push_back( k );
                }
            }
            irange.push_back( panel_rank_rows.size() );

            int64_t batch_count = 0;
            std::vector<Params> group_params;
            // loop over regions
            for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
            for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                // Loop over the tiles in this region,
                // save any that should be computed on this process & device
                // Two groups are needed to handle the different sizes
                Params group;
                group.mb = C.tileMb( panel_rank_rows[ irange[ ii ] ] );
                group.nb = C.tileNb( jrange[ jj ] );

                for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                for (int64_t i_ = irange[ ii ]; i_ < irange[ ii+1 ]; ++i_) {
                    int64_t i = panel_rank_rows[ i_ ];

                    if ((i > j)
                        && C.tileIsLocal( i, j ) && device == C.tileDevice( i, j )) {

                        // Add tiles to current group
                        auto Aij = A( i, 0, device );
                        auto Bij = B( j, 0, device );
                        auto Cij = C( i, j, device );

                        a_array_host[ batch_count ]  = Aij.data();
                        b_array_host[ batch_count ]  = Bij.data();
                        c_array_host[ batch_count ]  = Cij.data();

                        if (group.count == 0) {
                            group.ld[0] = Aij.stride();
                            group.ld[1] = Bij.stride();
                            group.ld[2] = Cij.stride();
                        }
                        else {
                            // assert( group.ld[0] == Aij.stride() );
                            // assert( group.ld[1] == Bij.stride() );
                            // assert( group.ld[2] == Cij.stride() );
                        }

                        ++group.count;
                        ++batch_count;
                    }
                }} // for j, i

                // If mb != nb, we need to start a new group for the upper
                // triangular logic
                // If the problem is square, we can use a single group for
                // better parallelism
                if (group.mb != group.nb) {
                    // If any tiles in the region should be computed here, save the group
                    if (group.count > 0) {
                        group_params.push_back( group );
                    }

                    std::swap( group.mb, group.nb );
                    group.count = 0;
                    group.ld[0] = 0;
                    group.ld[1] = 0;
                    group.ld[2] = 0;
                }

                for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                for (int64_t i_ = irange[ ii ]; i_ < irange[ ii+1 ]; ++i_) {
                    int64_t i = panel_rank_rows[ i_ ];

                    if ((i < j)
                        && C.tileIsLocal( j, i ) && device == C.tileDevice( j, i )) {

                        // Add tiles to current group
                        auto Aij = B( j, 0, device );
                        auto Bij = A( i, 0, device );
                        auto Cij = C( j, i, device );

                        a_array_host[ batch_count ] = Aij.data();
                        b_array_host[ batch_count ] = Bij.data();
                        c_array_host[ batch_count ] = Cij.data();

                        if (group.count == 0) {
                            group.ld[0] = Aij.stride();
                            group.ld[1] = Bij.stride();
                            group.ld[2] = Cij.stride();
                        }
                        else {
                            // assert( group.ld[0] == Aij.stride() );
                            // assert( group.ld[1] == Bij.stride() );
                            // assert( group.ld[2] == Cij.stride() );
                        }

                        ++group.count;
                        ++batch_count;
                    }
                }} // for j, i
                // If any tiles in the region should be computed here, save the group
                if (group.count > 0) {
                    group_params.push_back( group );
                }
            }} // for jj, ii

            {
                trace::Block trace_block( "blas::batch::gemm" );

                std::vector<Op> opA_( 1, opA );
                std::vector<Op> opB_( 1, opB );
                std::vector<scalar_t> alpha_( 1, alpha );
                std::vector<scalar_t> beta_( 1, beta );
                std::vector<int64_t> kb_( 1, A.tileNb(0) );
                // info size 0 disables slow checks in batched BLAS++.
                std::vector<int64_t> info;

                blas::Queue* queue = C.compute_queue( device, queue_index );
                // assert conflicts with default(none) in old gcc.
                //assert( queue != nullptr );

                for (size_t g = 0; g < group_params.size(); ++g) {

                    int64_t group_count = group_params[ g ].count;

                    std::vector<int64_t>    m( 1, group_params[g].mb );
                    std::vector<int64_t>    n( 1, group_params[g].nb );
                    std::vector<int64_t> ldda( 1, group_params[g].ld[0] );
                    std::vector<int64_t> lddb( 1, group_params[g].ld[1] );
                    std::vector<int64_t> lddc( 1, group_params[g].ld[2] );

                    std::vector<scalar_t*> a_array(a_array_host, a_array_host+group_count);
                    std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);
                    std::vector<scalar_t*> c_array(c_array_host, c_array_host+group_count);

                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, kb_,
                        alpha_, a_array, ldda,
                                b_array, lddb,
                        beta_,  c_array, lddc,
                        group_count, info, *queue );
                    a_array_host += group_count;
                    b_array_host += group_count;
                    c_array_host += group_count;
                }
                queue->sync();
            }
        }
    }

    if (err)
        slate_error( std::to_string( err ) );
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
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  HermitianMatrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  HermitianMatrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  HermitianMatrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  HermitianMatrix<float>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  HermitianMatrix<double>&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  HermitianMatrix< std::complex<float> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

// ----------------------------------------
template
void he2hb_her2k_offdiag_ranks< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  HermitianMatrix< std::complex<double> >&& C,
    std::vector<int64_t>& panel_rank_rows,
    int priority, int64_t queue_index );

} // namespace internal
} // namespace slate
