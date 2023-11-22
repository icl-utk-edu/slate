// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
    const LayoutConvert layoutc = LayoutConvert( layout );

    assert( A.nt() == B.mt() );

    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        #pragma omp task slate_omp_default_none \
            shared( A, B, C ) \
            firstprivate( alpha, beta, panel_rank, i, layoutc ) \
            priority( priority )
        {
            scalar_t beta_ = beta;
            for (int64_t k = 0; k < A.nt(); ++k) {
                if (A.tileRank( i, k ) == panel_rank) {
                    A.tileGetForReading( i, k, layoutc );
                    B.tileGetForReading( k, 0, layoutc );
                    C.tileGetForWriting( i, 0, layoutc );
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
    const LayoutConvert layoutc = LayoutConvert( layout );

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
        #pragma omp task slate_omp_default_none \
            shared( A, B, C, err ) \
            firstprivate( alpha, beta, panel_rank, queue_index, device, \
                          layout, layoutc ) \
            priority( priority )
        {
            Op opA = A.op();
            Op opB = B.op();

            scalar_t** host_work = A.array_host(device, queue_index);

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

                int64_t batch_size = C_tiles_set.size();

                scalar_t** a_array_host = host_work;
                scalar_t** b_array_host = a_array_host + batch_size;
                scalar_t** c_array_host = b_array_host + batch_size;

                // Variant of device_regions_build to handle he2hb_gemm
                using Params = device_regions_params<false, 3>;

                // Find ranges of matching mb's and ranges of matching nb's.
                auto irange = device_regions_range( RowCol::Row, C );

                // loop over regions
                int64_t batch_count = 0;
                std::vector<Params> group_params;
                for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                    // Loop over the tiles in this region,
                    // save any that should be computed on this process & device
                    Params group;
                    group.mb = C.tileMb( irange[ ii ] );
                    group.nb = C.tileNb( 0 );
                    for (int64_t i = irange[ ii ]; i < irange[ ii+1 ]; ++i) {
                        if (A.tileRank( i, k ) == panel_rank
                            && device == C.tileDevice( i, 0 )) {

                            // Add tiles to current group
                            auto Aij = A( i, k, device );
                            a_array_host[ batch_count ] = Aij.data();
                            auto Bij = B( k, 0, device );
                            b_array_host[ batch_count ] = Bij.data();
                            auto Cij = C( i, 0, device );
                            c_array_host[ batch_count ] = Cij.data();
                            if (group.count == 0) {
                                group.ld[0] = Aij.stride();
                                group.ld[1] = Bij.stride();
                                group.ld[2] = Cij.stride();
                            }
                            else {
                                // default(none) doesn't allow asserts
                                //assert( group.ld[0] == Aij.stride() );
                                //assert( group.ld[1] == Bij.stride() );
                                //assert( group.ld[2] == Bij.stride() );
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

                if (C.op() != Op::NoTrans) {
                    // swap A <=> B; swap m <=> n
                    swap( opA, opB );
                }

                {
                    trace::Block trace_block( "blas::batch::he2hb_gemm" );

                    std::vector<Op> opA_(1, opA);
                    std::vector<Op> opB_(1, opB);
                    std::vector<scalar_t> alpha_(1, alpha);
                    std::vector<scalar_t> beta_(1, beta);
                    std::vector<int64_t> kb(1, A.tileNb(k));
                    // info size 0 disables slow checks in batched BLAS++.
                    std::vector<int64_t> info;

                    blas::Queue* queue = C.compute_queue(device, queue_index);

                    for (size_t g = 0; g < group_params.size(); ++g) {

                        int64_t group_count = group_params[ g ].count;

                        std::vector<int64_t>    m(1, group_params[ g ].mb);
                        std::vector<int64_t>    n(1, group_params[ g ].nb);
                        std::vector<int64_t> ldda(1, group_params[ g ].ld[0]);
                        std::vector<int64_t> lddb(1, group_params[ g ].ld[1]);
                        std::vector<int64_t> lddc(1, group_params[ g ].ld[2]);

                        std::vector<scalar_t*> a_array(a_array_host, a_array_host+group_count);
                        std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);
                        std::vector<scalar_t*> c_array(c_array_host, c_array_host+group_count);

                        if (C.op() != Op::NoTrans) {
                            swap(m, n);
                            swap(a_array, b_array);
                            swap(ldda, lddb);
                        }

                        blas::batch::gemm(
                            layout, opA_, opB_,
                            m, n, kb,
                            alpha_, a_array, ldda,
                                    b_array, lddb,
                            beta_,  c_array, lddc,
                            group_count, info, *queue);

                        a_array_host += group_count;
                        b_array_host += group_count;
                        c_array_host += group_count;
                    }
                    queue->sync();
                }

                // todo: release tiles in top-level routine.
                // for (int64_t i = 0; i < A.mt(); ++i) {
                //     if (A.tileRank( i, k ) == panel_rank
                //         && device == C.tileDevice( i, 0 )) {
                //         // erase tmp local and remote device tiles;
                //         A.tileRelease( i, k, device );
                //         B.tileRelease( k, 0, device );
                //         // decrement life for remote tiles
                //         A.tileTick( i, k );
                //         B.tileTick( k, 0 );
                //     }
                // }
                beta = 1.0;
            } // for loop (k)
        } // pragma
    } // device

    if (err)
        slate_error( std::to_string( err ) );
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

} // namespace internal
} // namespace slate
