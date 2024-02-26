// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Dispatches to target implementations.
/// @ingroup trsm_internal
///
template <Target target, typename scalar_t>
void trsm(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority, Layout layout, int64_t queue_index )
{
    trsm(internal::TargetType<target>(),
         side,
         alpha, A,
                B,
         priority, layout, queue_index );
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host OpenMP task implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm(internal::TargetType<Target::HostTask>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, Layout layout, int64_t queue_index )
{
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::trsm()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'B(i, j).layout()'
    assert(layout == Layout::ColMajor);
    assert(A.mt() == 1);

    if (B.numLocalTiles() > 0) {
        A.tileGetForReading(0, 0, LayoutConvert(layout));
    }
    // alternatively, if (side == right), (conj)-transpose both A and B,
    // then assume side == left; see slate::trsm
    #pragma omp taskgroup
    if (side == Side::Right) {
        assert(B.nt() == 1);
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, 0)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B ) \
                    firstprivate( i, layout, side, alpha ) priority( priority )
                {
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    tile::trsm(
                        side, A.diag(),
                        alpha, A(0, 0), B(i, 0) );
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        for (int64_t j = 0; j < B.nt(); ++j) {
            if (B.tileIsLocal(0, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B ) \
                    firstprivate( j, layout, side, alpha ) priority( priority )
                {
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    tile::trsm(
                        side, A.diag(),
                        alpha, A(0, 0), B(0, j) );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host nested OpenMP implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm(internal::TargetType<Target::HostNest>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, Layout layout, int64_t queue_index )
{
    trsm( internal::TargetType<Target::HostTask>(),
            side, alpha, A, B, priority, layout, queue_index );
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// Host batched implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm(internal::TargetType<Target::HostBatch>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, Layout layout, int64_t queue_index )
{
    trsm( internal::TargetType<Target::HostTask>(),
            side, alpha, A, B, priority, layout, queue_index );
}

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
/// GPU device batched cuBLAS implementation.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm(internal::TargetType<Target::Devices>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          int priority, Layout layout, int64_t queue_index )
{
    using std::swap;
    using blas::conj;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    assert(B.num_devices() > 0);
    assert(A.mt() == 1);
    assert(B.uploPhysical() == Uplo::General);
    assert(A.mt() == A.nt());  // square
    assert(side == Side::Left ? A.mt() == B.mt() : A.mt() == B.nt());

    Uplo uploA = A.uploPhysical();
    Diag diagA = A.diag();
    Op opA = A.op();
    Side sideA = side;

    if (B.op() != Op::NoTrans) {
        if (A.is_complex && A.op() != Op::NoTrans && A.op() != B.op())
            throw std::exception();

        // switch op(A) <=> op(B), side left <=> right, m <=> n
        sideA = (side == Side::Left ? Side::Right : Side::Left);
        if (A.op() == Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || A.is_real) {
            // A and B are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans)
            alpha = conj(alpha);
    }

    #pragma omp taskgroup
    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority) \
            firstprivate(device, side, layout, sideA, uploA, opA, diagA) \
            firstprivate( alpha, queue_index )
        {
            std::set<ij_tuple> B_tiles_set;
            if (side == Side::Right) {
                for (int64_t i = 0; i < B.mt(); ++i) {
                    if (B.tileIsLocal(i, 0)
                        && device == B.tileDevice(i, 0))
                    {
                        B_tiles_set.insert({i, 0});
                    }
                }
            }
            else {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(0, j)
                        && device == B.tileDevice(0, j))
                    {
                        B_tiles_set.insert({0, j});
                    }
                }
            }

            int64_t batch_size = B_tiles_set.size();
            if (batch_size > 0) {

                A.tileGetForReading(0, 0, device, LayoutConvert(layout));
                B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));

                scalar_t** a_array_host = B.array_host(device, queue_index);
                scalar_t** b_array_host = a_array_host + batch_size;

                // B comes first since we do computation for a local B
                auto group_params = device_regions_build<false, 2, scalar_t>(
                        {B, A},
                        {b_array_host, a_array_host},
                        device );

                {
                    trace::Block trace_block("blas::batch::trsm");

                    std::vector<Side>      side_(1, sideA);
                    std::vector<Uplo>      uplo_(1, uploA);
                    std::vector<Op>         opA_(1, opA  );
                    std::vector<Diag>      diag_(1, diagA);
                    std::vector<scalar_t> alpha_(1, alpha);
                    // info size 0 disables slow checks in batched BLAS++.
                    std::vector<int64_t> info;

                    blas::Queue* queue = B.compute_queue(device, queue_index);
                    assert(queue != nullptr);

                    for (size_t g = 0; g < group_params.size(); ++g) {

                        int64_t group_count = group_params[ g ].count;

                        std::vector<int64_t>    m(1, group_params[ g ].mb);
                        std::vector<int64_t>    n(1, group_params[ g ].nb);
                        std::vector<int64_t> ldda(1, group_params[ g ].ld[1]);
                        std::vector<int64_t> lddb(1, group_params[ g ].ld[0]);

                        std::vector<scalar_t*> a_array(a_array_host, a_array_host+group_count);
                        std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);

                        if (B.op() != Op::NoTrans) {
                            swap(m, n);
                        }

                        blas::batch::trsm(
                            layout, side_, uplo_, opA_, diag_,
                            m, n,
                            alpha_, a_array, ldda,
                                    b_array, lddb,
                            group_count, info, *queue);

                        a_array_host += group_count;
                        b_array_host += group_count;
                    }

                    queue->sync();
                }
            }
        }
    }
    // end omp taskgroup
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trsm<Target::HostTask, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t queue_index );

// ----------------------------------------
template
void trsm<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t queue_index );

// ----------------------------------------
template
void trsm< Target::HostTask, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm< Target::HostBatch, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm< Target::Devices, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t queue_index );

// ----------------------------------------
template
void trsm< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm< Target::HostBatch, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t queue_index );

template
void trsm< Target::Devices, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t queue_index );

} // namespace internal
} // namespace slate
