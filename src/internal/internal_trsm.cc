//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

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
          int priority, Layout layout, int64_t batch_arrays_index)
{
    trsm(internal::TargetType<target>(),
         side,
         alpha, A,
                B,
         priority, layout, batch_arrays_index);
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
          int priority, Layout layout, int64_t batch_arrays_index)
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
    if (side == Side::Right) {
        assert(B.nt() == 1);
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, 0)) {
                #pragma omp task shared(A, B) priority(priority)
                {
                    B.tileGetForWriting(i, 0, LayoutConvert(layout));
                    trsm(side, A.diag(),
                         alpha, A(0, 0),
                                B(i, 0));
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                }
            }
        }
    }
    else {
        assert(B.mt() == 1);
        for (int64_t j = 0; j < B.nt(); ++j) {
            if (B.tileIsLocal(0, j)) {
                #pragma omp task shared(A, B) priority(priority)
                {
                    B.tileGetForWriting(0, j, LayoutConvert(layout));
                    trsm(side, A.diag(),
                         alpha, A(0, 0),
                                B(0, j));
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                }
            }
        }
    }

    #pragma omp taskwait
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
          int priority, Layout layout, int64_t batch_arrays_index)
{
    throw std::runtime_error(
        "TRSM currently doesn't support Target::HostNest.");
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
          int priority, Layout layout, int64_t batch_arrays_index)
{
    throw std::runtime_error(
        "TRSM currently doesn't support Target::HostBatch.");
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
          int priority, Layout layout, int64_t batch_arrays_index)
{
    using std::swap;
    using blas::conj;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // GPU assumes column major
    // todo:  relax this assumption, by allowing Tile_blas.hh::trsm() to take
    //        layout param
    // todo:  optimize for the number of layout conversions,
    //        by watching 'layout' and 'B(i, j).layout()'
    assert(layout == Layout::ColMajor);

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

    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority)
        {
            std::set<ij_tuple> B_tiles_set;
            if (side == Side::Right) {
                for (int64_t i = 0; i < B.mt(); ++i) {
                    if (B.tileIsLocal(i, 0)) {
                        if (device == B.tileDevice(i, 0)) {
                            B_tiles_set.insert({i, 0});
                        }
                    }
                }
            }
            else {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(0, j)) {
                        if (device == B.tileDevice(0, j)) {
                            B_tiles_set.insert({0, j});
                        }
                    }
                }
            }

            int64_t batch_size = B_tiles_set.size();
            if (batch_size > 0) {

                A.tileGetForReading(0, 0, device, LayoutConvert(layout));
                B.tileGetForWriting(B_tiles_set, device, LayoutConvert(layout));

                scalar_t** a_array_host =
                    B.array_host(device, batch_arrays_index);
                scalar_t** b_array_host = a_array_host + batch_size;

                int64_t batch_count = 0;

                int64_t batch_count_0 = 0;
                int64_t batch_count_1 = 0;

                int64_t lda0 = 0;
                int64_t ldb0 = 0;
                int64_t lda1 = 0;
                int64_t ldb1 = 0;

                int64_t mb0 = B.tileMb(0);
                int64_t nb0 = B.tileNb(0);
                int64_t mb1 = B.tileMb(B.mt()-1);
                int64_t nb1 = B.tileNb(B.nt()-1);

                if (side == Side::Right) {
                    for (int64_t i = 0; i < B.mt()-1; ++i) {
                        if (B.tileIsLocal(i, 0)) {
                            if (device == B.tileDevice(i, 0)) {
                                a_array_host[batch_count] = A(0, 0, device).data();
                                b_array_host[batch_count] = B(i, 0, device).data();
                                lda0 = A(0, 0, device).stride();
                                ldb0 = B(i, 0, device).stride();
                                ++batch_count_0;
                                ++batch_count;
                            }
                        }
                    }
                    {
                        int64_t i = B.mt()-1;
                        if (B.tileIsLocal(i, 0)) {
                            if (device == B.tileDevice(i, 0)) {
                                a_array_host[batch_count] = A(0, 0, device).data();
                                b_array_host[batch_count] = B(i, 0, device).data();
                                lda1 = A(0, 0, device).stride();
                                ldb1 = B(i, 0, device).stride();
                                ++batch_count_1;
                                ++batch_count;
                            }
                        }
                    }
                }
                else {
                    for (int64_t j = 0; j < B.nt()-1; ++j) {
                        if (B.tileIsLocal(0, j)) {
                            if (device == B.tileDevice(0, j)) {
                                a_array_host[batch_count] = A(0, 0, device).data();
                                b_array_host[batch_count] = B(0, j, device).data();
                                lda0 = A(0, 0, device).stride();
                                ldb0 = B(0, j, device).stride();
                                ++batch_count_0;
                                ++batch_count;
                            }
                        }
                    }
                    {
                        int64_t j = B.nt()-1;
                        if (B.tileIsLocal(0, j)) {
                            if (device == B.tileDevice(0, j)) {
                                a_array_host[batch_count] = A(0, 0, device).data();
                                b_array_host[batch_count] = B(0, j, device).data();
                                lda1 = A(0, 0, device).stride();
                                ldb1 = B(0, j, device).stride();
                                ++batch_count_1;
                                ++batch_count;
                            }
                        }
                    }
                }

                slate_assert(batch_count == batch_size);

                if (B.op() != Op::NoTrans) {
                    swap(mb0, nb0);
                    swap(mb1, nb1);
                }

                scalar_t** a_array_device = B.array_device(
                                                device, batch_arrays_index);
                scalar_t** b_array_device = a_array_device + batch_size;

                slate_cuda_call(cudaSetDevice(device));

                cudaStream_t stream = B.compute_stream(device);
                cublasHandle_t cublas_handle = B.cublas_handle(device);

                slate_cuda_call(
                    cudaMemcpyAsync(B.array_device(device, batch_arrays_index),
                                    B.array_host(device, batch_arrays_index),
                                    sizeof(scalar_t*) * batch_count * 2,
                                    cudaMemcpyHostToDevice,
                                    stream));
                {
                    trace::Block trace_block("cublasTrsmBatched");

                    if (batch_count_0 > 0) {
                        if (layout == Layout::ColMajor) {
                            slate_cublas_call(
                                cublasTrsmBatched(
                                    cublas_handle,
                                    cublas_side_const(sideA),
                                    cublas_uplo_const(uploA),
                                    cublas_op_const(opA),
                                    cublas_diag_const(diagA),
                                    mb0, nb0,
                                    &alpha,
                                    (const scalar_t**) a_array_device, lda0,
                                          (scalar_t**) b_array_device, ldb0,
                                    batch_count_0));
                        }
                        else {
                            // todo: RowMajor layout
                            throw std::runtime_error(
                              "Row major isn't supported in target=Devices.");
                        }

                        a_array_device += batch_count_0;
                        b_array_device += batch_count_0;
                    }

                    if (batch_count_1 > 0) {
                        if (layout == Layout::ColMajor) {
                            slate_cublas_call(
                                cublasTrsmBatched(
                                    cublas_handle,
                                    cublas_side_const(sideA),
                                    cublas_uplo_const(uploA),
                                    cublas_op_const(opA),
                                    cublas_diag_const(diagA),
                                    mb1, nb1,
                                    &alpha,
                                    (const scalar_t**) a_array_device, lda1,
                                          (scalar_t**) b_array_device, ldb1,
                                    batch_count_1));
                        }
                        else {
                            // todo: RowMajor layout
                            throw std::runtime_error(
                              "Row major isn't supported in target=Devices.");
                        }
                    }

                    slate_cuda_call(cudaStreamSynchronize(stream));
                }

                A.tileRelease(0, 0, device);
                for (auto i = 0; i < batch_size; ++i) {
                    A.tileTick(0, 0);
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trsm<Target::HostTask, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float>&& A,
                           Matrix<float>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

// ----------------------------------------
template
void trsm<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double>&& A,
                            Matrix<double>&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

// ----------------------------------------
template
void trsm< Target::HostTask, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm< Target::HostBatch, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm< Target::Devices, std::complex<float> >(
    Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

// ----------------------------------------
template
void trsm< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm< Target::HostBatch, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

template
void trsm< Target::Devices, std::complex<double> >(
    Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
    int priority, Layout layout, int64_t batch_arrays_index);

} // namespace internal
} // namespace slate
