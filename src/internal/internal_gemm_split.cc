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
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// General matrix multiply.
/// where A is a single block column and B is a single block row.
/// Dispatches to target implementations.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conjTranspose;
/// if $op(C)$ is conjTranspose, then $op(A)$ and $op(B)$ cannot be transpose.
///
/// @param[inout] batchArrays
///     holds the pointer arrays to be prepared for later execution
///     of the batch-gemm kernel in gemmExec()
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) to operate with.
///     Local tiles of matrix C and corresponding tiles of A & B
///        on target devices will be converted to layout.
///
/// @param[in] prefetched
///     Indicates whether the tile's data of the input matrices are already
///     prefetched, to avoid fetching again.
///
/// @ingroup gemm_internal
///
template <Target target, typename scalar_t>
void gemmPrep(scalar_t alpha, Matrix<scalar_t>&& A,
                              Matrix<scalar_t>&& B,
              scalar_t beta,  Matrix<scalar_t>&& C,
              GemmBatchArrays<scalar_t>* batchArrays,
              Layout layout, bool prefetched, int priority)
{
    if (C.is_complex &&
        ((C.op() == Op::Trans &&
         (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans)) ||
         (C.op() == Op::ConjTrans &&
         (A.op() == Op::Trans || B.op() == Op::Trans))))
    {
        throw std::exception();
    }

    gemmPrep(internal::TargetType<target>(),
             alpha, A,
                    B,
             beta,  C,
             batchArrays,
             layout, prefetched, priority);
}

//------------------------------------------------------------------------------
/// General matrix multiply.
/// Prepares the batch gemm by preloading the tiles
/// and preparing and preloading the pointer arrays.
/// GPU device batched cuBLAS implementation.
/// GPU can use either ColMajor or RowMajor.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemmPrep(internal::TargetType<Target::Devices>,
              scalar_t alpha, Matrix< scalar_t >& A,
                              Matrix< scalar_t >& B,
              scalar_t beta,  Matrix< scalar_t >& C,
              GemmBatchArrays<scalar_t>* batchArrays,
              Layout layout, bool prefetched, int priority)
{
    using blas::conj;
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // check dimensions
    slate_assert(A.nt() == 1);
    slate_assert(B.mt() == 1);
    slate_assert(A.mt() == C.mt());
    slate_assert(B.nt() == C.nt());

    slate_assert(C.num_devices() > 0);
    slate_assert(batchArrays->numDevices() == C.num_devices());

    batchArrays->setNumGroups(4);

    if (C.op() == Op::ConjTrans) {
        alpha = conj(alpha);
        beta  = conj(beta);
    }

    int err = 0;
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, B, C, err, batchArrays) \
                         priority(priority)
        {
            auto& deviceArrays = batchArrays->deviceArrays(device);

            int64_t batch_count = 0;
            std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
            for (int64_t i = 0; i < C.mt(); ++i) {
                for (int64_t j = 0; j < C.nt(); ++j) {
                    if (C.tileIsLocal(i, j) &&
                        device == C.tileDevice(i, j)) {
                        A_tiles_set.insert({i, 0});
                        B_tiles_set.insert({0, j});
                        C_tiles_set.insert({i, j});

                        int g = (i >= C.mt()-1) + (j >= C.nt()-1)*2;
                        deviceArrays.tiles(g).insert({i, j});
                        ++batch_count;
                    }
                }
            }
            deviceArrays.allocateBatchArrays(batch_count, device);

            if (! prefetched) {
                #pragma omp task default(shared)
                {
                    trace::Block trace_block(std::string("A.tileGetForReading("+std::to_string(device)+")").c_str());
                    A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp task default(shared)
                {
                    trace::Block trace_block(std::string("B.tileGetForReading("+std::to_string(device)+")").c_str());
                    B.tileGetForReading(B_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp task default(shared)
                {
                    trace::Block trace_block(std::string("C.tileGetForWriting("+std::to_string(device)+")").c_str());
                    C.tileGetForWriting(C_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp taskwait
            }

            scalar_t** a_array_host = deviceArrays.arrayHost(0);
            scalar_t** b_array_host = deviceArrays.arrayHost(1);
            scalar_t** c_array_host = deviceArrays.arrayHost(2);

            if (C.op() != Op::NoTrans) {
                swap(a_array_host, b_array_host);
            }

            trace::Block trace_block(std::string("collectPointers("+std::to_string(device)+")").c_str());

            auto index = 0;
            for (int g = 0; g < deviceArrays.numGroups(); ++g) {
                auto& tiles = deviceArrays.tiles(g);
                if (tiles.size() <= 0) continue;

                auto iter = tiles.begin();
                int64_t i = std::get<0>(*iter);
                int64_t j = std::get<1>(*iter);
                deviceArrays.ld(0, g) = A(i, 0, device).stride();
                deviceArrays.ld(1, g) = B(0, j, device).stride();
                deviceArrays.ld(2, g) = C(i, j, device).stride();
                deviceArrays.nb(0, g) = C.tileMb(i);
                deviceArrays.nb(1, g) = C.tileNb(j);
                deviceArrays.nb(2, g) = A.tileNb(0);
                for (; iter != tiles.end(); ++iter) {
                    int64_t i = std::get<0>(*iter);
                    int64_t j = std::get<1>(*iter);
                    a_array_host[index] = A(i, 0, device).data();
                    b_array_host[index] = B(0, j, device).data();
                    c_array_host[index] = C(i, j, device).data();
                    index++;
                }
            }
            slate_assert(index == batch_count);

            scalar_t** a_array_dev = deviceArrays.arrayDevice(0);

            slate_cuda_call(
                cudaSetDevice(device));

            // cublas_handle uses this stream
            cudaStream_t stream = C.comm_stream(device);

            slate_cuda_call(
                cudaMemcpyAsync(a_array_dev, a_array_host,
                                sizeof(scalar_t*)*(batch_count*3),
                                cudaMemcpyHostToDevice,
                                stream));

            slate_cuda_call(
                cudaStreamSynchronize(stream));
        }
    }

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// General matrix multiply: executes a gemm that is prepared by a corresponding
/// gemmPrep().
/// GPU can use either ColMajor or RowMajor.
/// @ingroup gemm_internal
///
template <Target target, typename scalar_t>
void gemmExec(scalar_t alpha, Matrix<scalar_t>&& A,
                              Matrix<scalar_t>&& B,
              scalar_t beta,  Matrix<scalar_t>&& C,
              GemmBatchArrays<scalar_t>* batchArrays,
              Layout layout, int priority)
{
    if (C.is_complex &&
        ((C.op() == Op::Trans &&
         (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans)) ||
         (C.op() == Op::ConjTrans &&
         (A.op() == Op::Trans || B.op() == Op::Trans))))
    {
        throw std::exception();
    }

    gemmExec(internal::TargetType<target>(),
             alpha, A,
                    B,
             beta,  C,
             batchArrays,
             layout, priority);
}


//------------------------------------------------------------------------------
/// General matrix multiply: executes a gemm that is prepared by a corresponding
/// gemmPrep().
/// GPU device batched cuBLAS implementation.
/// GPU can use either ColMajor or RowMajor.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemmExec(internal::TargetType<Target::Devices>,
              scalar_t alpha, Matrix< scalar_t >& A,
                              Matrix< scalar_t >& B,
              scalar_t beta,  Matrix< scalar_t >& C,
              GemmBatchArrays<scalar_t>* batchArrays,
              Layout layout, int priority)
{
    using blas::conj;
    using std::swap;

    // check dimensions
    slate_assert(A.nt() == 1);
    slate_assert(B.mt() == 1);
    slate_assert(A.mt() == C.mt());
    slate_assert(B.nt() == C.nt());

    slate_assert(C.num_devices() > 0);
    slate_assert(batchArrays->numDevices() == C.num_devices());

    if (C.op() == Op::ConjTrans) {
        alpha = conj(alpha);
        beta  = conj(beta);
    }

    int err = 0;
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, B, C, err) priority(priority)
        {
            // if op(C) is NoTrans, invert opA, opB if possible
            Op opA = A.op();
            if (C.op() != Op::NoTrans) {
                if (opA == Op::NoTrans)
                    opA = C.op();
                else if (A.op() == C.op() || C.is_real) {
                    // A and C are both Trans or both ConjTrans;
                    // Trans == ConjTrans if real
                    opA = Op::NoTrans;
                }
                else {
                    err = __LINE__;  // ConjNoTrans not supported
                }
            }

            Op opB = B.op();
            if (C.op() != Op::NoTrans) {
                if (opB == Op::NoTrans)
                    opB = C.op();
                else if (opB == C.op() || C.is_real) {
                    // B and C are both Trans or both ConjTrans;
                    // Trans == ConjTrans if real
                    opB = Op::NoTrans;
                }
                else {
                    err = __LINE__;  // ConjNoTrans not supported
                }
            }

            if (C.op() != Op::NoTrans) {
                // swap A <=> B; swap m <=> n
                swap(opA, opB);
            }

            slate_cuda_call(
                cudaSetDevice(device));

            // cublas_handle uses this stream
            cudaStream_t stream = C.compute_stream(device);
            cublasHandle_t cublas_handle = C.cublas_handle(device);

            auto& deviceArrays = batchArrays->deviceArrays(device);

            {
                trace::Block trace_block(std::string("cublasGemmBatched("+std::to_string(device)+")").c_str());

                auto num_groups  = deviceArrays.numGroups();
                slate_assert(num_groups == 4);

                auto a_array_dev = deviceArrays.arrayDevice(0);
                auto b_array_dev = deviceArrays.arrayDevice(1);
                auto c_array_dev = deviceArrays.arrayDevice(2);

                for (int g = 0; g < num_groups; ++g) {

                    int64_t batch_count_g = deviceArrays.tiles(g).size();
                    if (batch_count_g > 0) {
                        int64_t lda = deviceArrays.ld(0, g);
                        int64_t ldb = deviceArrays.ld(1, g);
                        int64_t ldc = deviceArrays.ld(2, g);
                        int64_t mb  = deviceArrays.nb(0, g);
                        int64_t nb  = deviceArrays.nb(1, g);
                        int64_t kb  = deviceArrays.nb(2, g);

                        if (C.op() != Op::NoTrans) {
                            // swap A <=> B; swap m <=> n
                            swap(lda, ldb);
                            swap(mb, nb);
                        }

                        if (layout == Layout::ColMajor) {
                            slate_cublas_call(
                                cublasGemmBatched(
                                    cublas_handle,  // uses stream
                                    cublas_op_const(opA), cublas_op_const(opB),
                                    mb, nb, kb,
                                    &alpha, (const scalar_t**) a_array_dev, lda,
                                            (const scalar_t**) b_array_dev, ldb,
                                    &beta,                     c_array_dev, ldc,
                                    batch_count_g));
                        }
                        else {
                            slate_cublas_call(
                                cublasGemmBatched(
                                    cublas_handle,  // uses stream
                                    cublas_op_const(opB), cublas_op_const(opA),
                                    nb, mb, kb,
                                    &alpha, (const scalar_t**) b_array_dev, ldb,
                                            (const scalar_t**) a_array_dev, lda,
                                    &beta,                     c_array_dev, ldc,
                                    batch_count_g));
                        }
                        a_array_dev += batch_count_g;
                        b_array_dev += batch_count_g;
                        c_array_dev += batch_count_g;
                    }
                }

                slate_cuda_call(
                    cudaStreamSynchronize(stream));
            }

            trace::Block trace_block(std::string("tileRelease("+std::to_string(device)+")").c_str());
            for (int g = 0; g < deviceArrays.numGroups(); ++g) {
                auto& tiles = deviceArrays.tiles(g);
                if (tiles.size() <= 0) continue;

                for (auto iter = tiles.begin();
                     iter != tiles.end();
                     ++iter) {
                    int64_t i = std::get<0>(*iter);
                    int64_t j = std::get<1>(*iter);
                    // erase tmp local and remote device tiles;
                    A.tileRelease(i, 0, device);
                    B.tileRelease(0, j, device);
                    // decrement life for remote tiles
                    A.tileTick(i, 0);
                    B.tileTick(0, j);
                }
            }
        }
    }

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void gemmPrep<Target::Devices, float>(
              float alpha, Matrix< float >&& A,
                           Matrix< float >&& B,
              float beta,  Matrix< float >&& C,
              GemmBatchArrays<float>* batchArrays,
              Layout layout, bool prefetched, int priority);

template
void gemmExec<Target::Devices, float>(
              float alpha, Matrix< float >&& A,
                           Matrix< float >&& B,
              float beta,  Matrix< float >&& C,
              GemmBatchArrays<float>* batchArrays,
              Layout layout, int priority);

// ----------------------------------------
template
void gemmPrep<Target::Devices, double>(
              double alpha, Matrix< double >&& A,
                            Matrix< double >&& B,
              double beta,  Matrix< double >&& C,
              GemmBatchArrays<double>* batchArrays,
              Layout layout, bool prefetched, int priority);

template
void gemmExec<Target::Devices, double>(
              double alpha, Matrix< double >&& A,
                            Matrix< double >&& B,
              double beta,  Matrix< double >&& C,
              GemmBatchArrays<double>* batchArrays,
              Layout layout, int priority);

// ----------------------------------------
template
void gemmPrep<Target::Devices, std::complex<float>>(
              std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
              std::complex<float> beta,  Matrix< std::complex<float> >&& C,
              GemmBatchArrays< std::complex<float> >* batchArrays,
              Layout layout, bool prefetched, int priority);

template
void gemmExec<Target::Devices, std::complex<float>>(
              std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                                         Matrix< std::complex<float> >&& B,
              std::complex<float> beta,  Matrix< std::complex<float> >&& C,
              GemmBatchArrays< std::complex<float> >* batchArrays,
              Layout layout, int priority);

// ----------------------------------------
template
void gemmPrep<Target::Devices, std::complex<double>>(
              std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
              std::complex<double> beta,  Matrix< std::complex<double> >&& C,
              GemmBatchArrays< std::complex<double> >* batchArrays,
              Layout layout, bool prefetched, int priority);

template
void gemmExec<Target::Devices, std::complex<double>>(
              std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                          Matrix< std::complex<double> >&& B,
              std::complex<double> beta,  Matrix< std::complex<double> >&& C,
              GemmBatchArrays< std::complex<double> >* batchArrays,
              Layout layout, int priority);

} // namespace internal
} // namespace slate
