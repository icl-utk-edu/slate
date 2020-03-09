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

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/Tile_aux.hh"
#include "slate/types.hh"

namespace slate {
namespace device {

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA)
    gecopy(m, n,
           (cuFloatComplex**) Aarray, lda,
           (cuFloatComplex**) Barray, ldb,
           batch_count, stream);
#endif
}

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<float>** Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA)
    gecopy(m, n,
           (cuFloatComplex**) Aarray, lda,
           (cuDoubleComplex**) Barray, ldb,
           batch_count, stream);
#endif
}

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    std::complex<double>** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA)
    gecopy(m, n,
           (cuDoubleComplex**) Aarray, lda,
           (cuDoubleComplex**) Barray, ldb,
           batch_count, stream);
#endif
}

template <>
void gecopy(
    int64_t m, int64_t n,
    std::complex<double>** Aarray, int64_t lda,
    std::complex<float>** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA)
    gecopy(m, n,
           (cuDoubleComplex**) Aarray, lda,
           (cuFloatComplex**) Barray, ldb,
           batch_count, stream);
#endif
}

//---------------------------------------------------
#if defined(SLATE_NO_CUDA)
// Specializations to allow compilation without CUDA.
template <>
void gecopy(
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
}

template <>
void gecopy(
    int64_t m, int64_t n,
    double** Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
}

template <>
void gecopy(
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
}
template <>
void gecopy(
    int64_t m, int64_t n,
    float** Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, cudaStream_t stream)
{
}
#endif // not SLATE_NO_CUDA

} // namespace device

namespace internal {

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Dispatches to target implementations.
/// @ingroup copy_internal
///
template <Target target, typename src_scalar_t, typename dst_scalar_t>
void copy(Matrix<src_scalar_t>&& A,
          Matrix<dst_scalar_t>&& B,
          int priority)
{
    copy(internal::TargetType<target>(),
         A, B,
         priority);
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// assumes A & B have same tile layout and dimensions, and have same distribution
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup copy_internal
///
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::HostTask>,
          Matrix<src_scalar_t>& A,
          Matrix<dst_scalar_t>& B,
          int priority)
{
    // trace::Block trace_block("copy");

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt == B.mt());
    assert(A_nt == B.nt());

    for (int64_t i = 0; i < A_mt; ++i) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (B.tileIsLocal(i, j)) {
                #pragma omp task shared(A, B) priority(priority)
                {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    // tileAcquire() to avoid un-needed copy
                    B.tileAcquire(i, j, A.tileLayout(i, j));
                    gecopy(A(i, j), B(i, j));
                    B.tileModified(i, j);
                    A.tileTick(i, j);// TODO is this correct here?
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Assumes A & B have same tile layout, dimensions, and distribution.
/// TODO: Inspect transposition?
/// GPU device implementation.
/// @ingroup copy_internal
///
template <typename src_scalar_t, typename dst_scalar_t>
void copy(internal::TargetType<Target::Devices>,
          Matrix<src_scalar_t>& A,
          Matrix<dst_scalar_t>& B,
          int priority)
{
    using ij_tuple = typename BaseMatrix<src_scalar_t>::ij_tuple;
    int64_t irange[4][2] = {
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   },
        { 0,        B.mt()-1 },
        { B.mt()-1, B.mt()   }
    };
    int64_t jrange[4][2] = {
        { 0,        B.nt()-1 },
        { 0,        B.nt()-1 },
        { B.nt()-1, B.nt()   },
        { B.nt()-1, B.nt()   }
    };

    for (int device = 0; device < B.num_devices(); ++device) {
        #pragma omp task shared(A, B) priority(priority)
        {
            std::set<ij_tuple> A_tiles_set;
            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        A_tiles_set.insert({i, j});
                        // tileAcquire() instead to avoid un-needed copy
                        B.tileAcquire(i, j, device, Layout::ColMajor);
                    }
                }
            }
            // no need to convert layout.
            A.tileGetForReading(A_tiles_set, device, LayoutConvert::None);

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the possibly different types.
            src_scalar_t** a_array_host = A.array_host(device);
            dst_scalar_t** b_array_host = B.array_host(device);

            int64_t batch_count = 0;
            int64_t mb[4], nb[4], lda[4], ldb[4], group_count[4];
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                ldb[q] = 0;
                mb[q] = B.tileMb(irange[q][0]);
                nb[q] = B.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                            a_array_host[batch_count] = A(i, j, device).data();
                            b_array_host[batch_count] = B(i, j, device).data();
                            lda[q] = A(i, j, device).stride();
                            ldb[q] = B(i, j, device).stride();
                            ++group_count[q];
                            ++batch_count;
                        }
                    }
                }
            }

            // Usually the output matrix (B) provides all the batch arrays.
            // Here we are using A, because of the differen types.
            src_scalar_t** a_array_dev = A.array_device(device);
            dst_scalar_t** b_array_dev = B.array_device(device);

            slate_cuda_call(cudaSetDevice(device));

            cudaStream_t stream = B.compute_stream(device);
            // cublasHandle_t cublas_handle = B.cublas_handle(device);
            slate_cuda_call(
                cudaMemcpyAsync(a_array_dev, a_array_host,
                                sizeof(src_scalar_t*)*batch_count,
                                cudaMemcpyHostToDevice,
                                stream));

            slate_cuda_call(
                cudaMemcpyAsync(b_array_dev, b_array_host,
                                sizeof(dst_scalar_t*)*batch_count,
                                cudaMemcpyHostToDevice,
                                stream));

            for (int q = 0; q < 4; ++q) {
                if (group_count[q] > 0) {
                    device::gecopy(mb[q], nb[q],
                                   a_array_dev, lda[q],
                                   b_array_dev, ldb[q],
                                   group_count[q], stream);
                    a_array_dev += group_count[q];
                    b_array_dev += group_count[q];
                }
            }

            slate_cuda_call(cudaStreamSynchronize(stream));

            for (int64_t i = 0; i < B.mt(); ++i) {
                for (int64_t j = 0; j < B.nt(); ++j) {
                    if (B.tileIsLocal(i, j) && device == B.tileDevice(i, j)) {
                        B.tileModified(i, j, device);
                        // update output tile layout
                        // todo: what if extended?
                        B.tileLayout(i, j, device, A.tileLayout(i, j, device));
                        // erase tmp local and remote device tiles;
                        A.tileRelease(i, j, device);
                        // decrement life for remote tiles
                        A.tileTick(i, j);
                    }
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
void copy<Target::HostTask, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority);

template
void copy<Target::HostTask, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority);

template
void copy<Target::Devices, float, float>(
    Matrix<float>&& A, Matrix<float>&& B,
    int priority);

template
void copy<Target::Devices, float, double>(
    Matrix<float>&& A, Matrix<double>&& B,
    int priority);

// ----------------------------------------
template
void copy<Target::HostTask, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority);

template
void copy<Target::HostTask, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority);

template
void copy<Target::Devices, double, double>(
    Matrix<double>&& A, Matrix<double>&& B,
    int priority);

template
void copy<Target::Devices, double, float>(
    Matrix<double>&& A, Matrix<float>&& B,
    int priority);

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<float>, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority);

template
void copy< Target::HostTask, std::complex<float>, std::complex<double> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<float>, std::complex<float>  >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<float>, std::complex<double>  >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<double> >&& B,
    int priority);

// ----------------------------------------
template
void copy< Target::HostTask, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority);

template
void copy< Target::HostTask, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<double>, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& B,
    int priority);

template
void copy< Target::Devices, std::complex<double>, std::complex<float> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<float> >&& B,
    int priority);

} // namespace internal
} // namespace slate
