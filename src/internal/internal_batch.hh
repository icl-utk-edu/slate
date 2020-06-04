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

//------------------------------------------------------------------------------
/// @file
/// Provides simple precision-independent wrappers around MKL and cuBLAS batch
/// routines. Eventually to be replaced by blaspp batch routines.
#ifndef SLATE_INTERNAL_BATCH_HH
#define SLATE_INTERNAL_BATCH_HH

#include "slate/Exception.hh"
#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

#include <complex>
#include <set>

namespace slate {
namespace internal {

#ifdef SLATE_WITH_MKL
//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const float* alpha_array,
    const float** A_array,
    const int* lda_array,
    const float** B_array,
    const int* ldb_array,
    const float* beta_array,
    float** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_sgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, A_array, lda_array,
                                   B_array, ldb_array,
                      beta_array,  C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const double* alpha_array,
    const double** A_array,
    const int* lda_array,
    const double** B_array,
    const int* ldb_array,
    const double* beta_array,
    double** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_dgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, A_array, lda_array,
                                   B_array, ldb_array,
                      beta_array,  C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const std::complex<float>* alpha_array,
    const std::complex<float>** A_array,
    const int* lda_array,
    const std::complex<float>** B_array,
    const int* ldb_array,
    const std::complex<float>* beta_array,
    std::complex<float>** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_cgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, (const void**) A_array, lda_array,
                                   (const void**) B_array, ldb_array,
                      beta_array,  (void**)       C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const std::complex<double>* alpha_array,
    const std::complex<double>** A_array,
    const int* lda_array,
    const std::complex<double>** B_array,
    const int* ldb_array,
    const std::complex<double>* beta_array,
    std::complex<double>** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_zgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, (const void**) A_array, lda_array,
                                   (const void**) B_array, ldb_array,
                      beta_array,  (void**)       C_array, ldc_array,
                      group_count, group_size);
}
#endif // SLATE_WITH_MKL

//------------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,  /* host or device pointer */
    const float* Aarray[],
    int lda,
    const float* Barray[],
    int ldb,
    const float* beta,   /* host or device pointer */
    float* Carray[],
    int ldc,
    int batchCount)
{
    return cublasSgemmBatched(handle, transa, transb, m, n, k,
                              alpha, Aarray, lda,
                                     Barray, ldb,
                              beta,  Carray, ldc,
                              batchCount);
}

//------------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const double* alpha,  /* host or device pointer */
    const double* Aarray[],
    int lda,
    const double* Barray[],
    int ldb,
    const double* beta,   /* host or device pointer */
    double* Carray[],
    int ldc,
    int batchCount)
{
    return cublasDgemmBatched(handle, transa, transb, m, n, k,
                              alpha, Aarray, lda,
                                     Barray, ldb,
                              beta,  Carray, ldc,
                              batchCount);
}

//------------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const std::complex<float>* alpha,  /* host or device pointer */
    const std::complex<float>* Aarray[],
    int lda,
    const std::complex<float>* Barray[],
    int ldb,
    const std::complex<float>* beta,   /* host or device pointer */
    std::complex<float>* Carray[],
    int ldc,
    int batchCount)
{
    return cublasCgemmBatched(handle, transa, transb, m, n, k,
                              (cuComplex*)  alpha,
                              (const cuComplex**) Aarray, lda,
                              (const cuComplex**) Barray, ldb,
                              (cuComplex*)  beta,
                              (cuComplex**) Carray, ldc,
                              batchCount);
}

//------------------------------------------------------------------------------
inline cublasStatus_t cublasGemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const std::complex<double>* alpha,  /* host or device pointer */
    const std::complex<double>* Aarray[],
    int lda,
    const std::complex<double>* Barray[],
    int ldb,
    const std::complex<double>* beta,   /* host or device pointer */
    std::complex<double>* Carray[],
    int ldc,
    int batchCount)
{
    return cublasZgemmBatched(handle, transa, transb, m, n, k,
                              (cuDoubleComplex*)  alpha,
                              (const cuDoubleComplex**) Aarray, lda,
                              (const cuDoubleComplex**) Barray, ldb,
                              (cuDoubleComplex*)  beta,
                              (cuDoubleComplex**) Carray, ldc,
                              batchCount);
}

//------------------------------------------------------------------------------
inline cublasStatus_t cublasTrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* Aarray[],
    int lda,
    float* Barray[],
    int ldb,
    int batchCount)
{
    return cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha,
                              Aarray, lda, Barray, ldb, batchCount);
}

//------------------------------------------------------------------------------
inline cublasStatus_t cublasTrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int m,
    int n,
    const double* alpha,
    const double* Aarray[],
    int lda,
    double* Barray[],
    int ldb,
    int batchCount)
{
    return cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha,
                              Aarray, lda, Barray, ldb, batchCount);
}

//------------------------------------------------------------------------------
inline cublasStatus_t cublasTrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int m,
    int n,
    const std::complex<float>* alpha,
    const std::complex<float>* Aarray[],
    int lda,
    std::complex<float>* Barray[],
    int ldb,
    int batchCount)
{
    return cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n,
                              (cuComplex*)  alpha,
                              (const cuComplex**) Aarray, lda,
                              (cuComplex**)       Barray, ldb,
                              batchCount);
}

//------------------------------------------------------------------------------
inline cublasStatus_t cublasTrsmBatched(
    cublasHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int m,
    int n,
    const std::complex<double>* alpha,
    const std::complex<double>* Aarray[],
    int lda,
    std::complex<double>* Barray[],
    int ldb,
    int batchCount)
{
    return cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n,
                              (cuDoubleComplex*)  alpha,
                              (const cuDoubleComplex**) Aarray, lda,
                              (cuDoubleComplex**)       Barray, ldb,
                              batchCount);
}

//==============================================================================
/// batch array workspace/holder for one device
///
/// @tparam dim_
///     number of batch-arrays per operation
///     Example: gemm needs pointer arrays for A, B, and C, thus dim_=3
///
template <typename scalar_t, int dim_>
class DeviceArrays {
public:
    using ij_tuple = std::tuple<int64_t, int64_t>;

    DeviceArrays()
        : array_host_(nullptr),
          array_dev_ (nullptr),
          batch_count_(0),
          num_groups_(0),
          device_(0)
    {}

    ~DeviceArrays()
    {
        freeBatchArrays();
    }

    int numGroups()
    {
        return num_groups_;
    }

    void setNumGroups(int groups)
    {
        // slate_assert(groups > 0);
        if (num_groups_ != groups) {
            num_groups_ = groups;
            for (int d = 0; d < dim_; ++d) {
                nb_[d].resize(num_groups_, 0);
                ld_[d].resize(num_groups_, 0);
            }
            tiles_.resize(num_groups_);
        }
    }

    void allocateBatchArrays(int64_t batch_size, int device)
    {
        device_ = device;
        slate_assert(batch_size >= 0);
        if (batch_count_ < batch_size) {
            slate_cuda_call(
                cudaSetDevice(device));

            // Free host arrays.
            slate_cuda_call(
                cudaFreeHost(array_host_));

            // Free device arrays.
            slate_cuda_call(
                cudaFree(array_dev_));

            // Allocate host arrays.
            slateCudaMallocHost(&array_host_, batch_size * dim_);

            // Allocate device arrays.
            slateCudaMalloc(&array_dev_, batch_size * dim_);

            batch_count_ = batch_size;
        }
    }

    void freeBatchArrays()
    {
        slate_cuda_call(
            cudaSetDevice(device_));

        // Free host arrays.
        if (batch_count_ > 0 && array_host_ != nullptr)
            slate_cuda_call(
                cudaFreeHost(array_host_));

        // Free device arrays.
        if (batch_count_ > 0 && array_dev_ != nullptr)
            slate_cuda_call(
                cudaFree(array_dev_));
    }

    scalar_t** arrayHost(int dim)
    {
        // todo: assert batch_count_ > 0 ?
        return array_host_ + dim * batch_count_;
    }

    scalar_t** arrayDevice(int dim)
    {
        // todo: assert batch_count_ > 0 ?
        return array_dev_ + dim * batch_count_;
    }

    std::set<ij_tuple>& tiles(int group)
    {
        return tiles_[group];
    }

    int64_t& ld(int dim, int group)
    {
        return ld_[dim][group];
    }

    int64_t& nb(int dim, int group)
    {
        return nb_[dim][group];
    }

private:
    scalar_t** array_host_;
    scalar_t** array_dev_;
    int64_t batch_count_;
    std::vector<int64_t> nb_[dim_]; // index by dim/group
    std::vector<int64_t> ld_[dim_]; // index by dim/group
    std::vector< std::set<ij_tuple> > tiles_; // index by group
    int num_groups_; ///< number of quarters/groups
    int device_;
};


//==============================================================================
/// batch array workspace/holder for multiple devices
///
/// @tparam dim_
///     number of batch-arrays per operation
///     Example: gemm needs pointer arrays for A, B, and C, thus dim_=3
///
template <typename scalar_t, int dim_>
class BatchArrays {
public:

    BatchArrays(int num_devices)
        : num_devices_(num_devices)
    {
        dev_arrays_.resize(num_devices_);
    }

    void setNumGroups(int groups)
    {
        for (int d = 0; d < num_devices_; ++d) {
            dev_arrays_[d].setNumGroups(groups);
        }
    }

    DeviceArrays<scalar_t, dim_>& deviceArrays(int device)
    {
        return dev_arrays_[device];
    }

    /// Returns number of devices.
    int numDevices() const { return num_devices_; }

private:
    int num_devices_;
    std::vector< DeviceArrays<scalar_t, dim_> > dev_arrays_; // index by device
};

//==============================================================================
template <typename scalar_t>
class GemmBatchArrays : public BatchArrays<scalar_t, 3> {
public:
    GemmBatchArrays(int num_devices)
        :  BatchArrays<scalar_t, 3>(num_devices)
    {}
};

} // namespace slate
} // namespace internal

#endif // SLATE_INTERNAL_BATCH_HH
