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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#include "slate_device.hh"
#include "slate_internal_batch.hh"
#include "slate_internal.hh"
#include "slate_Matrix.hh"
#include "slate_Tile_blas.hh"
#include "slate_types.hh"

#include <vector>

namespace slate {

///-----------------------------------------------------------------------------
// On macOS, nvcc using clang++ generates a different C++ name mangling
// (std::__1::complex) than g++ for std::complex. This solution is to use
// cu*Complex in .cu files, and cast from std::complex here.
namespace device {

template <>
void genorm(
    Norm norm,
    int64_t m, int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values,
    int64_t batch_count,
    cudaStream_t stream)
{
#if defined(SLATE_WITH_CUDA) || defined(__NVCC__)
    genorm(norm, m, n, (cuFloatComplex**) Aarray, lda, values, batch_count,
           stream);
#endif
}

template <>
void genorm(
    Norm norm,
    int64_t m, int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values,
    int64_t batch_count,
    cudaStream_t stream)
{
#if defined(SLATE_WITH_CUDA) || defined(__NVCC__)
    genorm(norm, m, n, (cuDoubleComplex**) Aarray, lda, values, batch_count,
           stream);
#endif
}

// Explicit instatiations allow compilation without CUDA
template <>
void genorm(
    Norm norm,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values,
    int64_t batch_count,
    cudaStream_t stream)
{
#if defined(SLATE_WITH_CUDA) || defined(__NVCC__)
    genorm(norm, m, n, Aarray, lda, values, batch_count,
           stream);
#endif
}

template <>
void genorm(
    Norm norm,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values,
    int64_t batch_count,
    cudaStream_t stream)
{
#if defined(SLATE_WITH_CUDA) || defined(__NVCC__)
    genorm(norm, m, n, Aarray, lda, values, batch_count,
           stream);
#endif
}


} // namespace device

namespace internal {

///-----------------------------------------------------------------------------
/// Square of number.
/// @return x^2
template <typename scalar_t>
scalar_t sqr(scalar_t x)
{
    return x*x;
}

//------------------------------------------------------------------------------
/// Adds two scaled, sum-of-squares representations.
/// On exit, scale1 and sumsq1 are updated such that:
///     scale1^2 sumsq1 := scale1^2 sumsq1 + scale2^2 sumsq2.
template <typename real_t>
void add_sumsq(
    real_t&       scale1, real_t&       sumsq1,
    real_t const& scale2, real_t const& sumsq2 )
{
    if (scale1 > scale2) {
        sumsq1 = sumsq1 + sumsq2*sqr(scale2 / scale1);
        // scale1 stays same
    }
    else {
        sumsq1 = sumsq1*sqr(scale1 / scale2) + sumsq2;
        scale1 = scale2;
    }
}

///-----------------------------------------------------------------------------
/// General matrix norm.
/// Dispatches to target implementations.
///
/// @param norm
/// - Norm::Max: values is dimension 1 and contains the local max.
/// - Norm::One: values is dimension n and contains the local column sum.
/// - Norm::Inf: values is dimension m and contains the local row sum.
/// - Norm::Fro: values is dimension 1 and contains the local sum-of-squares.
///
template <Target target, typename scalar_t>
void genorm(
    Norm norm, Matrix<scalar_t>&& A, blas::real_type<scalar_t>* values,
    int priority)
{
    genorm(internal::TargetType<target>(),
           norm, A, values,
           priority);
}

///-----------------------------------------------------------------------------
/// General matrix norm.
/// Host OpenMP task implementation.
template <typename scalar_t>
void genorm(
    internal::TargetType<Target::HostTask>,
    Norm norm, Matrix<scalar_t>& A, blas::real_type<scalar_t>* values,
    int priority)
{
    using real_t = blas::real_type<scalar_t>;

    // i, j are tile row, tile col indices; ii, jj are row, col indices.
    //---------
    // max norm
    // max_{ii,jj} abs( A_{ii,jj} )
    if (norm == Norm::Max) {

        // Find max of each tile, append to tiles_maxima.
        std::vector<real_t> tiles_maxima;
        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task shared(A, tiles_maxima) priority(priority)
                    {
                        A.tileCopyToHost(i, j, A.tileDevice(i, j));
                        real_t tile_max;
                        genorm(norm, A(i, j), &tile_max);
                        #pragma omp critical
                        {
                            tiles_maxima.push_back(tile_max);
                        }
                    }
                }
            }
        }

        #pragma omp taskwait

        // Find max of tiles_maxima.
        *values = lapack::lange(norm,
                                1, tiles_maxima.size(),
                                tiles_maxima.data(), 1);
    }
    //---------
    // one norm
    // max col sum = max_jj sum_ii abs( A_{ii,jj} )
    else if (norm == Norm::One) {

        // Sum each column within a tile.
        std::vector<real_t> tiles_sums(A.n()*A.mt(), 0.0);
        for (int64_t i = 0; i < A.mt(); ++i) {
            int64_t jj = 0;
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task shared(A, tiles_sums) priority(priority)
                    {
                        A.tileCopyToHost(i, j, A.tileDevice(i, j));
                        genorm(norm, A(i, j), &tiles_sums[A.n()*i+jj]);
                    }
                }
                jj += A.tileNb(j);
            }
        }

        #pragma omp taskwait

        // Sum tile results into local results.
        // todo: This is currently a performance bottleneck.
        // Perhaps omp taskloop could be applied here.
        // Perhaps with chunking of A.nb().
        std::fill_n(values, A.n(), 0.0);
        for (int64_t i = 0; i < A.mt(); ++i)
            for (int64_t jj = 0; jj < A.n(); ++jj)
                values[jj] += tiles_sums[A.n()*i + jj];
    }
    //---------
    // inf norm
    // max row sum = max_ii sum_jj abs( A_{ii,jj} )
    else if (norm == Norm::Inf) {

        // Sum each row within a tile.
        std::vector<real_t> tiles_sums(A.m()*A.nt(), 0.0);
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task shared(A, tiles_sums) priority(priority)
                    {
                        A.tileCopyToHost(i, j, A.tileDevice(i, j));
                        genorm(norm, A(i, j), &tiles_sums[A.m()*j + ii]);
                    }
                }
            }
            ii += A.tileMb(i);
        }

        #pragma omp taskwait

        // Sum tile results into local results.
        // todo: This is currently a performance bottleneck.
        // Perhaps omp taskloop could be applied here.
        // Perhaps with chunking of A.nb().
        std::fill_n(values, A.m(), 0.0);
        for (int64_t j = 0; j < A.nt(); ++j)
            for (int64_t ii = 0; ii < A.m(); ++ii)
                values[ii] += tiles_sums[A.m()*j + ii];
    }
    //---------
    // Frobenius norm
    // sqrt( sum_{ii,jj} abs( A_{ii,jj} )^2 )
    // In scaled form: scale^2 sumsq = sum abs( A_{ii,jj}^2 )
    else if (norm == Norm::Fro) {

        values[0] = 0;  // scale
        values[1] = 1;  // sumsq
        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (A.tileIsLocal(i, j)) {
                    #pragma omp task shared(A, values) priority(priority)
                    {
                        A.tileCopyToHost(i, j, A.tileDevice(i, j));
                        real_t tile_values[2];
                        genorm(norm, A(i, j), tile_values);
                        #pragma omp critical
                        {
                            add_sumsq(values[0], values[1],
                                      tile_values[0], tile_values[1]);
                        }
                    }
                }
            }
        }
    }
}

///-----------------------------------------------------------------------------
/// General matrix norm.
/// Host nested OpenMP implementation.
template <typename scalar_t>
void genorm(
    internal::TargetType<Target::HostNest>,
    Norm norm, Matrix<scalar_t>& A, blas::real_type<scalar_t>* values,
    int priority)
{
    using real_t = blas::real_type<scalar_t>;

    std::vector<real_t> tiles_maxima;

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                A.tileCopyToHost(i, j, A.tileDevice(i, j));
                real_t tile_max;
                genorm(norm, A(i, j), &tile_max);
                #pragma omp critical
                {
                    tiles_maxima.push_back(tile_max);
                }
            }
        }
    }

    #pragma omp taskwait

    *values = lapack::lange(norm,
                            1, tiles_maxima.size(),
                            tiles_maxima.data(), 1);
}

///-----------------------------------------------------------------------------
/// General matrix norm.
/// GPU device implementation.
template <typename scalar_t>
void genorm(
    internal::TargetType<Target::Devices>,
    Norm norm, Matrix<scalar_t>& A, blas::real_type<scalar_t>* values,
    int priority)
{
    using real_t = blas::real_type<scalar_t>;

    assert(A.num_devices() > 0);

    std::vector<std::vector<scalar_t*> > a_arrays_host(A.num_devices());
    std::vector<std::vector<real_t> > vals_arrays_host(A.num_devices());

    std::vector<scalar_t**> a_arrays_dev(A.num_devices());
    std::vector<real_t*> vals_arrays_dev(A.num_devices());

    // devices_values used for max and Frobenius norms.
    std::vector<real_t> devices_values(A.num_devices());

    int64_t vals_chunk;
    if (norm == Norm::Max) {
        vals_chunk = 1;
        devices_values.resize(A.num_devices());
    }
    else if (norm == Norm::One) {
        vals_chunk = A.tileNb(0);
    }
    else if (norm == Norm::Inf) {
        vals_chunk = A.tileMb(0);
    }
    else if (norm == Norm::Fro) {
        vals_chunk = 2;
        devices_values.resize(A.num_devices() * 2);
    }

    for (int device = 0; device < A.num_devices(); ++device) {

        slate_cuda_call(
            cudaSetDevice(device));

        int64_t num_tiles = A.getMaxDeviceTiles(device);

        a_arrays_host[device].resize(num_tiles);
        vals_arrays_host[device].resize(num_tiles*vals_chunk);

        slate_cuda_call(
            cudaMalloc((void**)&a_arrays_dev[device],
                       sizeof(scalar_t*)*num_tiles));

        slate_cuda_call(
            cudaMalloc((void**)&vals_arrays_dev[device],
                       sizeof(real_t)*num_tiles*vals_chunk));
    }

    // Define index ranges for quadrants of matrix.
    // Tiles in each quadrant are all the same size.
    int64_t irange[4][2] = {
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   },
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   }
    };
    int64_t jrange[4][2] = {
        { 0,        A.nt()-1 },
        { 0,        A.nt()-1 },
        { A.nt()-1, A.nt()   },
        { A.nt()-1, A.nt()   }
    };

    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task shared(A, devices_values, vals_arrays_host) \
                         priority(priority)
        {
            for (int64_t i = 0; i < A.mt(); ++i)
                for (int64_t j = 0; j < A.nt(); ++j)
                    if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j))
                        A.tileCopyToDevice(i, j, device);

            // Setup batched arguments.
            scalar_t** a_array_host = a_arrays_host[device].data();
            scalar_t** a_array_dev = a_arrays_dev[device];

            int64_t batch_count = 0;
            int64_t mb[4], nb[4], lda[4], group_count[4];
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                mb[q] = A.tileMb(irange[q][0]);
                nb[q] = A.tileNb(jrange[q][0]);
                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                        if (A.tileIsLocal(i, j) &&
                            device == A.tileDevice(i, j))
                        {
                            a_array_host[batch_count] = A(i, j, device).data();
                            lda[q] = A(i, j, device).stride();
                            ++group_count[q];
                            ++batch_count;
                        }
                    }
                }
            }

            real_t* vals_array_host = vals_arrays_host[device].data();
            real_t* vals_array_dev = vals_arrays_dev[device];

            // Batched call to compute partial results for each tile.
            {
                trace::Block trace_block("slate::device::genorm");

                slate_cuda_call(
                    cudaSetDevice(device));

                cudaStream_t stream = A.compute_stream(device);
                slate_cuda_call(
                    cudaMemcpyAsync(a_array_dev, a_array_host,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream));

                for (int q = 0; q < 4; ++q) {
                    if (group_count[q] > 0) {
                        device::genorm(norm,
                                       mb[q], nb[q],
                                       a_array_dev, lda[q],
                                       vals_array_dev, group_count[q], stream);
                        a_array_dev += group_count[q];
                        vals_array_dev += group_count[q] * vals_chunk;
                    }
                }

                vals_array_dev = vals_arrays_dev[device];

                slate_cuda_call(
                    cudaMemcpyAsync(vals_array_host, vals_array_dev,
                                    sizeof(real_t)*batch_count*vals_chunk,
                                    cudaMemcpyDeviceToHost,
                                    stream));

                slate_cuda_call(
                    cudaStreamSynchronize(stream));
            }

            // Reduction over tiles to device result.
            if (norm == Norm::Max) {
                devices_values[device] =
                    lapack::lange(norm, 1, batch_count, vals_array_host, 1);
            }
            else if (norm == Norm::Fro) {
                for (int64_t k = 0; k < batch_count; ++k) {
                    add_sumsq(devices_values[2*device + 0],
                              devices_values[2*device + 1],
                              vals_array_host[2*k + 0],
                              vals_array_host[2*k + 1]);
                }
            }
        }
    }

    #pragma omp taskwait

    for (int device = 0; device < A.num_devices(); ++device) {
        slate_cuda_call(
            cudaSetDevice(device));
        slate_cuda_call(
            cudaFree((void*)a_arrays_dev[device]));
        slate_cuda_call(
            cudaFree((void*)vals_arrays_dev[device]));
    }

    // Reduction over devices to local result.
    if (norm == Norm::Max) {
        *values = lapack::lange(norm,
                                1, devices_values.size(),
                                devices_values.data(), 1);
    }
    else if (norm == Norm::One) {

        for (int device = 0; device < A.num_devices(); ++device) {

            real_t* vals_array_host = vals_arrays_host[device].data();

            int64_t batch_count = 0;
            for (int q = 0; q < 4; ++q) {
                int64_t nb = A.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (A.tileIsLocal(i, j) &&
                            device == A.tileDevice(i, j))
                        {
                            blas::axpy(
                                nb, 1.0,
                                &vals_array_host[batch_count*vals_chunk], 1,
                                &values[j*vals_chunk], 1);
                            ++batch_count;
                        }
                    }
                }
            }
        }
    }
    else if (norm == Norm::Inf) {

        for (int device = 0; device < A.num_devices(); ++device) {

            real_t* vals_array_host = vals_arrays_host[device].data();

            int64_t batch_count = 0;
            for (int q = 0; q < 4; ++q) {
                int64_t mb = A.tileMb(irange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (A.tileIsLocal(i, j) &&
                            device == A.tileDevice(i, j))
                        {
                            blas::axpy(
                                mb, 1.0,
                                &vals_array_host[batch_count*vals_chunk], 1,
                                &values[i*vals_chunk], 1);
                            ++batch_count;
                        }
                    }
                }
            }
        }
    }
    else if (norm == Norm::Fro) {
        values[0] = 0;
        values[1] = 1;
        for (int device = 0; device < A.num_devices(); ++device) {
            add_sumsq(values[0], values[1],
                      devices_values[2*device + 0],
                      devices_values[2*device + 1]);
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void genorm<Target::HostTask, float>(
    Norm norm, Matrix<float>&& A,
    float* values,
    int priority);

template
void genorm<Target::HostNest, float>(
    Norm norm, Matrix<float>&& A,
    float* values,
    int priority);

template
void genorm<Target::Devices, float>(
    Norm norm, Matrix<float>&& A,
    float* values,
    int priority);

// ----------------------------------------
template
void genorm<Target::HostTask, double>(
    Norm norm, Matrix<double>&& A,
    double* values,
    int priority);

template
void genorm<Target::HostNest, double>(
    Norm norm, Matrix<double>&& A,
    double* values,
    int priority);

template
void genorm<Target::Devices, double>(
    Norm norm, Matrix<double>&& A,
    double* values,
    int priority);

// ----------------------------------------
template
void genorm< Target::HostTask, std::complex<float> >(
    Norm norm, Matrix< std::complex<float> >&& A,
    float* values,
    int priority);

template
void genorm< Target::HostNest, std::complex<float> >(
    Norm norm, Matrix< std::complex<float> >&& A,
    float* values,
    int priority);

template
void genorm< Target::Devices, std::complex<float> >(
    Norm norm, Matrix< std::complex<float> >&& A,
    float* values,
    int priority);

// ----------------------------------------
template
void genorm< Target::HostTask, std::complex<double> >(
    Norm norm, Matrix< std::complex<double> >&& A,
    double* values,
    int priority);

template
void genorm< Target::HostNest, std::complex<double> >(
    Norm norm, Matrix< std::complex<double> >&& A,
    double* values,
    int priority);

template
void genorm< Target::Devices, std::complex<double> >(
    Norm norm, Matrix< std::complex<double> >&& A,
    double* values,
    int priority);

} // namespace internal
} // namespace slate
