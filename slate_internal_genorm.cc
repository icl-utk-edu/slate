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

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// General matrix norm.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
blas::real_type<scalar_t>
genorm(Norm norm, Matrix<scalar_t>&& A,
       int priority)
{
    return genorm(internal::TargetType<target>(),
                  norm, A,
                  priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// General matrix norm.
/// Host OpenMP task implementation.
template <typename scalar_t>
blas::real_type<scalar_t>
genorm(internal::TargetType<Target::HostTask>,
       Norm norm, Matrix<scalar_t>& A,
       int priority)
{
    using real_t = blas::real_type<scalar_t>;

    std::vector<real_t> tiles_maxima;

    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task shared(A, tiles_maxima) priority(priority)
                {
                    A.tileCopyToHost(i, j, A.tileDevice(i, j));
                    real_t tile_max = genorm(norm, A(i, j));
                    #pragma omp critical
                    {
                        tiles_maxima.push_back(tile_max);
                    }
                }
            }
        }
    }

    #pragma omp taskwait

    return lapack::lange(norm, tiles_maxima.size(), 1, tiles_maxima.data(), 1);
}

///-----------------------------------------------------------------------------
/// \brief
/// General matrix norm.
/// Host nested OpenMP implementation.
template <typename scalar_t>
blas::real_type<scalar_t>
genorm(internal::TargetType<Target::HostNest>,
       Norm norm, Matrix<scalar_t>& A,
       int priority)
{
    using real_t = blas::real_type<scalar_t>;

    std::vector<real_t> tiles_maxima;

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {

                A.tileCopyToHost(i, j, A.tileDevice(i, j));
                real_t tile_max = genorm(norm, A(i, j));
                #pragma omp critical
                {
                    tiles_maxima.push_back(tile_max);
                }
            }
        }
    }

    #pragma omp taskwait

    return lapack::lange(norm, tiles_maxima.size(), 1, tiles_maxima.data(), 1);
}

///-----------------------------------------------------------------------------
/// \brief
/// General matrix norm.
/// GPU device implementation.
template <typename scalar_t>
blas::real_type<scalar_t>
genorm(internal::TargetType<Target::Devices>,
       Norm norm, Matrix< scalar_t >& A,
       int priority)
{
    using real_t = blas::real_type<scalar_t>;

    assert(A.num_devices() > 0);

    // Allocate norm-specific batch arrays.
    // In the case of norms, the internal routine is only called once.
    // Therefore, memory management has no detrimental effect on performance.

    std::vector<scalar_t**> a_arrays_host(A.num_devices());
    std::vector<real_t*> norm_arrays_host(A.num_devices());

    std::vector<scalar_t**> a_arrays_dev(A.num_devices());
    std::vector<real_t*> norm_arrays_dev(A.num_devices());

    for (int device = 0; device < A.num_devices(); ++device) {

        cudaError_t error;
        error = cudaSetDevice(device);
        assert(error == cudaSuccess);

        int64_t max_tiles = A.getMaxDeviceTiles(device);

        error = cudaMallocHost((void**)&a_arrays_host.at(device),
                               sizeof(scalar_t*)*max_tiles);
        assert(error == cudaSuccess);

        error = cudaMallocHost((void**)&norm_arrays_host.at(device),
                               sizeof(real_t)*max_tiles);
        assert(error == cudaSuccess);

        error = cudaMalloc((void**)&a_arrays_dev.at(device),
                           sizeof(scalar_t*)*max_tiles);
        assert(error == cudaSuccess);

        error = cudaMalloc((void**)&norm_arrays_dev.at(device),
                           sizeof(real_t)*max_tiles);
        assert(error == cudaSuccess);
    }

    std::vector<real_t> devices_maxima(A.num_devices());

    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task shared(A, devices_maxima) priority(priority)
        {
            for (int64_t i = 0; i < A.mt(); ++i)
                for (int64_t j = 0; j < A.nt(); ++j)
                    if (A.tileIsLocal(i, j))
                        if (device == A.tileDevice(i, j))
                            A.tileCopyToDevice(i, j, device);

            scalar_t** a_array_host = a_arrays_host.at(device);
            scalar_t** a_array_dev = a_arrays_dev.at(device);

            int64_t batch_count = 0;
            int64_t batch_count_00 = 0;
            int64_t lda00 = 0;
            int64_t mb00 = A.tileMb(0);
            int64_t nb00 = A.tileNb(0);
            for (int64_t i = 0; i < A.mt()-1; ++i) {
                for (int64_t j = 0; j < A.nt()-1; ++j) {
                    if (A.tileIsLocal(i, j)) {
                        if (device == A.tileDevice(i, j)) {
                            a_array_host[batch_count] = A(i, j, device).data();
                            lda00 = A(i, j, device).stride();
                            ++batch_count_00;
                            ++batch_count;
                        }
                    }
                }
            }

            int64_t batch_count_10 = 0;
            int64_t lda10 = 0;
            int64_t mb10 = A.tileMb(A.mt()-1);
            int64_t nb10 = A.tileNb(0);
            {
                int64_t i = A.mt()-1;
                for (int64_t j = 0; j < A.nt()-1; ++j) {
                    if (A.tileIsLocal(i, j)) {
                        if (device == A.tileDevice(i, j)) {
                            a_array_host[batch_count] = A(i, j, device).data();
                            lda10 = A(i, j, device).stride();
                            ++batch_count_10;
                            ++batch_count;
                        }
                    }
                }
            }

            int64_t batch_count_01 = 0;
            int64_t lda01 = 0;
            int64_t mb01 = A.tileMb(0);
            int64_t nb01 = A.tileNb(A.nt()-1);
            {
                int64_t j = A.nt()-1;
                for (int64_t i = 0; i < A.mt()-1; ++i) {
                    if (A.tileIsLocal(i, j)) {
                        if (device == A.tileDevice(i, j)) {
                            a_array_host[batch_count] = A(i, j, device).data();
                            lda01 = A(i, j, device).stride();
                            ++batch_count_01;
                            ++batch_count;
                        }
                    }
                }
            }

            int64_t batch_count_11 = 0;
            int64_t lda11 = 0;
            int64_t mb11 = A.tileMb(A.mt()-1);
            int64_t nb11 = A.tileNb(A.nt()-1);
            {
                int64_t i = A.mt()-1;
                int64_t j = A.nt()-1;
                if (A.tileIsLocal(i, j)) {
                    if (device == A.tileDevice(i, j)) {
                        a_array_host[batch_count] = A(i, j, device).data();
                        lda11 = A(i, j, device).stride();
                        ++batch_count_11;
                        ++batch_count;
                    }
                }
            }

            real_t* norm_array_host = norm_arrays_host.at(device);
            real_t* norm_array_dev = norm_arrays_dev.at(device);

            {
                trace::Block trace_block("slate::device::genorm");

                cudaError_t error;
                error = cudaSetDevice(device);
                assert(error == cudaSuccess);

                cudaStream_t stream = A.compute_stream(device);
                error = cudaMemcpyAsync(a_array_dev, a_array_host,
                                        sizeof(scalar_t*)*batch_count,
                                        cudaMemcpyHostToDevice,
                                        stream);
                assert(error == cudaSuccess);
/*
                if (batch_count_00 > 0) {
                        device::genormMax(
                            mb00, nb00,
                            a_array_dev, lda00,
                            norm_array_dev,
                            batch_count_00,
                            stream);
                    a_array_dev += batch_count_00;
                    norm_array_dev += batch_count_00;
                }

                if (batch_count_10 > 0) {
                        device::genormMax(
                            mb10, nb10,
                            a_array_dev, lda10,
                            norm_array_dev,
                            batch_count_10,
                            stream);
                    a_array_dev += batch_count_10;
                    norm_array_dev += batch_count_10;
                }

                if (batch_count_01 > 0) {
                        device::genormMax(
                            mb01, nb01,
                            a_array_dev, lda01,
                            norm_array_dev,
                            batch_count_01,
                            stream);
                    a_array_dev += batch_count_01;
                    norm_array_dev += batch_count_01;
                }

                if (batch_count_11 > 0) {
                        device::genormMax(
                            mb11, nb11,
                            a_array_dev, lda11,
                            norm_array_dev,
                            batch_count_11,
                            stream);
                }
*/
                norm_array_dev = norm_arrays_dev.at(device);

                error = cudaMemcpyAsync(norm_array_host, norm_array_dev,
                                        sizeof(scalar_t*)*batch_count,
                                        cudaMemcpyDeviceToHost,
                                        stream);
                assert(error == cudaSuccess);

                error = cudaStreamSynchronize(stream);
                assert(error == cudaSuccess);
            }

            devices_maxima[device] =
                lapack::lange(norm, batch_count, 1, norm_array_host, 1);
        }
    }

    #pragma omp taskwait

    return lapack::lange(
        norm, devices_maxima.size(), 1, devices_maxima.data(), 1);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
float genorm<Target::HostTask, float>(
    Norm norm, Matrix<float>&& A,
    int priority);

template
float genorm<Target::HostNest, float>(
    Norm norm, Matrix<float>&& A,
    int priority);

template
float genorm<Target::Devices, float>(
    Norm norm, Matrix<float>&& A,
    int priority);

// ----------------------------------------
template
double genorm<Target::HostTask, double>(
    Norm norm, Matrix<double>&& A,
    int priority);

template
double genorm<Target::HostNest, double>(
    Norm norm, Matrix<double>&& A,
    int priority);

template
double genorm<Target::Devices, double>(
    Norm norm, Matrix<double>&& A,
    int priority);

// ----------------------------------------
template
float genorm< Target::HostTask, std::complex<float> >(
    Norm norm, Matrix< std::complex<float> >&& A,
    int priority);

template
float genorm< Target::HostNest, std::complex<float> >(
    Norm norm, Matrix< std::complex<float> >&& A,
    int priority);

template
float genorm< Target::Devices, std::complex<float> >(
    Norm norm, Matrix< std::complex<float> >&& A,
    int priority);

// ----------------------------------------
template
double genorm< Target::HostTask, std::complex<double> >(
    Norm norm, Matrix< std::complex<double> >&& A,
    int priority);

template
double genorm< Target::HostNest, std::complex<double> >(
    Norm norm, Matrix< std::complex<double> >&& A,
    int priority);

template
double genorm< Target::Devices, std::complex<double> >(
    Norm norm, Matrix< std::complex<double> >&& A,
    int priority);

} // namespace internal
} // namespace slate
