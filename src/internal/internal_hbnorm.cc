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

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/Tile_lapack.hh"
#include "internal/Tile_henorm.hh"
#include "internal/Tile_synorm.hh"
#include "slate/types.hh"

#include <vector>

namespace slate {
#if 0
//------------------------------------------------------------------------------
// On macOS, nvcc using clang++ generates a different C++ name mangling
// (std::__1::complex) than g++ for std::complex. This solution is to use
// cu*Complex in .cu files, and cast from std::complex here.
namespace device {

template <>
void henorm(
    Norm in_norm, Uplo uplo,
    int64_t n,
    std::complex<float> const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    henorm(in_norm, uplo, n, (cuFloatComplex**) Aarray, lda,
           values, ldv, batch_count, stream);
#endif
}

template <>
void henorm(
    Norm in_norm, Uplo uplo,
    int64_t n,
    std::complex<double> const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream)
{
#if !defined(SLATE_NO_CUDA) || defined(__NVCC__)
    henorm(in_norm, uplo, n, (cuDoubleComplex**) Aarray, lda,
           values, ldv, batch_count, stream);
#endif
}

#if defined(SLATE_NO_CUDA)
// Specializations to allow compilation without CUDA.
template <>
void henorm(
    Norm in_norm, Uplo uplo,
    int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream)
{
}

template <>
void henorm(
    Norm in_norm, Uplo uplo,
    int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream)
{
}
#endif // not SLATE_NO_CUDA

} // namespace device
#endif
namespace internal {

//------------------------------------------------------------------------------
/// Hermitian banded matrix norm.
/// Dispatches to target implementations.
///
/// @param[in] in_norm
/// - Norm::Max: values is dimension 1 and contains the local max.
/// - Norm::One: values is dimension n and contains the local column sum.
/// - Norm::Inf: for Hermitian, same as Norm::One.
/// - Norm::Fro: values is dimension 2 and contains the local scale and
///              sum-of-squares.
///
template <Target target, typename scalar_t>
void norm(
    Norm in_norm, NormScope scope, HermitianBandMatrix<scalar_t>&& A,
    blas::real_type<scalar_t>* values,
    int priority)
{
    norm(internal::TargetType<target>(),
         in_norm, scope, A, values,
         priority);
}

//------------------------------------------------------------------------------
/// Hermitian matrix norm.
/// Host OpenMP task implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostTask>,
    Norm in_norm, NormScope scope, HermitianBandMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority)
{
    using blas::max;
    using blas::min;
    using real_t = blas::real_type<scalar_t>;

    // norms assumes column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;

    if (scope != NormScope::Matrix) {
        slate_error("Not implemented yet");
    }

    bool lower = (A.uploLogical() == Uplo::Lower);

    int64_t kd = A.bandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t kdt = ceildiv( kd, A.tileNb(0) );

    // i, j are tile row, tile col indices; ii, jj are row, col indices.
    //---------
    // max norm
    // max_{ii,jj} abs( A_{ii,jj} )
    if (in_norm == Norm::Max) {

        // Note: same code in slate::internal::trnorm( Norm::Max ).
        // Find max of each tile, append to tiles_maxima.
        std::vector<real_t> tiles_maxima;
        for (int64_t j = 0; j < A.nt(); ++j) {
            // diagonal tile
            if (j < A.mt() && A.tileIsLocal(j, j)) {
                #pragma omp task shared(A, tiles_maxima) priority(priority)
                {
                    A.tileGetForReading(j, j, LayoutConvert(layout));
                    real_t tile_max;
                    henorm(in_norm, A(j, j), &tile_max);
                    #pragma omp critical
                    {
                        tiles_maxima.push_back(tile_max);
                    }
                }
            }
            // off-diagonal tiles
            if (lower) {
                int64_t i_end = min(j + kdt + 1, A.mt());
                for (int64_t i = j+1; i < i_end; ++i) {  // strictly lower
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task shared(A, tiles_maxima) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_max;
                            genorm(in_norm, NormScope::Matrix, A(i, j), &tile_max);
                            #pragma omp critical
                            {
                                tiles_maxima.push_back(tile_max);
                            }
                        }
                    }
                }
            }
            else { // Uplo::Upper
                int64_t i_begin = max(j - kdt, 0);
                for (int64_t i = i_begin; i < j && i < A.mt(); ++i) {  // strictly upper
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task shared(A, tiles_maxima) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_max;
                            genorm(in_norm, NormScope::Matrix, A(i, j), &tile_max);
                            #pragma omp critical
                            {
                                tiles_maxima.push_back(tile_max);
                            }
                        }
                    }
                }
            }
        }

        #pragma omp taskwait

        // Find max of tiles_maxima.
        *values = lapack::lange(in_norm,
                                1, tiles_maxima.size(),
                                tiles_maxima.data(), 1);
    }
    //---------
    // one norm
    // max col sum = max_jj sum_ii abs( A_{ii,jj} )
    else if (in_norm == Norm::One || in_norm == Norm::Inf) {
        // Sum each column within a tile.
        std::vector<real_t> tiles_sums(A.n()*A.mt(), 0.0);
        int64_t jj = 0;
        for (int64_t j = 0; j < A.nt(); ++j) {
            // diagonal tile
            if (j < A.mt() && A.tileIsLocal(j, j)) {
                #pragma omp task shared(A, tiles_sums) priority(priority)
                {
                    A.tileGetForReading(j, j, LayoutConvert(layout));
                    henorm(in_norm, A(j, j), &tiles_sums[A.n()*j + jj]);
                }
            }
            // off-diagonal tiles (same as synorm)
            if (lower) {
                int64_t ii = jj + A.tileNb(j);
                int64_t i_end = min(j + kdt + 1, A.mt());
                for (int64_t i = j+1; i < i_end; ++i) {  // strictly lower
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task shared(A, tiles_sums) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            synormOffdiag(in_norm, A(i, j),
                                          &tiles_sums[A.n()*i + jj],
                                          &tiles_sums[A.n()*j + ii]);
                        }
                    }
                    ii += A.tileMb(i);
                }
            }
            else { // Uplo::Upper
                int64_t i_begin = max(j - kdt, 0);
                int64_t i_end   = min(j + kdt + 1, A.mt());
                // todo: Assuming a fixed tile size
                int64_t ii = i_begin * A.tileMb(0);
                for (int64_t i = 0; i < j && i < i_end; ++i) {  // strictly upper
                    if (i >= i_begin) {
                        if (A.tileIsLocal(i, j)) {
                            #pragma omp task shared(A, tiles_sums) priority(priority)
                            {
                                A.tileGetForReading(i, j, LayoutConvert(layout));
                                synormOffdiag(in_norm, A(i, j),
                                              &tiles_sums[A.n()*i + jj],
                                              &tiles_sums[A.n()*j + ii]);
                            }
                    }
                    ii += A.tileMb(i);
                    //ii = (A.tileMb(i) % kdt == 0) ? ii + A.tileMb(i) : ii + 1;
                    }
                }
            }
            jj += A.tileNb(j);
        }

        #pragma omp taskwait

        // Sum tile results into local results.
        // Right now it goes over the partial sums of the entire matrix,
        // with all the non-local sums being zero.
        // todo: Eventually this needs to be done like in the device code,
        // by summing up local contributions only.
        std::fill_n(values, A.n(), 0.0);
        for (int64_t i = 0; i < A.mt(); ++i)
            #pragma omp taskloop shared(A, tiles_sums, values) priority(priority)
            for (int64_t jj = 0; jj < A.n(); ++jj)
                values[jj] += tiles_sums[A.n()*i + jj];
    }
    //---------
    // Frobenius norm
    // sqrt( sum_{ii,jj} abs( A_{ii,jj} )^2 )
    // In scaled form: scale^2 sumsq = sum abs( A_{ii,jj}^2 )
    else if (in_norm == Norm::Fro) {
        values[0] = 0;  // scale
        values[1] = 1;  // sumsq
        for (int64_t j = 0; j < A.nt(); ++j) {
            // diagonal tile
            if (j < A.mt() && A.tileIsLocal(j, j)) {
                A.tileGetForReading(j, j, LayoutConvert(layout));
                real_t tile_values[2];
                henorm(in_norm, A(j, j), tile_values);
                #pragma omp critical
                {
                    add_sumsq(values[0], values[1],
                              tile_values[0], tile_values[1]);
                }
            }
            // off-diagonal tiles
            if (lower) {
                int64_t i_end = min(j + kdt + 1, A.mt());
                for (int64_t i = j+1; i < i_end; ++i) {  // strictly lower
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task shared(A, values) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_values[2];
                            genorm(in_norm, NormScope::Matrix, A(i, j), tile_values);
                            // double for symmetric entries
                            tile_values[1] *= 2;
                            #pragma omp critical
                            {
                                add_sumsq(values[0], values[1],
                                          tile_values[0], tile_values[1]);
                            }
                        }
                    }
                }
            }
            else { // Uplo::Upper
                int64_t i_begin = max(j - kdt, 0);
                for (int64_t i = i_begin; i < j && i < A.mt(); ++i) {  // strictly upper
                    if (A.tileIsLocal(i, j)) {
                        #pragma omp task shared(A, values) priority(priority)
                        {
                            A.tileGetForReading(i, j, LayoutConvert(layout));
                            real_t tile_values[2];
                            genorm(in_norm, NormScope::Matrix, A(i, j), tile_values);
                            // double for symmetric entries
                            tile_values[1] *= 2;
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
}

//------------------------------------------------------------------------------
/// Hermitian banded matrix norm.
/// Host nested OpenMP implementation.
/// @ingroup norm_internal
///
template <typename scalar_t>
void norm(
    internal::TargetType<Target::HostNest>,
    Norm in_norm, NormScope scope, HermitianBandMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority)
{
    throw Exception("HostNested not yet implemented");
}

//------------------------------------------------------------------------------
/// Hermitian banded matrix norm.
/// GPU device implementation.
/// @ingroup norm_internal
///
#if 1
template <typename scalar_t>
void norm(
    internal::TargetType<Target::Devices>,
    Norm in_norm, NormScope scope, HermitianBandMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority)
{
    throw Exception("target=Devices not yet implemented");
}
#else
template <typename scalar_t>
void norm(
    internal::TargetType<Target::Devices>,
    Norm in_norm, NormScope scope, HermitianBandMatrix<scalar_t>& A,
    blas::real_type<scalar_t>* values,
    int priority)
{
    using real_t = blas::real_type<scalar_t>;

    // norms assumes column major
    // todo: relax this assumption, a few cases need to be adjusted only
    const Layout layout = Layout::ColMajor;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    if (scope != NormScope::Matrix) {
        slate_error("Not implemented yet");
    }

    bool lower = (A.uploLogical() == Uplo::Lower);

    assert(A.num_devices() > 0);

    std::vector<std::vector<scalar_t*> > a_host_arrays(A.num_devices());
    std::vector<std::vector<real_t> > vals_host_arrays(A.num_devices());

    std::vector<scalar_t**> a_dev_arrays(A.num_devices());
    std::vector<real_t*> vals_dev_arrays(A.num_devices());

    // devices_values used for max and Frobenius norms.
    std::vector<real_t> devices_values;

    int64_t ldv = 0;
    if (in_norm == Norm::Max) {
        ldv = 1;
        devices_values.resize(A.num_devices());
    }
    else if (in_norm == Norm::One || in_norm == Norm::Inf) {
        ldv = 2*A.tileNb(0);
    }
    else if (in_norm == Norm::Fro) {
        ldv = 2;
        devices_values.resize(A.num_devices() * 2);
    }

    for (int device = 0; device < A.num_devices(); ++device) {
        slate_cuda_call(
            cudaSetDevice(device));

        int64_t num_tiles = A.getMaxDeviceTiles(device);

        a_host_arrays[device].resize(num_tiles);
        vals_host_arrays[device].resize(num_tiles*ldv);

        slate_cuda_call(
            cudaMalloc((void**)&a_dev_arrays[device],
                       sizeof(scalar_t*)*num_tiles));

        slate_cuda_call(
            cudaMalloc((void**)&vals_dev_arrays[device],
                       sizeof(real_t)*num_tiles*ldv));
    }

    // Define index ranges for quadrants of matrix.
    // Tiles in each quadrant are all the same size.
    int64_t irange[6][2] = {
        // off-diagonal
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   },
        { 0,        A.mt()-1 },
        { A.mt()-1, A.mt()   },
        // diagonal
        { 0,                          std::min(A.mt(), A.nt())-1 },
        { std::min(A.mt(), A.nt())-1, std::min(A.mt(), A.nt())   }
    };
    int64_t jrange[6][2] = {
        // off-diagonal
        { 0,        A.nt()-1 },
        { 0,        A.nt()-1 },
        { A.nt()-1, A.nt()   },
        { A.nt()-1, A.nt()   },
        // diagonal
        { 0,                          std::min(A.mt(), A.nt())-1 },
        { std::min(A.mt(), A.nt())-1, std::min(A.mt(), A.nt())   }
    };

    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task shared(A, devices_values, vals_host_arrays) \
                         priority(priority)
        {
            std::set<ij_tuple> A_tiles_set;

            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j) &&
                        device == A.tileDevice(i, j) &&
                        ( (  lower && i >= j) ||
                          (! lower && i <= j) ))
                    {
                        A_tiles_set.insert({i, j});
                    }
                }
            }
            A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));

            // Setup batched arguments.
            scalar_t** a_host_array = a_host_arrays[device].data();
            scalar_t** a_dev_array = a_dev_arrays[device];

            int64_t batch_count = 0;
            int64_t mb[6], nb[6], lda[6], group_count[6];
            // off-diagonal blocks
            for (int q = 0; q < 4; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                mb[q] = A.tileMb(irange[q][0]);
                nb[q] = A.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (A.tileIsLocal(i, j) &&
                            device == A.tileDevice(i, j) &&
                            ( (  lower && i > j) ||
                              (! lower && i < j) ))
                        {
                            a_host_array[batch_count] = A(i, j, device).data();
                            lda[q] = A(i, j, device).stride();
                            ++group_count[q];
                            ++batch_count;
                        }
                    }
                }
            }
            // diagonal blocks
            for (int q = 4; q < 6; ++q) {
                group_count[q] = 0;
                lda[q] = 0;
                mb[q] = A.tileMb(jrange[q][0]);
                nb[q] = A.tileNb(jrange[q][0]);
                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    if (A.tileIsLocal(j, j) &&
                        device == A.tileDevice(j, j))
                    {
                        a_host_array[batch_count] = A(j, j, device).data();
                        lda[q] = A(j, j, device).stride();
                        ++group_count[q];
                        ++batch_count;
                    }
                }
            }

            real_t* vals_host_array = vals_host_arrays[device].data();
            real_t* vals_dev_array = vals_dev_arrays[device];

            // Batched call to compute partial results for each tile.
            {
                trace::Block trace_block("slate::device::henorm");

                slate_cuda_call(
                    cudaSetDevice(device));

                cudaStream_t stream = A.compute_stream(device);
                slate_cuda_call(
                    cudaMemcpyAsync(a_dev_array, a_host_array,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream));

                // off-diagonal blocks (same as synorm)
                for (int q = 0; q < 4; ++q) {
                    if (group_count[q] > 0) {
                        if (in_norm == Norm::One || in_norm == Norm::Inf) {
                            device::synormOffdiag(in_norm,
                                                  mb[q], nb[q],
                                                  a_dev_array, lda[q],
                                                  vals_dev_array, ldv,
                                                  group_count[q], stream);
                        }
                        else {
                            device::genorm(in_norm, NormScope::Matrix,
                                           mb[q], nb[q],
                                           a_dev_array, lda[q],
                                           vals_dev_array, ldv,
                                           group_count[q], stream);
                        }
                        a_dev_array += group_count[q];
                        vals_dev_array += group_count[q] * ldv;
                    }
                }
                // diagonal blocks
                for (int q = 4; q < 6; ++q) {
                    if (group_count[q] > 0) {
                        device::henorm(in_norm, A.uploPhysical(),
                                       nb[q],
                                       a_dev_array, lda[q],
                                       vals_dev_array, ldv,
                                       group_count[q], stream);
                        a_dev_array += group_count[q];
                        vals_dev_array += group_count[q] * ldv;
                    }
                }

                vals_dev_array = vals_dev_arrays[device];

                slate_cuda_call(
                    cudaMemcpyAsync(vals_host_array, vals_dev_array,
                                    sizeof(real_t)*batch_count*ldv,
                                    cudaMemcpyDeviceToHost,
                                    stream));

                slate_cuda_call(
                    cudaStreamSynchronize(stream));
            }

            // Reduction over tiles to device result.
            if (in_norm == Norm::Max) {
                devices_values[device] =
                    lapack::lange(in_norm, 1, batch_count, vals_host_array, 1);
            }
            else if (in_norm == Norm::Fro) {
                int64_t batch_count = 0;
                for (int q = 0; q < 6; ++q) {
                    // double for symmetric entries in off-diagonal blocks
                    real_t mult = (q < 4 ? 2.0 : 1.0);
                    for (int64_t k = 0; k < group_count[q]; ++k) {
                        add_sumsq(devices_values[2*device + 0],
                                  devices_values[2*device + 1],
                                  vals_host_array[2*batch_count + 0],
                                  vals_host_array[2*batch_count + 1] * mult);
                        ++batch_count;
                    }
                }
            }
        }
    }

    #pragma omp taskwait

    for (int device = 0; device < A.num_devices(); ++device) {
        slate_cuda_call(
            cudaSetDevice(device));
        slate_cuda_call(
            cudaFree((void*)a_dev_arrays[device]));
        slate_cuda_call(
            cudaFree((void*)vals_dev_arrays[device]));
    }

    // Reduction over devices to local result.
    if (in_norm == Norm::Max) {
        *values = lapack::lange(in_norm,
                                1, devices_values.size(),
                                devices_values.data(), 1);
    }
    else if (in_norm == Norm::One || in_norm == Norm::Inf) {
        for (int device = 0; device < A.num_devices(); ++device) {
            real_t* vals_host_array = vals_host_arrays[device].data();

            int64_t batch_count = 0;
            // off-diagonal blocks
            int64_t nb0 = A.tileNb(0);
            for (int q = 0; q < 4; ++q) {
                int64_t mb = A.tileMb(irange[q][0]);
                int64_t nb = A.tileNb(jrange[q][0]);
                for (int64_t i = irange[q][0]; i < irange[q][1]; ++i) {
                    for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                        if (A.tileIsLocal(i, j) &&
                            device == A.tileDevice(i, j) &&
                            ( (  lower && i > j) ||
                              (! lower && i < j) ))
                        {
                            // col sums
                            blas::axpy(
                                nb, 1.0,
                                &vals_host_array[batch_count*ldv], 1,
                                &values[j*nb0], 1);
                            // row sums
                            blas::axpy(
                                mb, 1.0,
                                &vals_host_array[batch_count*ldv + nb], 1,
                                &values[i*nb0], 1);
                            ++batch_count;
                        }
                    }
                }
            }
            // diagonal blocks
            for (int q = 4; q < 6; ++q) {
                int64_t nb = A.tileNb(jrange[q][0]);
                for (int64_t j = jrange[q][0]; j < jrange[q][1]; ++j) {
                    if (A.tileIsLocal(j, j) &&
                        device == A.tileDevice(j, j))
                    {
                        blas::axpy(
                            nb, 1.0,
                            &vals_host_array[batch_count*ldv], 1,
                            &values[j*nb0], 1);
                        ++batch_count;
                    }
                }
            }
        }
    }
    else if (in_norm == Norm::Fro) {
        values[0] = 0;
        values[1] = 1;
        for (int device = 0; device < A.num_devices(); ++device) {
            add_sumsq(values[0], values[1],
                      devices_values[2*device + 0],
                      devices_values[2*device + 1]);
        }
    }
}
#endif
//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void norm<Target::HostTask, float>(
    Norm in_norm, NormScope scope, HermitianBandMatrix<float>&& A,
    float* values,
    int priority);

template
void norm<Target::HostNest, float>(
    Norm in_norm, NormScope scope, HermitianBandMatrix<float>&& A,
    float* values,
    int priority);

template
void norm<Target::Devices, float>(
    Norm in_norm, NormScope scope, HermitianBandMatrix<float>&& A,
    float* values,
    int priority);

// ----------------------------------------
template
void norm<Target::HostTask, double>(
    Norm in_norm, NormScope scope, HermitianBandMatrix<double>&& A,
    double* values,
    int priority);

template
void norm<Target::HostNest, double>(
    Norm in_norm, NormScope scope, HermitianBandMatrix<double>&& A,
    double* values,
    int priority);

template
void norm<Target::Devices, double>(
    Norm in_norm, NormScope scope, HermitianBandMatrix<double>&& A,
    double* values,
    int priority);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<float> >(
    Norm in_norm, NormScope scope, HermitianBandMatrix< std::complex<float> >&& A,
    float* values,
    int priority);

template
void norm< Target::HostNest, std::complex<float> >(
    Norm in_norm, NormScope scope, HermitianBandMatrix< std::complex<float> >&& A,
    float* values,
    int priority);

template
void norm< Target::Devices, std::complex<float> >(
    Norm in_norm, NormScope scope, HermitianBandMatrix< std::complex<float> >&& A,
    float* values,
    int priority);

// ----------------------------------------
template
void norm< Target::HostTask, std::complex<double> >(
    Norm in_norm, NormScope scope, HermitianBandMatrix< std::complex<double> >&& A,
    double* values,
    int priority);

template
void norm< Target::HostNest, std::complex<double> >(
    Norm in_norm, NormScope scope, HermitianBandMatrix< std::complex<double> >&& A,
    double* values,
    int priority);

template
void norm< Target::Devices, std::complex<double> >(
    Norm in_norm, NormScope scope, HermitianBandMatrix< std::complex<double> >&& A,
    double* values,
    int priority);

} // namespace internal
} // namespace slate
