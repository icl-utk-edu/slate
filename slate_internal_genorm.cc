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

#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_blas.hh"
#include "slate_internal.hh"
#include "slate_internal_batch.hh"

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


    return 345.678;
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
