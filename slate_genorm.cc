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

#include "slate.hh"
#include "slate_internal.hh"
#include "slate_mpi.hh"

#include <list>
#include <tuple>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::genorm from internal::specialization::genorm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix norm.
/// Generic implementation for any target.
/// @ingroup genorm
template <Target target, typename scalar_t>
blas::real_type<scalar_t>
genorm(slate::internal::TargetType<target>,
       Norm norm, Matrix<scalar_t>& A)
{
    using real_t = blas::real_type<scalar_t>;

    real_t local_max;
    real_t global_max;

    #pragma omp parallel
    #pragma omp master
    {
        local_max =
            internal::genorm<target>(norm, A.sub(0, A.mt()-1, 0, A.nt()-1));
    }

    int retval;
    #pragma omp critical(slate_mpi)
    {
        trace::Block trace_block("MPI_Allreduce");
        retval =
            MPI_Allreduce(&local_max, &global_max, 1, mpi_type<scalar_t>::value,
                          MPI_MAX, A.mpiComm());
    }
    assert(retval == MPI_SUCCESS);

    return global_max;
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup genorm
template <Target target, typename scalar_t>
blas::real_type<scalar_t>
genorm(Norm norm, Matrix<scalar_t>& A,
       const std::map<Option, Value>& opts)

{
    return internal::specialization::genorm(internal::TargetType<target>(),
                                            norm, A);
}

//------------------------------------------------------------------------------
/// Distributed parallel general matrix norm.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] norm
///
/// @param[in] A
///     The m-by-n matrix A.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup genorm
template <typename scalar_t>
blas::real_type<scalar_t>
genorm(Norm norm, Matrix<scalar_t>& A,
       const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return genorm<Target::HostTask>(norm, A, opts);
            break;
        case Target::HostNest:
            return genorm<Target::HostNest>(norm, A, opts);
            break;
        case Target::Devices:
            return genorm<Target::Devices>(norm, A, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
float genorm<float>(
    Norm norm, Matrix<float>& A,
    const std::map<Option, Value>& opts);

template
double genorm<double>(
    Norm norm, Matrix<double>& A,
    const std::map<Option, Value>& opts);

template
float genorm< std::complex<float> >(
    Norm norm, Matrix< std::complex<float> >& A,
    const std::map<Option, Value>& opts);

template
double genorm< std::complex<double> >(
    Norm norm, Matrix< std::complex<double> >& A,
    const std::map<Option, Value>& opts);

} // namespace slate
