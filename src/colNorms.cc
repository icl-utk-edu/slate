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

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"
#include "slate/internal/mpi.hh"

#include <list>
#include <tuple>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::norm from internal::specialization::norm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel matrix norm.
/// Generic implementation for any target.
/// @ingroup norm_specialization
///
template <Target target, typename matrix_type>
void colNorms(slate::internal::TargetType<target>,
              Norm in_norm,
              matrix_type A,
              blas::real_type<typename matrix_type::value_type>* values)
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;

    // Undo any transpose.
    if (A.op() == Op::ConjTrans || A.op() == Op::Trans) {
        // todo:
        // if (in_norm == Norm::One)
        //     in_norm = Norm::Inf;
        // else if (in_norm == Norm::Inf)
        //     in_norm = Norm::One;
    }
    if (A.op() == Op::ConjTrans)
        A = conjTranspose(A);
    else if (A.op() == Op::Trans)
        A = transpose(A);

    //---------
    // all max norm (max of each column)
    // max_{i,j} abs( A_{i,j} )
    if (in_norm == Norm::Max) {

        std::vector<real_t> local_maxes(A.n());

        if (target == Target::Devices)
            A.reserveDeviceWorkspace();

        #pragma omp parallel
        #pragma omp master
        {
            internal::norm<target>(in_norm, NormScope::Columns, std::move(A), local_maxes.data());
        }

        MPI_Op op_max_nan;
        #pragma omp critical(slate_mpi)
        {
            slate_mpi_call(
                MPI_Op_create(mpi_max_nan, true, &op_max_nan));
        }

        #pragma omp critical(slate_mpi)
        {
            trace::Block trace_block("MPI_Allreduce");
            slate_mpi_call(
                MPI_Allreduce(local_maxes.data(), values,
                              A.n(), mpi_type<real_t>::value,
                              op_max_nan, A.mpiComm()));
        }

        #pragma omp critical(slate_mpi)
        {
            slate_mpi_call(
                MPI_Op_free(&op_max_nan));
        }
    }
    //---------
    // one norm
    // max col sum = max_j sum_i abs( A_{i,j} )
    else if (in_norm == Norm::One) {
        slate_error("Not implemented yet");
    }
    //---------
    // inf norm
    // max row sum = max_i sum_j abs( A_{i,j} )
    else if (in_norm == Norm::Inf) {
        slate_error("Not implemented yet");
    }
    //---------
    // Frobenius norm
    // sqrt( sum_{i,j} abs( A_{i,j} )^2 )
    else if (in_norm == Norm::Fro) {
        slate_error("Not implemented yet");
    }
    else {
        slate_error("invalid norm");
    }

    // todo: is this correct here?
    A.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup norm_specialization
///
template <Target target, typename matrix_type>
void colNorms(Norm in_norm,
              matrix_type& A,
              blas::real_type<typename matrix_type::value_type>* values,
              Options const& opts)
{
    return internal::specialization::colNorms(internal::TargetType<target>(),
                                              in_norm, A,
                                              values);
}

//------------------------------------------------------------------------------
/// Distributed parallel matrix norm.
///
//------------------------------------------------------------------------------
/// @tparam matrix_type
///     Any SLATE matrix type: Matrix, SymmetricMatrix, HermitianMatrix,
///     TriangularMatrix, etc.
//------------------------------------------------------------------------------
/// @param[in] in_norm
///     Norm to compute:
///     - Norm::Max: maximum element,    $\max_{i, j}   \abs{ A_{i, j} }$
///     - Norm::One: maximum column sum, $\max_j \sum_i \abs{ A_{i, j} }$
///     - Norm::Inf: maximum row sum,    $\max_i \sum_j \abs{ A_{i, j} }$
///       For symmetric and Hermitian matrices, the One and Inf norms are the same.
///     - Norm::Fro: Frobenius norm, $\sqrt{ \sum_{i, j} \abs{ A_{i, j} }^2 }$
///
/// @param[in] A
///     The matrix A.
///
/// @param[out] values
///     todo: undocumented.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup norm
///
template <typename matrix_type>
void colNorms(Norm in_norm,
              matrix_type& A,
              blas::real_type<typename matrix_type::value_type>* values,
              Options const& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return colNorms<Target::HostTask>(in_norm, A, values, opts);
            break;
        case Target::HostBatch:
        case Target::HostNest:
            return colNorms<Target::HostNest>(in_norm, A, values, opts);
            break;
        case Target::Devices:
            return colNorms<Target::Devices>(in_norm, A, values, opts);
            break;
    }
    throw std::exception();  // todo: invalid target
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void colNorms(Norm in_norm,
              Matrix<float>& A,
              float* values,
              Options const& opts);

template
void colNorms(Norm in_norm,
              Matrix<double>& A,
              double* values,
              Options const& opts);

template
void colNorms(Norm in_norm,
              Matrix< std::complex<float> >& A,
              float* values,
              Options const& opts);

template
void colNorms(Norm in_norm,
              Matrix< std::complex<double> >& A,
              double* values,
              Options const& opts);

} // namespace slate
