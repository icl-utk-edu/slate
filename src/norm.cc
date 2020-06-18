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
/// Distributed parallel general matrix norm.
/// Generic implementation for any target.
/// @ingroup norm_specialization
///
template <Target target, typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(slate::internal::TargetType<target>,
     Norm in_norm, matrix_type A)
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;

    // Undo any transpose, which switches one <=> inf norms.
    if (A.op() == Op::ConjTrans || A.op() == Op::Trans) {
        if (in_norm == Norm::One)
            in_norm = Norm::Inf;
        else if (in_norm == Norm::Inf)
            in_norm = Norm::One;
    }
    if (A.op() == Op::ConjTrans)
        A = conjTranspose(A);
    else if (A.op() == Op::Trans)
        A = transpose(A);

    //---------
    // max norm
    // max_{i,j} abs( A_{i,j} )
    if (in_norm == Norm::Max) {

        real_t local_max;
        real_t global_max;

        // TODO: Allocate batch arrays here, not in internal.
        if (target == Target::Devices)
            A.reserveDeviceWorkspace();

        #pragma omp parallel
        #pragma omp master
        {
            internal::norm<target>(in_norm, NormScope::Matrix, std::move(A), &local_max);
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
                MPI_Allreduce(&local_max, &global_max,
                              1, mpi_type<real_t>::value,
                              op_max_nan, A.mpiComm()));
        }

        #pragma omp critical(slate_mpi)
        {
            slate_mpi_call(
                MPI_Op_free(&op_max_nan));
        }

        A.clearWorkspace();

        return global_max;
    }
    //---------
    // one norm
    // max col sum = max_j sum_i abs( A_{i,j} )
    else if (in_norm == Norm::One) {

        std::vector<real_t> local_sums(A.n());

        if (target == Target::Devices)
            A.reserveDeviceWorkspace();

        #pragma omp parallel
        #pragma omp master
        {
            internal::norm<target>(in_norm, NormScope::Matrix, std::move(A), local_sums.data());
        }

        std::vector<real_t> global_sums(A.n());

        #pragma omp critical(slate_mpi)
        {
            trace::Block trace_block("MPI_Allreduce");
            slate_mpi_call(
                MPI_Allreduce(local_sums.data(), global_sums.data(),
                              A.n(), mpi_type<real_t>::value,
                              MPI_SUM, A.mpiComm()));
        }

        A.clearWorkspace();

        return lapack::lange(Norm::Max, 1, A.n(), global_sums.data(), 1);
    }
    //---------
    // inf norm
    // max row sum = max_i sum_j abs( A_{i,j} )
    else if (in_norm == Norm::Inf) {

        std::vector<real_t> local_sums(A.m());

        if (target == Target::Devices)
            A.reserveDeviceWorkspace();

        #pragma omp parallel
        #pragma omp master
        {
            internal::norm<target>(in_norm, NormScope::Matrix, std::move(A), local_sums.data());
        }

        std::vector<real_t> global_sums(A.m());

        #pragma omp critical(slate_mpi)
        {
            trace::Block trace_block("MPI_Allreduce");
            slate_mpi_call(
                MPI_Allreduce(local_sums.data(), global_sums.data(),
                              A.m(), mpi_type<real_t>::value,
                              MPI_SUM, A.mpiComm()));
        }

        A.releaseWorkspace();

        return lapack::lange(Norm::Max, 1, A.m(), global_sums.data(), 1);
    }
    //---------
    // Frobenius norm
    // sqrt( sum_{i,j} abs( A_{i,j} )^2 )
    else if (in_norm == Norm::Fro) {

        real_t local_values[2];
        real_t local_sumsq;
        real_t global_sumsq;

        if (target == Target::Devices)
            A.reserveDeviceWorkspace();

        #pragma omp parallel
        #pragma omp master
        {
            internal::norm<target>(in_norm, NormScope::Matrix, std::move(A), local_values);
        }

        #pragma omp critical(slate_mpi)
        {
            trace::Block trace_block("MPI_Allreduce");
            // todo: propogate scale
            local_sumsq = local_values[0] * local_values[0] * local_values[1];
            slate_mpi_call(
                MPI_Allreduce(&local_sumsq, &global_sumsq,
                              1, mpi_type<real_t>::value,
                              MPI_SUM, A.mpiComm()));
        }

        A.clearWorkspace();

        return sqrt(global_sumsq);
    }
    else {
        throw std::exception();  // todo: invalid norm
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup norm_specialization
///
template <Target target, typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(Norm norm, matrix_type& A,
     Options const& opts)
{
    return internal::specialization::norm(internal::TargetType<target>(),
                                          norm, A);
}

//------------------------------------------------------------------------------
/// Distributed parallel general matrix norm.
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
blas::real_type<typename matrix_type::value_type>
norm(Norm in_norm, matrix_type& A,
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
            return norm<Target::HostTask>(in_norm, A, opts);
            break;
        case Target::HostBatch:
        case Target::HostNest:
            return norm<Target::HostNest>(in_norm, A, opts);
            break;
        case Target::Devices:
            return norm<Target::Devices>(in_norm, A, opts);
            break;
    }
    throw std::exception();  // todo: invalid target
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
float norm(
    Norm in_norm, Matrix<float>& A,
    Options const& opts);

template
double norm(
    Norm in_norm, Matrix<double>& A,
    Options const& opts);

template
float norm(
    Norm in_norm, Matrix< std::complex<float> >& A,
    Options const& opts);

template
double norm(
    Norm in_norm, Matrix< std::complex<double> >& A,
    Options const& opts);

//--------------------
template
float norm(
    Norm in_norm, HermitianMatrix<float>& A,
    Options const& opts);

template
double norm(
    Norm in_norm, HermitianMatrix<double>& A,
    Options const& opts);

template
float norm(
    Norm in_norm, HermitianMatrix< std::complex<float> >& A,
    Options const& opts);

template
double norm(
    Norm in_norm, HermitianMatrix< std::complex<double> >& A,
    Options const& opts);

//--------------------
template
float norm(
    Norm in_norm, SymmetricMatrix<float>& A,
    Options const& opts);

template
double norm(
    Norm in_norm, SymmetricMatrix<double>& A,
    Options const& opts);

template
float norm(
    Norm in_norm, SymmetricMatrix< std::complex<float> >& A,
    Options const& opts);

template
double norm(
    Norm in_norm, SymmetricMatrix< std::complex<double> >& A,
    Options const& opts);

//--------------------
template
float norm(
    Norm in_norm, TrapezoidMatrix<float>& A,
    Options const& opts);

template
double norm(
    Norm in_norm, TrapezoidMatrix<double>& A,
    Options const& opts);

template
float norm(
    Norm in_norm, TrapezoidMatrix< std::complex<float> >& A,
    Options const& opts);

template
double norm(
    Norm in_norm, TrapezoidMatrix< std::complex<double> >& A,
    Options const& opts);

//--------------------
template
float norm(
    Norm in_norm, BandMatrix<float>& A,
    Options const& opts);

template
double norm(
    Norm in_norm, BandMatrix<double>& A,
    Options const& opts);

template
float norm(
    Norm in_norm, BandMatrix< std::complex<float> >& A,
    Options const& opts);

template
double norm(
    Norm in_norm, BandMatrix< std::complex<double> >& A,
    Options const& opts);

//--------------------
template
float norm(
    Norm in_norm, HermitianBandMatrix<float>& A,
    Options const& opts);

template
double norm(
    Norm in_norm, HermitianBandMatrix<double>& A,
    Options const& opts);

template
float norm(
    Norm in_norm, HermitianBandMatrix< std::complex<float> >& A,
    Options const& opts);

template
double norm(
    Norm in_norm, HermitianBandMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
