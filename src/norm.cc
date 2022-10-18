// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"
#include "slate/internal/mpi.hh"

#include <list>
#include <tuple>

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix norm.
/// Generic implementation for any target.
/// @ingroup norm_impl
///
template <Target target, typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(
    Norm in_norm, matrix_type A,
    Options const& opts )
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;
    using internal::mpi_max_nan;

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

        if (target == Target::Devices) {
            A.reserveDeviceWorkspace();
        }

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

        if (target == Target::Devices) {
            A.reserveDeviceWorkspace();
        }

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

        if (target == Target::Devices) {
            A.reserveDeviceWorkspace();
        }

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
        slate_error("invalid norm.");
    }
}

} // namespace impl

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
norm(
    Norm in_norm, matrix_type& A,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return impl::norm<Target::HostTask>( in_norm, A, opts );
            break;

        case Target::HostBatch:
        case Target::HostNest:
            return impl::norm<Target::HostNest>( in_norm, A, opts );
            break;

        case Target::Devices:
            return impl::norm<Target::Devices>( in_norm, A, opts );
            break;
    }
    return -1.0;  // unreachable; silence error
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
