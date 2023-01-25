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
/// Distributed parallel matrix norm.
/// Generic implementation for any target.
/// @ingroup norm_impl
///
template <Target target, typename matrix_type>
void colNorms(
    Norm in_norm,
    matrix_type A,
    blas::real_type< typename matrix_type::value_type >* values,
    Options const& opts )
{
    using internal::mpi_max_nan;
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
        slate_not_implemented("Norm::One isn't yet supported.");
    }
    //---------
    // inf norm
    // max row sum = max_i sum_j abs( A_{i,j} )
    else if (in_norm == Norm::Inf) {
        slate_not_implemented("Norm::Inf isn't yet supported.");
    }
    //---------
    // Frobenius norm
    // sqrt( sum_{i,j} abs( A_{i,j} )^2 )
    else if (in_norm == Norm::Fro) {
        slate_not_implemented("Norm::Fro isn't yet supported.");
    }
    else {
        slate_error("invalid norm");
    }

    // todo: is this correct here?
    A.releaseWorkspace();
}

} // namespace impl

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
void colNorms(
    Norm in_norm,
    matrix_type& A,
    blas::real_type<typename matrix_type::value_type>* values,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::colNorms<Target::HostTask>( in_norm, A, values, opts );
            break;

        case Target::HostBatch:
        case Target::HostNest:
            impl::colNorms<Target::HostNest>( in_norm, A, values, opts );
            break;

        case Target::Devices:
            impl::colNorms<Target::Devices>( in_norm, A, values, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void colNorms(
    Norm in_norm,
    Matrix<float>& A,
    float* values,
    Options const& opts);

template
void colNorms(
    Norm in_norm,
    Matrix<double>& A,
    double* values,
    Options const& opts);

template
void colNorms(
    Norm in_norm,
    Matrix< std::complex<float> >& A,
    float* values,
    Options const& opts);

template
void colNorms(
    Norm in_norm,
    Matrix< std::complex<double> >& A,
    double* values,
    Options const& opts);

} // namespace slate
