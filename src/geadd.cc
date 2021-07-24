// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

#include <list>
#include <tuple>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::geadd from internal::specialization::geadd
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix-matrix addition.
/// Generic implementation for any target.
/// @ingroup geadd_specialization
///
template <Target target, typename scalar_t>
void geadd(slate::internal::TargetType<target>,
           scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           int64_t lookahead)
{
    if (target == Target::Devices) {
        B.allocateBatchArrays();
        B.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::geadd<target>(alpha, std::move(A),
                                beta, std::move(B));
        #pragma omp taskwait
        B.tileUpdateAllOrigin();
    }

    B.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup geadd_specialization
///
template <Target target, typename scalar_t>
void geadd(scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           Options const& opts)
{
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    internal::specialization::geadd(internal::TargetType<target>(),
                                    alpha, A,
                                    beta,  B,
                                    lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel general matrix-matrix addition.
/// Performs the matrix-matrix operation
/// \[
///     B = \alpha A + \beta B,
/// \]
/// where alpha and beta are scalars, and $A$ and $B$ are matrices, with
/// $A$ an m-by-n matrix and $B$ a m-by-n matrix.
/// Transposition is currently not supported.
/// TODO: Support transposition.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The m-by-n matrix A.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result $\alpha A + \beta B$.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of blocks to overlap communication and computation.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup geadd
///
template <typename scalar_t>
void geadd(scalar_t alpha, Matrix<scalar_t>& A,
           scalar_t beta,  Matrix<scalar_t>& B,
           Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            geadd<Target::HostTask>(alpha, A, beta, B, opts);
            break;
        case Target::HostNest:
            geadd<Target::HostNest>(alpha, A, beta, B, opts);
            break;
        case Target::HostBatch:
            geadd<Target::HostBatch>(alpha, A, beta, B, opts);
            break;
        case Target::Devices:
            geadd<Target::Devices>(alpha, A, beta, B, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geadd<float>(
    float alpha, Matrix<float>& A,
    float beta,  Matrix<float>& B,
    Options const& opts);

template
void geadd<double>(
    double alpha, Matrix<double>& A,
    double beta,  Matrix<double>& B,
    Options const& opts);

template
void geadd< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
    std::complex<float> beta,  Matrix< std::complex<float> >& B,
    Options const& opts);

template
void geadd< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
    std::complex<double> beta,  Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
