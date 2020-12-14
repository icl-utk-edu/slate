// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::set from internal::specialization::set
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Set matrix entries.
/// Generic implementation for any target.
/// @ingroup set_specialization
///
template <Target target, typename scalar_t>
void set(slate::internal::TargetType<target>,
         scalar_t alpha, scalar_t beta, Matrix<scalar_t>& A)
{
    if (target == Target::Devices) {
        A.allocateBatchArrays();
        // todo: is this needed here when the matrix is already on devices?
        A.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::set<target>(alpha, beta, std::move(A));
        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup set_specialization
///
template <Target target, typename scalar_t>
void set(scalar_t alpha, scalar_t beta, Matrix<scalar_t>& A,
         Options const& opts)
{
    internal::specialization::set(internal::TargetType<target>(),
                                  alpha, beta, A);
}

//------------------------------------------------------------------------------
/// Set matrix entries.
/// Transposition is currently ignored.
/// TODO: Inspect transposition?
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///         The m-by-n matrix A.
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
/// @ingroup set
///
template <typename scalar_t>
void set(scalar_t alpha, scalar_t beta, Matrix<scalar_t>& A,
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
        default: // todo: this is to silence a warning, should err otherwise
            set<Target::HostTask>(alpha, beta, A, opts);
            break;
//      case Target::HostNest:
//          set<Target::HostNest>(alpha, beta, A, opts);
//          break;
//      case Target::HostBatch:
//          set<Target::HostBatch>(alpha, beta, A, opts);
//          break;
        case Target::Devices:
            set<Target::Devices>(alpha, beta, A, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void set(
    float alpha, float beta, Matrix<float>& A,
    Options const& opts);

template
void set(
    double alpha, double beta, Matrix<double>& A,
    Options const& opts);

template
void set(
    std::complex<float> alpha, std::complex<float> beta,
    Matrix<std::complex<float> >& A,
    Options const& opts);

template
void set(
    std::complex<double> alpha, std::complex<double> beta,
    Matrix<std::complex<double> >& A,
    Options const& opts);

} // namespace slate
