// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Set matrix entries.
/// Generic implementation for any target.
/// @ingroup set_impl
///
template <Target target, typename scalar_t>
void set(
    scalar_t offdiag_value, scalar_t diag_value,
    Matrix<scalar_t>& A,
    Options const& opts )
{
    if (target == Target::Devices) {
        A.allocateBatchArrays();
        // todo: is this needed here when the matrix is already on devices?
        A.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::set<target>( offdiag_value, diag_value, std::move( A ) );
        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
}

} // namespace impl

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
void set(
    scalar_t offdiag_value, scalar_t diag_value,
    Matrix<scalar_t>& A,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
        default: // todo: this is to silence a warning, should err otherwise
            impl::set<Target::HostTask>( offdiag_value, diag_value, A, opts );
            break;
//      case Target::HostNest:
//          set<Target::HostNest>( offdiag_value, diag_value, A, opts );
//          break;
//      case Target::HostBatch:
//          set<Target::HostBatch>( offdiag_value, diag_value, A, opts );
//          break;

        case Target::Devices:
            impl::set<Target::Devices>( offdiag_value, diag_value, A, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void set(
    float offdiag_value, float diag_value,
    Matrix<float>& A,
    Options const& opts);

template
void set(
    double offdiag_value, double diag_value,
    Matrix<double>& A,
    Options const& opts);

template
void set(
    std::complex<float> offdiag_value, std::complex<float> diag_value,
    Matrix<std::complex<float> >& A,
    Options const& opts);

template
void set(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    Matrix<std::complex<double> >& A,
    Options const& opts);

//==============================================================================
// For BaseTrapezoidMatrix.
//==============================================================================

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Set matrix entries.
/// Generic implementation for any target.
/// @ingroup set_impl
///
template <Target target, typename scalar_t>
void set(
    scalar_t offdiag_value, scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts )
{
    if (target == Target::Devices) {
        A.allocateBatchArrays();
        // todo: is this needed here when the matrix is already on devices?
        A.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::set<target>( offdiag_value, diag_value, std::move( A ) );
        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
}

} // namespace impl

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
void set(
    scalar_t offdiag_value, scalar_t diag_value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
        default: // todo: this is to silence a warning, should err otherwise
            impl::set<Target::HostTask>( offdiag_value, diag_value, A, opts );
            break;
//      case Target::HostNest:
//          set<Target::HostNest>( offdiag_value, diag_value, A, opts );
//          break;
//      case Target::HostBatch:
//          set<Target::HostBatch>( offdiag_value, diag_value, A, opts );
//          break;

        case Target::Devices:
            impl::set<Target::Devices>( offdiag_value, diag_value, A, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void set(
    float offdiag_value, float diag_value,
    BaseTrapezoidMatrix<float>& A,
    Options const& opts);

template
void set(
    double offdiag_value, double diag_value,
    BaseTrapezoidMatrix<double>& A,
    Options const& opts);

template
void set(
    std::complex<float> offdiag_value, std::complex<float> diag_value,
    BaseTrapezoidMatrix<std::complex<float> >& A,
    Options const& opts);

template
void set(
    std::complex<double> offdiag_value, std::complex<double> diag_value,
    BaseTrapezoidMatrix<std::complex<double> >& A,
    Options const& opts);

} // namespace slate
