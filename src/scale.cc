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
/// Scale matrix entries by the real scalar numer/denom.
/// Generic implementation for any target.
/// @ingroup set_impl
///
template <Target target, typename scalar_t>
void scale(
    blas::real_type<scalar_t> numer,
    blas::real_type<scalar_t> denom,
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
        internal::scale<target>(numer, denom, std::move(A));
        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Scale matrix entries by the real scalar numer/denom.
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
void scale(
    blas::real_type<scalar_t> numer,
    blas::real_type<scalar_t> denom,
    Matrix<scalar_t>& A,
    Options const& opts )
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
            impl::scale<Target::HostTask>( numer, denom, A, opts );
            break;
        // case Target::HostNest:
        //     scale<Target::HostNest>(numer, denom, A, opts);
        //     break;
        // case Target::HostBatch:
        //     scale<Target::HostBatch>(numer, denom, A, opts);
        //     break;

        case Target::Devices:
            impl::scale<Target::Devices>( numer, denom, A, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void scale(
    float numer, float denom, Matrix<float>& A,
    Options const& opts);

template
void scale(
    double numer, double denom, Matrix<double>& A,
    Options const& opts);

template
void scale(
    float numer, float denom,
    Matrix<std::complex<float> >& A,
    Options const& opts);

template
void scale(
    double numer, double denom,
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
void scale(
    blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
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
        internal::scale<target>(numer, denom, std::move(A));
        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Scale matrix entries by the real scalar numer/denom.
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
void scale(
    blas::real_type<scalar_t> numer,
    blas::real_type<scalar_t> denom,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts )
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
            impl::scale<Target::HostTask>( numer, denom, A, opts );
            break;
        // case Target::HostNest:
        //     scale<Target::HostNest>(numer, denom, A, opts);
        //     break;
        // case Target::HostBatch:
        //     scale<Target::HostBatch>(numer, denom, A, opts);
        //     break;

        case Target::Devices:
            impl::scale<Target::Devices>( numer, denom, A, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void scale(
    float numer, float denom, BaseTrapezoidMatrix<float>& A,
    Options const& opts);

template
void scale(
    double numer, double denom, BaseTrapezoidMatrix<double>& A,
    Options const& opts);

template
void scale(
    float numer, float denom,
    BaseTrapezoidMatrix<std::complex<float> >& A,
    Options const& opts);

template
void scale(
    double numer, double denom,
    BaseTrapezoidMatrix<std::complex<double> >& A,
    Options const& opts);

} // namespace slate
