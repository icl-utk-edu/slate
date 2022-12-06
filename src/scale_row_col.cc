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
/// Apply row or column scaling, or both, to a Matrix.
/// Generic implementation for any target.
/// @ingroup scale_impl
///
template <Target target, typename scalar_t, typename scalar_t2>
void scale_row_col(
    Equed equed,
    std::vector<scalar_t2> const& R,
    std::vector<scalar_t2> const& C,
    Matrix<scalar_t>& A,
    Options const& opts )
{
    //scalar_t2* dR = nullptr;
    //scalar_t2* dC = nullptr;
    //scalar_t2** dRarray = nullptr;
    //scalar_t2** dCarray = nullptr;

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();

        //if (equed == Equed::Row || equed == Equed::Both) {
        //    dR      = blas::device_malloc< scalar_t2  >( A.m()  );
        //    dRarray = blas::device_malloc< scalar_t2* >( A.mt() );
        //}
        //if (equed == Equed::Col || equed == Equed::Both) {
        //    dC      = blas::device_malloc< scalar_t2  >( A.n()  );
        //    dCarray = blas::device_malloc< scalar_t2* >( A.nt() );
        //}
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::scale_row_col<target>(
            equed, R, C, std::move(A) );
            //dR, dRarray, dC, dCarray );
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();

    //if (target == Target::Devices) {
    //    blas::device_free( dR );
    //    blas::device_free( dRarray );
    //    blas::device_free( dC );
    //    blas::device_free( dCarray );
    //}
}

} // namespace impl

//------------------------------------------------------------------------------
/// Apply row or column scaling, or both, to a Matrix.
/// Transposition is currently ignored.
/// TODO: Inspect transposition?
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] equed
///     Form of scaling to do.
///     - Equed::Row:  sets $ A = diag(R) A         $
///     - Equed::Col:  sets $ A =         A diag(C) $
///     - Equed::Both: sets $ A = diag(R) A diag(C) $
///
/// @param[in,out] R
///     Vector of length m containing row scaling factors.
///
/// @param[in,out] C
///     Vector of length n containing column scaling factors.
///
/// @param[in,out] A
///     The m-by-n matrix A.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  [uses HostTask]
///       - HostBatch: [uses HostTask]
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup scale
///
template <typename scalar_t, typename scalar_t2>
void scale_row_col(
    Equed equed,
    std::vector<scalar_t2> const& R,
    std::vector<scalar_t2> const& C,
    Matrix<scalar_t>& A, Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
        case Target::HostNest:
        case Target::HostBatch:
            impl::scale_row_col<Target::HostTask>( equed, R, C, A, opts );
            break;

        case Target::Devices:
            impl::scale_row_col<Target::Devices>( equed, R, C, A, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void scale_row_col(
    Equed equed,
    std::vector<float> const& R,
    std::vector<float> const& C,
    Matrix<float>& A,
    Options const& opts);

template
void scale_row_col(
    Equed equed,
    std::vector<double> const& R,
    std::vector<double> const& C,
    Matrix<double>& A,
    Options const& opts);

// real R, C
template
void scale_row_col(
    Equed equed,
    std::vector< float > const& R,
    std::vector< float > const& C,
    Matrix< std::complex<float> >& A,
    Options const& opts);

template
void scale_row_col(
    Equed equed,
    std::vector< double > const& R,
    std::vector< double > const& C,
    Matrix< std::complex<double> >& A,
    Options const& opts);

// complex R, C
template
void scale_row_col(
    Equed equed,
    std::vector< std::complex<float> > const& R,
    std::vector< std::complex<float> > const& C,
    Matrix< std::complex<float> >& A,
    Options const& opts);

template
void scale_row_col(
    Equed equed,
    std::vector< std::complex<double> > const& R,
    std::vector< std::complex<double> > const& C,
    Matrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
