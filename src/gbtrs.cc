// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::gbtrs from internal::specialization::gbtrs
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel band LU solve.
/// Generic implementation for any target.
/// @ingroup gbsv_specialization
///
template <Target target, typename scalar_t>
void gbtrs(slate::internal::TargetType<target>,
           BandMatrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B, int64_t lookahead)
{
    assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    auto L = TriangularBandMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);
    auto U = TriangularBandMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);
//printf( "L kd %lld\n", L.bandwidth() );
//printf( "U kd %lld\n", U.bandwidth() );

    if (A.op() == Op::NoTrans) {
        // forward substitution, Y = L^{-1} P B
        tbsm(Side::Left, scalar_t(1.0), L, pivots, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // backward substitution, X = U^{-1} Y
        tbsm(Side::Left, scalar_t(1.0), U, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});
    }
    else {
        // forward substitution, Y = U^{-T} B
        tbsm(Side::Left, scalar_t(1.0), U, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // backward substitution, X = P^T L^{-T} Y
        tbsm(Side::Left, scalar_t(1.0), L, pivots, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gbsv_specialization
///
template <Target target, typename scalar_t>
void gbtrs(BandMatrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    internal::specialization::gbtrs(internal::TargetType<target>(),
                                    A, pivots, B, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel band LU solve.
///
/// Solves a system of linear equations
/// \[
///     A X = B
/// \]
/// with a general n-by-n band matrix $A$ using the LU factorization computed
/// by gbtrf. $A$ can be transposed or conjugate-transposed.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     The factors $L$ and $U$ from the factorization $A = L U$
///     as computed by gbtrf.
///
/// @param[in] pivots
///     The pivot indices that define the permutations
///     as computed by gbtrf.
///
/// @param[in,out] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///     On exit, the n-by-nrhs solution matrix $X$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup gbsv_computational
///
template <typename scalar_t>
void gbtrs(BandMatrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            gbtrs<Target::HostTask>(A, pivots, B, opts);
            break;
        case Target::HostNest:
            gbtrs<Target::HostNest>(A, pivots, B, opts);
            break;
        case Target::HostBatch:
            gbtrs<Target::HostBatch>(A, pivots, B, opts);
            break;
        case Target::Devices:
            gbtrs<Target::Devices>(A, pivots, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gbtrs<float>(
    BandMatrix<float>& A, Pivots& pivots,
    Matrix<float>& B,
    Options const& opts);

template
void gbtrs<double>(
    BandMatrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    Options const& opts);

template
void gbtrs< std::complex<float> >(
    BandMatrix< std::complex<float> >& A, Pivots& pivots,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void gbtrs< std::complex<double> >(
    BandMatrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
