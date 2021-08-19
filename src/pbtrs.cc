// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianBandMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::pbtrs from internal::specialization::pbtrs
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky solve.
/// Generic implementation for any target.
/// @ingroup pbsv_specialization
///
template <Target target, typename scalar_t>
void pbtrs(slate::internal::TargetType<target>,
           HermitianBandMatrix<scalar_t> A,
           Matrix<scalar_t>& B, int64_t lookahead)
{
    // assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper)
        A = conjTranspose(A);

    auto L = TriangularBandMatrix<scalar_t>(Diag::NonUnit, A);
    auto LT = conjTranspose(L);

    tbsm(Side::Left, scalar_t(1.0), L, B,
         {{Option::Lookahead, lookahead},
          {Option::Target, target}});

    tbsm(Side::Left, scalar_t(1.0), LT, B,
         {{Option::Lookahead, lookahead},
          {Option::Target, target}});

}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup pbsv_specialization
///
template <Target target, typename scalar_t>
void pbtrs(HermitianBandMatrix<scalar_t>& A,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    internal::specialization::pbtrs(internal::TargetType<target>(),
                                    A, B, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky solve.
///
/// Solves a system of linear equations
/// \[
///     A X = B
/// \]
/// with a Hermitian positive definite hermitian matrix $A$ using the Cholesky
/// factorization $A = U^H U$ or $A = L L^H$ computed by potrf.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     The n-by-n triangular factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$, computed by potrf.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
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
/// @ingroup pbsv_computational
///
template <typename scalar_t>
void pbtrs(HermitianBandMatrix<scalar_t>& A,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            pbtrs<Target::HostTask>(A, B, opts);
            break;
        case Target::HostNest:
            pbtrs<Target::HostNest>(A, B, opts);
            break;
        case Target::HostBatch:
            pbtrs<Target::HostBatch>(A, B, opts);
            break;
        case Target::Devices:
            pbtrs<Target::Devices>(A, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void pbtrs<float>(
    HermitianBandMatrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void pbtrs<double>(
    HermitianBandMatrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void pbtrs< std::complex<float> >(
    HermitianBandMatrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void pbtrs< std::complex<double> >(
    HermitianBandMatrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
