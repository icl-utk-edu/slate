// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "aux/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::getrs from internal::specialization::getrs
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel LU solve.
/// Generic implementation for any target.
/// @ingroup gesv_specialization
///
template <Target target, typename scalar_t>
void getrs(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B, int64_t lookahead)
{
    assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    auto L = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);
    auto U = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);

    if (A.op() == Op::NoTrans) {
        // Pivot the right hand side matrix.
        for (int64_t k = 0; k < B.mt(); ++k) {
            // swap rows in B(k:mt-1, 0:nt-1)
            internal::permuteRows<Target::HostTask>(
                Direction::Forward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                pivots.at(k), Layout::ColMajor);
        }

        // Forward substitution, Y = L^{-1} P B.
        trsm(Side::Left, scalar_t(1.0), L, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // Backward substitution, X = U^{-1} Y.
        trsm(Side::Left, scalar_t(1.0), U, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});
    }
    else {
        // Forward substitution, Y = U^{-T} B.
        trsm(Side::Left, scalar_t(1.0), U, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // Backward substitution, Xhat = L^{-T} Y.
        trsm(Side::Left, scalar_t(1.0), L, B,
             {{Option::Lookahead, lookahead},
              {Option::Target, target}});

        // Pivot the right hand side matrix, X = P^T Xhat
        for (int64_t k = B.mt()-1; k >= 0; --k) {
            // swap rows in B(k:mt-1, 0:nt-1)
            internal::permuteRows<Target::HostTask>(
                Direction::Backward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                pivots.at(k), Layout::ColMajor);
        }
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gesv_specialization
///
template <Target target, typename scalar_t>
void getrs(Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range&) {
        lookahead = 1;
    }

    internal::specialization::getrs(internal::TargetType<target>(),
                                    A, pivots, B, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel LU solve.
///
/// Solves a system of linear equations
/// \[
///     A X = B
/// \]
/// with a general n-by-n matrix $A$ using the LU factorization computed
/// by getrf. $A$ can be transposed or conjugate-transposed.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     The factors $L$ and $U$ from the factorization $A = P L U$
///     as computed by getrf.
///
/// @param[in] pivots
///     The pivot indices that define the permutation matrix $P$
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
/// @ingroup gesv_computational
///
template <typename scalar_t>
void getrs(Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
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
            getrs<Target::HostTask>(A, pivots, B, opts);
            break;
        case Target::HostNest:
            getrs<Target::HostNest>(A, pivots, B, opts);
            break;
        case Target::HostBatch:
            getrs<Target::HostBatch>(A, pivots, B, opts);
            break;
        case Target::Devices:
            getrs<Target::Devices>(A, pivots, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrs<float>(
    Matrix<float>& A, Pivots& pivots,
    Matrix<float>& B,
    Options const& opts);

template
void getrs<double>(
    Matrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    Options const& opts);

template
void getrs< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void getrs< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
