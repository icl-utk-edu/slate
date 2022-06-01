// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel LU inversion (out-of-place version).
///
/// Computes the inverse of a matrix $A$ using the LU factorization $A = L*U$
/// computed by `getrf`. Stores the result in $B$. Does not change $A$.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     On entry, the factors $L$ and $U$ from the factorization $A = P L U$
///     as computed by getrf.
///     On exit, the inverse of the original matrix $A$.
///
/// @param[in] pivots
///     The pivot indices that define the permutation matrix $P$
///     as computed by getrf.
///
/// @param[out] B
///     On exit, if return value = 0, the n-by-n inverse of marix $A$.
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
void getri(Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           Options const& opts)
{
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    slate_assert(A.mt() == A.nt());  // square
    slate_assert(B.mt() == B.nt());  // square
    slate_assert(B.mt() == A.mt());  // same size

    // B = Identity.
    set( zero, one, B, opts );

    // solve
    getrs(A, pivots, B, opts);

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getri<float>(
    Matrix<float>& A, Pivots& pivots,
    Matrix<float>& B,
    Options const& opts);

template
void getri<double>(
    Matrix<double>& A, Pivots& pivots,
    Matrix<double>& B,
    Options const& opts);

template
void getri< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void getri< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
