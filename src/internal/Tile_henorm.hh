// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_HENORM_HH
#define SLATE_TILE_HENORM_HH

#include <blas.hh>

#include "slate/Tile.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Hermitian matrix norm.
/// The only difference from symmetric matrix norm is the diagonal is assumed
/// to be real.
/// @ingroup norm_tile
///
template <typename scalar_t>
void henorm(Norm norm, Tile<scalar_t> const& A,
            blas::real_type<scalar_t>* values)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::max;
    using blas::min;
    using blas::real;

    trace::Block trace_block("lapack::lanhe");

    assert(A.uplo() != Uplo::General);
    assert(A.op() == Op::NoTrans);
    assert(A.mb() == A.nb());

    if (norm == Norm::Max) {
        // max norm
        // values[0] = max_{i,j} A_{i,j}
        *values = lapack::lanhe(norm, A.uploPhysical(),
                                A.nb(),
                                A.data(), A.stride());
    }
    else if (norm == Norm::One || norm == Norm::Inf) {
        // one norm
        // values[j] = sum_i abs( A_{i,j} )
        std::fill_n(values, A.nb(), 0);
        for (int64_t j = 0; j < A.nb(); ++j) {
            if (A.uplo() == Uplo::Lower) {
                values[j] += std::abs( real( A(j, j) ) );  // diag (real)
                for (int64_t i = j+1; i < A.mb(); ++i) { // strictly lower
                    real_t tmp = std::abs(A(i, j));
                    values[j] += tmp;
                    values[i] += tmp;
                }
            }
            else { // upper
                for (int64_t i = 0; i < j; ++i) { // strictly upper
                    real_t tmp = std::abs(A(i, j));
                    values[j] += tmp;
                    values[i] += tmp;
                }
                values[j] += std::abs( real( A(j, j) ) );  // diag (real)
            }
        }
    }
    else if (norm == Norm::Fro) {
        // Frobenius norm
        // values[0] = scale, values[1] = sumsq such that
        // scale^2 * sumsq = sum_{i,j} abs( A_{i,j} )^2
        values[0] = 0;  // scale
        values[1] = 1;  // sumsq
        // off-diagonal elements
        if (A.uplo() == Uplo::Lower) {
            // lower: A[ j+1:mb, j ]
            for (int64_t j = 0; j < A.nb() - 1; ++j) {
                lapack::lassq(A.mb() - j - 1, &A.at(j+1, j), 1, &values[0], &values[1]);
            }
        }
        else {
            // upper: A[ 0:j-1, j ]
            for (int64_t j = 1; j < A.nb(); ++j) {
                lapack::lassq(j, &A.at(0, j), 1, &values[0], &values[1]);
            }
        }
        // double for symmetric entries
        values[1] *= 2;
        // diagonal elements (real)
        // if A is complex, double the stride: sizeof(complex) / sizeof(real) = 2
        lapack::lassq(A.nb(), reinterpret_cast<real_t const*>( &A.at(0, 0) ),
                      sizeof(scalar_t) / sizeof(real_t) * (A.stride()+1),
                      &values[0], &values[1]);
    }
    else {
        throw std::exception();  // invalid norm
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup norm_tile
///
template <typename scalar_t>
void henorm(Norm norm, Tile<scalar_t> const&& A,
            blas::real_type<scalar_t>* values)
{
    return henorm(norm, A, values);
}

} // namespace slate

#endif // SLATE_TILE_HENORM_HH
