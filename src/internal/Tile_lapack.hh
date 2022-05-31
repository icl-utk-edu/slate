// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_LAPACK_HH
#define SLATE_TILE_LAPACK_HH

#include <blas.hh>

#include "slate/Tile.hh"
#include "slate/internal/util.hh"

#include <list>

namespace slate {

//------------------------------------------------------------------------------
/// General matrix norm.
/// @ingroup norm_tile
///
template <typename scalar_t>
void genorm(Norm norm, NormScope scope, Tile<scalar_t> const& A,
            blas::real_type<scalar_t>* values)
{
    trace::Block trace_block("lapack::lange");

    assert(A.uploPhysical() == Uplo::General);
    assert(A.op() == Op::NoTrans);
    int64_t mb = A.mb();
    int64_t nb = A.nb();

    if (scope == NormScope::Matrix) {

        // max norm
        // values[0] = max_{i,j} A_{i,j}
        if (norm == Norm::Max) {
            *values = lapack::lange(norm,
                                    mb, nb,
                                    A.data(), A.stride());
        }
        // one norm
        // values[j] = sum_i abs( A_{i,j} )
        else if (norm == Norm::One) {
            for (int64_t j = 0; j < nb; ++j) {
                const scalar_t* Aj = &A.at(0, j);
                values[j] = std::abs( Aj[0] ); // A(0, j)
                for (int64_t i = 1; i < mb; ++i) {
                    values[j] += std::abs( Aj[i] );  // A(i, j)
                }
            }
        }
        // inf norm
        // values[i] = sum_j abs( A_{i,j} )
        else if (norm == Norm::Inf) {
            const scalar_t* Aj = &A.at(0, 0);
            for (int64_t i = 0; i < mb; ++i) {
                values[i] = std::abs( Aj[i] );  // A(i, 0)
            }
            for (int64_t j = 1; j < nb; ++j) {
                Aj = &A.at(0, j);
                for (int64_t i = 0; i < mb; ++i) {
                    values[i] += std::abs( Aj[i] );  // A(i, j)
                }
            }
        }
        // Frobenius norm
        // values[0] = scale, values[1] = sumsq such that
        // scale^2 * sumsq = sum_{i,j} abs( A_{i,j} )^2
        else if (norm == Norm::Fro) {
            values[0] = 0;  // scale
            values[1] = 1;  // sumsq
            for (int64_t j = 0; j < nb; ++j) {
                lapack::lassq(mb, &A.at(0, j), 1, &values[0], &values[1]);
            }
        }
        else {
            throw std::exception();  // invalid norm
        }
    }
    else if (scope == NormScope::Columns) {

        if (norm == Norm::Max) {
            // All max norm
            // values[j] = max_i abs( A_{i,j} )
            // todo: handle layout and transpose, also sliced matrices
            // todo: parallel for ??
            for (int64_t j = 0; j < nb; ++j) {
                values[j] = lapack::lange(norm,
                                          mb, 1,
                                          A.data() + j*A.stride(), A.stride());
            }
        }
        else {
            slate_error("Not implemented yet");
        }
    }
    else {
        slate_error("Not implemented yet");
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup norm_tile
///
template <typename scalar_t>
void genorm(Norm norm, NormScope scope, Tile<scalar_t> const&& A,
            blas::real_type<scalar_t>* values)
{
    return genorm(norm, scope, A, values);
}

//------------------------------------------------------------------------------
/// Trapezoid and triangular matrix norm.
/// @ingroup norm_tile
///
template <typename scalar_t>
void trnorm(Norm norm, Diag diag, Tile<scalar_t> const& A,
            blas::real_type<scalar_t>* values)
{
    using blas::max;
    using blas::min;

    trace::Block trace_block("lapack::lantr");

    assert(A.uploPhysical() != Uplo::General);
    assert(A.op() == Op::NoTrans);
    int64_t mb = A.mb();
    int64_t nb = A.nb();

    if (norm == Norm::Max) {
        // max norm
        // values[0] = max_{i,j} A_{i,j}
        *values = lapack::lantr(norm, A.uploPhysical(), diag,
                                mb, nb,
                                A.data(), A.stride());
    }
    else if (norm == Norm::One) {
        // one norm
        // values[j] = sum_i abs( A_{i,j} )
        for (int64_t j = 0; j < nb; ++j) {
            values[j] = 0;
            const scalar_t* Aj = &A.at(0, j);
            // diagonal element
            if (j < mb) {
                if (diag == Diag::Unit) {
                    values[j] += 1;
                }
                else {
                    values[j] += std::abs(Aj[j]);  // A(j, j)
                }
            }
            // off-diagonal elements
            if (A.uplo() == Uplo::Lower) {
                for (int64_t i = j+1; i < mb; ++i) { // strictly lower
                    values[j] += std::abs(Aj[i]);  // A(i, j)
                }
            }
            else {
                for (int64_t i = 0; i < j && i < mb; ++i) { // strictly upper
                    values[j] += std::abs(Aj[i]);  // A(i, j)
                }
            }
        }
    }
    else if (norm == Norm::Inf) {
        // inf norm
        // values[i] = sum_j abs( A_{i,j} )
        for (int64_t i = 0; i < mb; ++i) {
            values[i] = 0;
        }
        for (int64_t j = 0; j < nb; ++j) {
            // diagonal element
            const scalar_t* Aj = &A.at(0, j);
            if (j < mb) {
                if (diag == Diag::Unit) {
                    values[j] += 1;
                }
                else {
                    values[j] += std::abs(Aj[j]);  // A(j, j)
                }
            }
            // off-diagonal elements
            if (A.uplo() == Uplo::Lower) {
                for (int64_t i = j+1; i < mb; ++i) { // strictly lower
                    values[i] += std::abs(Aj[i]);  // A(i, j)
                }
            }
            else {
                for (int64_t i = 0; i < j && i < mb; ++i) { // strictly upper
                    values[i] += std::abs(Aj[i]);  // A(i, j)
                }
            }
        }
    }
    else if (norm == Norm::Fro) {
        // Frobenius norm
        // values[0] = scale, values[1] = sumsq such that
        // scale^2 * sumsq = sum_{i,j} abs( A_{i,j} )^2
        values[0] = 0;  // scale
        values[1] = 1;  // sumsq
        if (diag == Diag::Unit) {
            // diagonal elements: sum 1^2 + ... + 1^2 = min( mb, nb )
            values[0] = 1;
            values[1] = min(A.mb(), A.nb());
            // off-diagonal elements
            if (A.uplo() == Uplo::Lower) {
                // strictly lower: A[ j+1:mb, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = max(A.mb() - (j+1), 0);
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(j+1, j), 1, &values[0], &values[1]);
                }
            }
            else {
                // strictly upper: A[ 0:j-1, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = min(j, A.mb());
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(0, j), 1, &values[0], &values[1]);
                }
            }
        }
        else {
            if (A.uplo() == Uplo::Lower) {
                // lower: A[ j:mb, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = max(A.mb() - j, 0);
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(j, j), 1, &values[0], &values[1]);
                }
            }
            else {
                // upper: A[ 0:j, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = min(j+1, A.mb());
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(0, j), 1, &values[0], &values[1]);
                }
            }
        }
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
void trnorm(Norm norm, Tile<scalar_t> const&& A,
            blas::real_type<scalar_t>* values)
{
    return trnorm(norm, A, values);
}

//------------------------------------------------------------------------------
/// Cholesky factorization of tile: $L L^H = A$ or $U^H U = A$.
/// uplo is set in the tile.
/// @ingroup posv_tile
///
template <typename scalar_t>
int64_t potrf(Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::potrf");

    return lapack::potrf(A.uploPhysical(),
                         A.nb(),
                         A.data(), A.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup posv_tile
///
template <typename scalar_t>
int64_t potrf(Tile<scalar_t>&& A)
{
    return potrf(A);
}


//------------------------------------------------------------------------------
/// Triangular inversion of tile.
/// uplo is set in the tile.
/// @ingroup tr_tile
///
template <typename scalar_t>
int64_t trtri(Diag diag, Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::trtri");

    return lapack::trtri(A.uploPhysical(), diag,
                         A.nb(),
                         A.data(), A.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup tr_tile
///
template <typename scalar_t>
int64_t trtri(Diag diag, Tile<scalar_t>&& A)
{
    return trtri(diag, A);
}

//------------------------------------------------------------------------------
/// Triangular multiplication $L = L^H L$ or $U = U U^H$
/// uplo is set in the tile.
/// @ingroup tr_tile
///
template <typename scalar_t>
int64_t trtrm(Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::lauum");

    return lapack::lauum(A.uploPhysical(),
                         A.nb(),
                         A.data(), A.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup tr_tile
///
template <typename scalar_t>
int64_t trtrm(Tile<scalar_t>&& A)
{
    return trtrm(A);
}

//------------------------------------------------------------------------------
/// Reduces a complex Hermitian positive-definite generalized eigenvalue problem
/// to the standard form of single tile.
/// uplo is set in the tile.
/// @ingroup hegv_tile
///
template <typename scalar_t>
int64_t hegst(int64_t itype, Tile<scalar_t>& A, Tile<scalar_t>& B)
{
    trace::Block trace_block("lapack::hegst");

    return lapack::hegst(itype, A.uploPhysical(), A.nb(), A.data(), A.stride(),
                                                          B.data(), B.stride());
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup hegv_tile
///
template <typename scalar_t>
int64_t hegst(int64_t itype, Tile<scalar_t>&& A, Tile<scalar_t>&& B)
{
    return hegst(itype, A, B);
}

//------------------------------------------------------------------------------
/// Scale matrix entries by the real scalar numer/denom.
/// uplo is set in the tile.
/// @ingroup scale_tile
///
template<typename scalar_t>
int64_t scale(blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
              Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::lascl");
    //TODO lower and upper bandwidth values
    return lapack::lascl((lapack::MatrixType)A.uploPhysical(),
                         0, 0, denom, numer, A.mb(), A.nb(),
                         A.data(), A.stride());
}

//------------------------------------------------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup scale_tile
///
template<typename scalar_t>
int64_t scale(blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
              Tile<scalar_t>&& A)
{
    return scale(numer, denom, A);
}

//------------------------------------------------------------------------------
/// Scale matrix entries by the real scalar numer/denom.
/// uplo is set in the tile.
/// @ingroup scale_tile
///
template<typename scalar_t>
int64_t scale(scalar_t value, Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::lascl");

    scalar_t one = 1;
    //TODO lower and upper bandwidth values
    return lapack::lascl((lapack::MatrixType)A.uploPhysical(),
                         0, 0, one, value, A.mb(), A.nb(),
                         A.data(), A.stride());
}
//------------------------------------------------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup scale_tile
///
template<typename scalar_t>
int64_t scale(scalar_t value, Tile<scalar_t>&& A)
{
    return scale(value, A);
}

} // namespace slate

#endif // SLATE_TILE_LAPACK_HH
