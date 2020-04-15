//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

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
/// @ingroup trtri_tile
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
/// @ingroup trtri_tile
///
template <typename scalar_t>
int64_t trtri(Diag diag, Tile<scalar_t>&& A)
{
    return trtri(diag, A);
}

//------------------------------------------------------------------------------
/// Triangular multiplication $L = L^H L$ or $U = U U^H$
/// uplo is set in the tile.
/// @ingroup trtrm_tile
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
/// @ingroup trtrm_tile
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
/// @ingroup hegst_tile
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
/// @ingroup hegst_tile
///
template <typename scalar_t>
int64_t hegst(int64_t itype, Tile<scalar_t>&& A, Tile<scalar_t>&& B)
{
    return hegst(itype, A, B);
}

} // namespace slate

#endif // SLATE_TILE_LAPACK_HH
