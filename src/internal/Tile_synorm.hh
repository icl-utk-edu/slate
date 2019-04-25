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

#ifndef SLATE_TILE_SYNORM_HH
#define SLATE_TILE_SYNORM_HH

#include <blas.hh>

#include "slate/Tile.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Symmetric matrix norm.
/// @ingroup norm_tile
///
template <typename scalar_t>
void synorm(Norm norm, Tile<scalar_t> const& A,
            blas::real_type<scalar_t>* values)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::max;
    using blas::min;

    trace::Block trace_block("lapack::lansy");

    assert(A.uplo() != Uplo::General);
    assert(A.op() == Op::NoTrans);
    assert(A.mb() == A.nb());

    if (norm == Norm::Max) {
        // max norm
        // values[0] = max_{i,j} A_{i,j}
        *values = lapack::lansy(norm, A.uploPhysical(),
                                A.nb(),
                                A.data(), A.stride());
    }
    else if (norm == Norm::One || norm == Norm::Inf) {
        // one norm
        // values[j] = sum_i abs( A_{i,j} )
        std::fill_n(values, A.nb(), 0);
        for (int64_t j = 0; j < A.nb(); ++j) {
            if (A.uplo() == Uplo::Lower) {
                values[j] += std::abs(A(j, j));  // diag
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
                values[j] += std::abs(A(j, j));  // diag
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
        // diagonal elements
        lapack::lassq(A.nb(), &A.at(0, 0), A.stride()+1, &values[0], &values[1]);
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
void synorm(Norm norm, Tile<scalar_t> const&& A,
            blas::real_type<scalar_t>* values)
{
    return synorm(norm, A, values);
}

//------------------------------------------------------------------------------
/// Symmetric matrix norm, off-diagonal tiles.
/// @ingroup norm_tile
///
template <typename scalar_t>
void synormOffdiag(Norm norm, Tile<scalar_t> const& A,
                    blas::real_type<scalar_t>* col_sums,
                    blas::real_type<scalar_t>* row_sums)
{
    using real_t = blas::real_type<scalar_t>;

    trace::Block trace_block("lapack::lansy2");

    assert(A.uplo() == Uplo::General);
    assert(A.op() == Op::NoTrans);

    // one norm
    // col_sums[j] = sum_i abs( A_{i,j} )
    // row_sums[i] = sum_j abs( A_{i,j} )
    if (norm == Norm::One || norm == Norm::Inf) {
        std::fill_n(row_sums, A.mb(), 0);
        for (int64_t j = 0; j < A.nb(); ++j) {
            real_t tmp = std::abs(A(0, j));
            col_sums[j] = tmp;
            row_sums[0] += tmp;
            for (int64_t i = 1; i < A.mb(); ++i) {
                tmp = std::abs(A(i, j));
                col_sums[j] += tmp;
                row_sums[i] += tmp;
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
void synormOffdiag(Norm norm, Tile<scalar_t> const&& A,
                    blas::real_type<scalar_t>* col_sums,
                    blas::real_type<scalar_t>* row_sums)
{
    return synormOffdiag(norm, A, col_sums, row_sums);
}

} // namespace slate

#endif // SLATE_TILE_SYNORM_HH
