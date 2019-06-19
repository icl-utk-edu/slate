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

#ifndef SLATE_TILE_AUX_HH
#define SLATE_TILE_AUX_HH

// #include "slate/Tile.hh"
#include "slate/internal/util.hh"
#include "slate/internal/device.hh"

namespace slate {

// forward declaration
template <typename scalar_t>
class Tile;

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(Tile<src_scalar_t> const& A, Tile<dst_scalar_t>& B)
{
//  trace::Block trace_block("aux::copy");

    assert(A.mb() == B.mb());
    assert(A.nb() == B.nb());

    for (int64_t j = 0; j < B.nb(); ++j)
        for (int64_t i = 0; i < B.mb(); ++i)
            B.at(i, j) = A.at(i, j);
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void gecopy(Tile<src_scalar_t> const&& A, Tile<dst_scalar_t>&& B)
{
    gecopy(A, B);
}

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(Tile<src_scalar_t> const& A, Tile<dst_scalar_t>& B)
{
//  trace::Block trace_block("aux::copy");

    // TODO: Can be loosened?
    assert(A.uplo() != Uplo::General);
    assert(B.uplo() == A.uplo());

    assert(A.op() == Op::NoTrans);
    assert(B.op() == A.op());

    assert(A.mb() == B.mb());
    assert(A.nb() == B.nb());

    for (int64_t j = 0; j < B.nb(); ++j) {
        if (j < B.mb()) {
            B.at(j, j) = A.at(j, j);
        }
        if (B.uplo() == Uplo::Lower) {
            for (int64_t i = j; i < B.mb(); ++i) {
                B.at(i, j) = A.at(i, j);
            }
        }
        else {
            for (int64_t i = 0; i <= j && i < B.mb(); ++i) {
                B.at(i, j) = A.at(i, j);
            }
        }
    }
}

//-----------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup copy_tile
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(Tile<src_scalar_t> const&& A, Tile<dst_scalar_t>&& B)
{
    tzcopy(A, B);
}

//------------------------------------------------------------------------------
/// @deprecated
/// In-place conversion between column and row-major layout for square tiles.
/// Takes a pointer to the original tile in MatrixStorage, instead of a
/// reference to a copy of the tile, in order to adjust the tile's layout flag.
/// @ingroup convert_tile
///
template <typename scalar_t>
void convert_layout(Tile<scalar_t>* X)
{
    trace::Block trace_block("slate::convert_layout");
    assert(X->mb() == X->nb());

    for (int64_t j = 0; j < X->nb(); ++j) {
        for (int64_t i = 0; i < j; ++i) { // upper
            std::swap(X->at(i, j), X->at(j, i));
        }
    }

    X->layout(X->layout() == Layout::RowMajor ? Layout::ColMajor
                                              : Layout::RowMajor);
}

//------------------------------------------------------------------------------
/// @deprecated
/// In-place conversion between column and row-major layout for square tiles.
/// Takes a pointer to the original tile in MatrixStorage, instead of a
/// reference to a copy of the tile, in order to adjust the tile's layout flag.
/// @ingroup convert_tile
///
template <typename scalar_t>
void convert_layout(Tile<scalar_t>* X, cudaStream_t stream)
{
    trace::Block trace_block("slate::device::transpose");
    assert(X->mb() == X->nb());

    device::transpose(X->mb(), X->data(), X->stride(), stream);
    slate_cuda_call(
        cudaStreamSynchronize(stream));

    X->layout(X->layout() == Layout::RowMajor ? Layout::ColMajor
                                              : Layout::RowMajor);
}

//------------------------------------------------------------------------------
/// Transpose a square data in-place.
/// Host implementation
///
/// @param[in] n
///     Number of rows and columns of matrix.
///
/// @param[in,out] A
///     Buffer holding input data.
///     On output: holds the transposed data.
///
/// @param[in] lda
///     Leading dimension of matrix A.
///
template <typename scalar_t>
void transpose( int64_t n,
                scalar_t* A, int64_t lda)
{
    // square in-place transpose
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < j; ++i) { // upper
            std::swap(A[i + j*lda], A[j + i*lda]);
        }
    }
}

//------------------------------------------------------------------------------
/// Transpose a rectangular data out-of-place.
/// Host implementation
///
/// @param[in] m
///     Number of rows.
///
/// @param[in] n
///     Number of columns.
///
/// @param[in] A
///     Buffer holding input data.
///
/// @param[in] lda
///     Leading dimension of matrix A.
///
/// @param[out] AT
///     On output: holds the transposed data.
///
/// @param[in] ldat
///     Leading dimension of matrix AT.
///
template <typename scalar_t>
void transpose( int64_t m, int64_t n,
                scalar_t* A, int64_t lda,
                scalar_t* AT, int64_t ldat)
{
    // rectangular out-of-place transpose
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            AT[j + i*ldat] = A[i + j*lda];
        }
    }
}

} // namespace slate

#endif // SLATE_TILE_AUX_HH