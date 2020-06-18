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

#ifndef SLATE_BAND_MATRIX_HH
#define SLATE_BAND_MATRIX_HH

#include "slate/BaseBandMatrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"
#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// General banded, non-symmetric, m-by-n, distributed, tiled matrices.
template <typename scalar_t>
class BandMatrix: public BaseBandMatrix<scalar_t> {
public:
    // constructors
    BandMatrix();

    BandMatrix(int64_t m, int64_t n, int64_t kl, int64_t ku,
               int64_t nb, int p, int q, MPI_Comm mpi_comm);

    BandMatrix(int64_t kl, int64_t ku, Matrix<scalar_t>& orig);

    BandMatrix<scalar_t> slice(
        int64_t row1, int64_t row2,
        int64_t col1, int64_t col2);

    template <typename out_scalar_t=scalar_t>
    static
    BandMatrix<out_scalar_t> emptyLike(BaseMatrix<scalar_t>& orig, int64_t kl,
                                       int64_t ku, int64_t mb=0, int64_t nb=0,
                                       Op deepOp=Op::NoTrans);

protected:
    // used by slice
    BandMatrix(BaseBandMatrix<scalar_t>& orig,
           typename BaseMatrix<scalar_t>::Slice slice);

public:
    template <typename T>
    friend void swap(BandMatrix<T>& A, BandMatrix<T>& B);

    int64_t lowerBandwidth() const;
    void    lowerBandwidth(int64_t kl);

    int64_t upperBandwidth() const;
    void    upperBandwidth(int64_t ku);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
BandMatrix<scalar_t>::BandMatrix():
    BaseBandMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n band matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in] kl
///     Number of subdiagonals within band. kl >= 0.
///
/// @param[in] ku
///     Number of superdiagonals within band. ku >= 0.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
BandMatrix<scalar_t>::BandMatrix(
    int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseBandMatrix<scalar_t>(m, n, kl, ku, nb, p, q, mpi_comm)
{}

//------------------------------------------------------------------------------
/// Conversion from general Matrix.
/// Creates a shallow copy view of the band region [kl, ku] of the original matrix.
///
/// @param[in] kl
///     Lower bandwidth.
///
/// @param[in] ku
///     Upper bandwidth.
///
/// @param[in] orig
///     Original matrix.
///
template <typename scalar_t>
BandMatrix<scalar_t>::BandMatrix(
    int64_t kl, int64_t ku, Matrix<scalar_t>& orig)
    : BaseBandMatrix<scalar_t>(kl, ku, orig)
{}

//------------------------------------------------------------------------------
/// Sliced matrix constructor creates shallow copy view of parent matrix,
/// A[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
///
/// @param[in] orig
///     Original matrix of which to make sub-matrix.
///
/// @param[in] slice
///     Contains start and end row and column indices.
///
template <typename scalar_t>
BandMatrix<scalar_t>::BandMatrix(
    BaseBandMatrix<scalar_t>& orig, typename BaseMatrix<scalar_t>::Slice slice)
    : BaseBandMatrix<scalar_t>(orig, slice)
{
    this->uplo_ = Uplo::General;
}

//------------------------------------------------------------------------------
/// Returns sliced matrix that is a shallow copy view of the
/// parent matrix, A[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
///
/// @param[in] row1
///     Starting row index. 0 <= row1 < m.
///
/// @param[in] row2
///     Ending row index (inclusive). row2 < m.
///
/// @param[in] col1
///     Starting column index. 0 <= col1 < n.
///
/// @param[in] col2
///     Ending column index (inclusive). col2 < n.
///
template <typename scalar_t>
BandMatrix<scalar_t> BandMatrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    return BandMatrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

//------------------------------------------------------------------------------
/// Named constructor returns a new, empty Matrix with the same structure
/// (size and distribution) as the matrix orig. Tiles are not allocated.
///
/// @param[in] orig
///     Original matrix of which to make an empty matrix with the same structure
///     (size and distribution) as this original matrix.
///
/// @param[in] kl
///     Number of subdiagonals within band. kl >= 0.
///
/// @param[in] ku
///     Number of superdiagonals within band. ku >= 0.
///
/// @param[in] mb
///     Row block size of new matrix.
///     If mb = 0, uses the same mb and m as this matrix;
///     otherwise, m = mb * mt.
///
/// @param[in] nb
///     Column block size of new matrix.
///     If nb = 0, uses the same nb and n as this matrix;
///     otherwise, n = nb * nt.
///
/// @param[in] deepOp
///     Additional deep-transposition operation to apply. If deepOp=Trans, the
///     new matrix has the transposed structure (distribution and number of
///     tiles) of this matrix, but its shallow-transpose op() flag is set to
///     NoTrans. For a 1x4 matrix A, compare:
///     - transpose(A).emptyLike() creates a new 1x4 matrix, then transposes it
///       to return a 4x1 matrix with its op set to Trans.
///     - A.emptyLike(mb, nb, Op::Trans) creates and returns a new 4x1 matrix
///       with its op set to NoTrans.
///
template <typename scalar_t>
template <typename out_scalar_t>
BandMatrix<out_scalar_t> BandMatrix<scalar_t>::emptyLike(
    BaseMatrix<scalar_t>& orig, int64_t kl, int64_t ku, int64_t mb, int64_t nb,
    Op deepOp)
{
    auto B = orig.template baseEmptyLike<out_scalar_t>(mb, nb, deepOp);
    auto M = Matrix<out_scalar_t>(B, 0, B.mt()-1, 0, B.nt()-1);
    return BandMatrix<out_scalar_t>(kl, ku, M);
}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(BandMatrix<scalar_t>& A, BandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseBandMatrix<scalar_t>& >(A),
         static_cast< BaseBandMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// @return number of subdiagonals within band.
template <typename scalar_t>
int64_t BandMatrix<scalar_t>::lowerBandwidth() const
{
    return (this->op() == Op::NoTrans ? this->kl_ : this->ku_);
}

//------------------------------------------------------------------------------
/// Sets number of subdiagonals within band.
template <typename scalar_t>
void BandMatrix<scalar_t>::lowerBandwidth(int64_t kl)
{
    if (this->op() == Op::NoTrans)
        this->kl_ = kl;
    else
        this->ku_ = kl;
}

//------------------------------------------------------------------------------
/// @return number of superdiagonals within band.
template <typename scalar_t>
int64_t BandMatrix<scalar_t>::upperBandwidth() const
{
    return (this->op() == Op::NoTrans ? this->ku_ : this->kl_);
}

//------------------------------------------------------------------------------
/// Sets number of superdiagonals within band.
template <typename scalar_t>
void BandMatrix<scalar_t>::upperBandwidth(int64_t ku)
{
    if (this->op() == Op::NoTrans)
        this->ku_ = ku;
    else
        this->kl_ = ku;
}

} // namespace slate

#endif // SLATE_BAND_MATRIX_HH
