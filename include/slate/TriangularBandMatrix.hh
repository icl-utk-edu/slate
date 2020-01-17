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

#ifndef SLATE_TRIANGULAR_BAND_MATRIX_HH
#define SLATE_TRIANGULAR_BAND_MATRIX_HH

#include "slate/BaseTriangularBandMatrix.hh"
#include "slate/BandMatrix.hh"
#include "slate/TriangularMatrix.hh"
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
/// Triangular banded, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class TriangularBandMatrix: public BaseTriangularBandMatrix<scalar_t> {
public:
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // constructors
    TriangularBandMatrix();

    TriangularBandMatrix(Uplo uplo, Diag diag, int64_t n, int64_t kd,
                         std::function<int64_t (int64_t j)>& inTileNb,
                         std::function<int (ij_tuple ij)>& inTileRank,
                         std::function<int (ij_tuple ij)>& inTileDevice,
                         MPI_Comm mpi_comm);

    TriangularBandMatrix(
        Uplo uplo, Diag diag,
        int64_t n, int64_t kd,
        int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    TriangularBandMatrix(Uplo uplo, Diag diag, BandMatrix<scalar_t>& orig);
    TriangularBandMatrix(Diag diag, BaseTriangularBandMatrix<scalar_t>& orig);

    TriangularMatrix<scalar_t> sub(int64_t i1, int64_t i2);

    // sub-matrix
    Matrix<scalar_t> sub(int64_t i1, int64_t i2,
                         int64_t j1, int64_t j2);
    Matrix<scalar_t> slice(int64_t row1, int64_t row2,
                           int64_t col1, int64_t col2);

public:
    template <typename T>
    friend void swap(TriangularBandMatrix<T>& A, TriangularBandMatrix<T>& B);

    void    gather(scalar_t* A, int64_t lda);
    void    gatherAll(std::set<int>& rank_set, int tag = 0, int64_t life_factor = 1);
    void    ge2tbGather(Matrix<scalar_t>& A);

    Diag diag() { return diag_; }
    void diag(Diag in_diag) { diag_ = in_diag; }

protected:
    Diag diag_;
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix()
    : BaseTriangularBandMatrix<scalar_t>(),
      diag_(Diag::NonUnit)
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n matrix, with no tiles allocated,
/// where tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
///
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Uplo uplo, Diag diag, int64_t n, int64_t kd,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : BaseTriangularBandMatrix<scalar_t>(uplo, n, kd, inTileNb, inTileRank,
                                inTileDevice, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n band matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in] kd
///     Number of sub (if lower) or super (if upper) diagonals within band.
///     kd >= 0.
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
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Uplo uplo, Diag diag,
    int64_t n, int64_t kd, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTriangularBandMatrix<scalar_t>(uplo, n, kd, nb, p, q, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// todo:
/// Conversion from general band matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Uplo uplo, Diag diag, BandMatrix<scalar_t>& orig)
    : BaseTriangularBandMatrix<scalar_t>(uplo, orig),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// todo:
/// Conversion from base triangular band matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Diag diag, BaseTriangularBandMatrix<scalar_t>& orig)
    : BaseTriangularBandMatrix<scalar_t>(orig),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, i1:i2 ].
/// This version returns a TriangularMatrix with the same diagonal as the
/// parent matrix.
/// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
///                                  int64_t j1, int64_t j2)
///
/// @param[in] i1
///     Starting block row and column index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row and column index (inclusive). i2 < mt.
///
template <typename scalar_t>
TriangularMatrix<scalar_t> TriangularBandMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2)
{
    return TriangularMatrix<scalar_t>(*this, i1, i2);
}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
/// This version returns a Matrix.
///
/// @param[in] i1
///     Starting block-row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block-row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block-column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block-column index (inclusive). j2 < nt.
///
template <typename scalar_t>
Matrix<scalar_t> TriangularBandMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    // todo: assert that sub-matrix falls within upper/lower band
    return Matrix<scalar_t>(*this, i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Returns sliced matrix that is a shallow copy view of the
/// parent matrix, A[ row1:row2, col1:col2 ].
/// This takes row & col indices instead of block row & block col indices.
/// The sub-matrix cannot overlap the diagonal.
/// - if uplo = Lower, 0 <= col1 <= col2 <= row1 <= row2 < n;
/// - if uplo = Upper, 0 <= row1 <= row2 <= col1 <= col2 < n.
///
/// @param[in] row1
///     Starting row index.
///
/// @param[in] row2
///     Ending row index (inclusive).
///
/// @param[in] col1
///     Starting column index.
///
/// @param[in] col2
///     Ending column index (inclusive).
///
template <typename scalar_t>
Matrix<scalar_t> TriangularBandMatrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    // todo: assert that sub-matrix falls within upper/lower band
    return Matrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(TriangularBandMatrix<scalar_t>& A, TriangularBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseTriangularBandMatrix<scalar_t>& >(A),
         static_cast< BaseTriangularBandMatrix<scalar_t>& >(B));
    swap(A.diag_, B.diag_);
}

//------------------------------------------------------------------------------
/// Gather all tiles on each rank
// WARNING: this is a demanding process storage and communication wise,
// avoid if possible.
//
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::gatherAll(std::set<int>& rank_set, int tag, int64_t life_factor)
{
    trace::Block trace_block("slate::gatherAll");

    auto upper = this->uplo() == Uplo::Upper;
    auto layout = this->layout(); // todo: is this correct?

    // If this rank is not in the set.
    if (rank_set.find(this->mpiRank()) == rank_set.end())
        return;

    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
    for (int64_t j = 0; j < nt; ++j) {
        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = istart; i <= iend; ++i) {

            // If receiving the tile.
            if (! this->tileIsLocal(i, j)) {
                // Create tile to receive data, with life span.
                // If tile already exists, add to its life span.
                LockGuard guard(this->storage_->getTilesMapLock()); // todo: accessor
                auto iter = this->storage_->find(this->globalIndex(i, j, this->hostNum()));

                int64_t life = life_factor;
                if (iter == this->storage_->end())
                    this->tileInsertWorkspace(i, j, this->hostNum());
                else
                    life += this->tileLife(i, j); // todo: use temp tile to receive
                this->tileLife(i, j, life);
            }

            // Send across MPI ranks.
            // Previous used MPI bcast: tileBcastToSet(i, j, rank_set);
            // Currently uses 2D hypercube p2p send.
            this->tileBcastToSet(i, j, rank_set, 2, tag, layout);
        }
    }
}

//------------------------------------------------------------------------------
/// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
/// Primarily for debugging purposes.
///
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::gather(scalar_t* A, int64_t lda)
{
    // this code assumes the matrix is not transposed
    Op op_save = this->op();
    this->op_ = Op::NoTrans;
    auto upper = this->uplo() == Uplo::Upper;

    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < nt; ++j) {
        int64_t jb = this->tileNb(j);

        int64_t ii = 0;
        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (i >= istart && i <= iend) {
                if (this->mpi_rank_ == 0) {
                    if (! this->tileIsLocal(i, j)) {
                        // erase any existing non-local tile and insert new one
                        this->tileErase(i, j, this->host_num_);
                        this->tileInsert(i, j, this->host_num_,
                                         &A[(size_t)lda*jj + ii], lda);
                        auto Aij = this->at(i, j);
                        Aij.recv(this->tileRank(i, j), this->mpi_comm_, this->layout());
                        this->tileLayout(i, j, this->layout_);
                    }
                    else {
                        this->tileGetForReading(i, j, LayoutConvert(this->layout()));
                        // copy local tiles if needed.
                        auto Aij = this->at(i, j);
                        if (Aij.data() != &A[(size_t)lda*jj + ii]) {
                            lapack::lacpy(lapack::MatrixType::General, ib, jb,
                                          Aij.data(), Aij.stride(),
                                          &A[(size_t)lda*jj + ii], lda);
                        }
                    }
                }
                else if (this->tileIsLocal(i, j)) {
                    this->tileGetForReading(i, j, LayoutConvert(this->layout()));
                    auto Aij = this->at(i, j);
                    Aij.send(0, this->mpi_comm_);
                }
            }
            ii += ib;
        }
        jj += jb;
    }

    this->op_ = op_save;
}

//------------------------------------------------------------------------------
/// Gather the distributed triangular band portion of a general Matrix A
/// to TriangularBandMatrix B on MPI rank 0.
/// Primarily for SVD code
///
// todo: parameter for rank to collect on, default 0
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::ge2tbGather(Matrix<scalar_t>& A)
{
    Op op_save = this->op();
    this->op_ = Op::NoTrans;
    auto upper = this->uplo() == Uplo::Upper;

    int64_t mt = A.mt();
    int64_t nt = A.nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
    // i, j are tile (block row, block col) indices
    for (int64_t j = 0; j < nt; ++j) {

        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = 0; i < mt; ++i) {
            if (i >= istart && i <= iend) {
                if (this->mpi_rank_ == 0) {
                    if (! A.tileIsLocal(i, j)) {
                        this->tileInsert(i, j, this->host_num_);
                        auto Bij = this->at(i, j);
                        Bij.recv(A.tileRank(i, j), this->mpi_comm_, this->layout());
                    }
                    else {
                        A.tileGetForReading(i, j, LayoutConvert(this->layout()));
                        // copy local tiles if needed.
                        auto Aij = A(i, j);
                        auto Bij = this->at(i, j);
                        if (Aij.data() != Bij.data() ) {
                            gecopy(A(i, j), Bij );
                        }
                    }
                }
                else if (A.tileIsLocal(i, j)) {
                    A.tileGetForReading(i, j, LayoutConvert(this->layout()));
                    auto Aij = A(i, j);
                    Aij.send(0, this->mpi_comm_);
                }
            }
        }
    }

    this->op_ = op_save;
}


} // namespace slate

#endif // SLATE_TRIANGULAR_BAND_MATRIX_HH
