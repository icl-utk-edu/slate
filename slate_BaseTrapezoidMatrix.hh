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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#ifndef SLATE_BASE_TRAPEZOID_MATRIX_HH
#define SLATE_BASE_TRAPEZOID_MATRIX_HH

#include "slate_BaseMatrix.hh"
#include "slate_Matrix.hh"
#include "slate_Tile.hh"
#include "slate_types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate_cuda.hh"
#include "slate_cublas.hh"
#include "slate_mpi.hh"
#include "slate_openmp.hh"

namespace slate {

//==============================================================================
/// Base class for all SLATE distributed, tiled trapezoidal storage matrices.
/// Either the upper or lower trapezoid is stored, with the opposite triangle
/// assumed by symmetry (SymmetricMatrix, HermitianMatrix),
/// or assumed to be zero (TrapezoidMatrix, TriangularMatrix).
template <typename scalar_t>
class BaseTrapezoidMatrix: public BaseMatrix<scalar_t> {
protected:
    // constructors
    BaseTrapezoidMatrix();

    // conversion
    BaseTrapezoidMatrix(Uplo uplo, Matrix< scalar_t >& orig);

    // sub-matrix
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2);

    // used by sub-classes' fromLAPACK
    BaseTrapezoidMatrix(Uplo in_uplo, int64_t m, int64_t n,
                        scalar_t* A, int64_t lda, int64_t nb,
                        int p, int q, MPI_Comm mpi_comm);

    // used by sub-classes' fromScaLAPACK
    BaseTrapezoidMatrix(Uplo in_uplo, int64_t m, int64_t n,
                        scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                        int p, int q, MPI_Comm mpi_comm);

    // used by sub-classes' off-diagonal sub
    BaseTrapezoidMatrix(Uplo uplo, Matrix<scalar_t>& orig,
                        int64_t i1, int64_t i2,
                        int64_t j1, int64_t j2);

    // used by sub
    BaseTrapezoidMatrix(BaseTrapezoidMatrix& orig,
                        int64_t i1, int64_t i2,
                        int64_t j1, int64_t j2);

public:
    template <typename T>
    friend void swap(BaseTrapezoidMatrix<T>& A, BaseTrapezoidMatrix<T>& B);

    int64_t getMaxHostTiles();
    int64_t getMaxDeviceTiles(int device);
    void allocateBatchArrays();
    void reserveHostWorkspace();
    void reserveDeviceWorkspace();
    void gather(scalar_t* A, int64_t lda);
    Uplo uplo() const;
    Uplo uplo_logical() const;
};

///-------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix()
    : BaseMatrix<scalar_t>()
{
    this->uplo_ = Uplo::Lower;
}

///-------------------------------------------------------------------------
/// Construct matrix by wrapping existing memory of an m-by-n lower
/// or upper trapezoidal storage LAPACK matrix. Triangular, symmetric, and
/// Hermitian matrices all use this storage scheme (with m = n).
/// The caller must ensure that the memory remains valid for the lifetime
/// of the Matrix object and any shallow copies of it.
/// Input format is an LAPACK-style column-major matrix with leading
/// dimension (column stride) lda >= m, that is replicated across all nodes.
/// Matrix gets tiled with square nb-by-nb tiles.
///
/// @param[in] in_uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of the array A. lda >= m.
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
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo in_uplo, int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    this->uplo_ = in_uplo;

    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    if (uplo() == Uplo::Lower) {
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);

            int64_t ii = j*nb;
            for (int64_t i = j; i < this->mt(); ++i) {  // lower
                int64_t ib = this->tileMb(i);

                if (this->tileIsLocal(i, j)) {
                    this->tileInsert(i, j, this->host_num_,
                                     &A[ii + jj*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
    else {  // Upper
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);

            int64_t ii = 0;
            for (int64_t i = 0; i <= j && i < this->mt(); ++i) {  // upper
                int64_t ib = this->tileMb(i);

                if (this->tileIsLocal(i, j)) {
                    this->tileInsert(i, j, this->host_num_,
                                     &A[ii + jj*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
}

///-------------------------------------------------------------------------
/// Construct matrix by wrapping existing memory of an m-by-n lower
/// or upper trapezoidal storage ScaLAPACK matrix. Triangular, symmetric, and
/// Hermitian matrices all use this storage scheme (with m = n).
/// The caller must ensure that the memory remains valid for the lifetime
/// of the Matrix object and any shallow copies of it.
/// Input format is a ScaLAPACK-style 2D block-cyclic column-major matrix
/// with local leading dimension (column stride) lda,
/// p block rows and q block columns.
/// Matrix gets tiled with square nb-by-nb tiles.
/// This differs from LAPACK constructor by adding mb.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The local portion of the 2D block cyclic distribution of
///     the m-by-n matrix A, with local leading dimension lda.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] mb
///     Same as nb; used to differentiate from LAPACK constructor.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
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
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo in_uplo, int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    assert(mb == nb);
    this->uplo_ = in_uplo;

    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    if (uplo() == Uplo::Lower) {
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);
            // Using Scalapack indxg2l
            int64_t jj_loc = nb*(jj/(nb*q)) + (jj % nb);

            int64_t ii = j*nb;
            for (int64_t i = j; i < this->mt(); ++i) {  // lower
                int64_t ib = this->tileMb(i);
                // Using Scalapack indxg2l
                int64_t ii_loc = mb*(ii/(mb*p)) + (ii % mb);

                if (this->tileIsLocal(i, j)) {
                    this->tileInsert(i, j, this->host_num_,
                                     &A[ii_loc + jj_loc*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
    else {  // Upper
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);
            // Using Scalapack indxg2l
            int64_t jj_loc = nb*(jj/(nb*q)) + (jj % nb);

            int64_t ii = 0;
            for (int64_t i = 0; i <= j && i < this->mt(); ++i) {  // upper
                int64_t ib = this->tileMb(i);
                // Using Scalapack indxg2l
                int64_t ii_loc = mb*(ii/(mb*p)) + (ii % mb);

                if (this->tileIsLocal(i, j)) {
                    this->tileInsert(i, j, this->host_num_,
                                     &A[ii_loc + jj_loc*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
}

///-------------------------------------------------------------------------
/// Conversion from general matrix
/// creates shallow copy view of original matrix.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] orig
///     Original matrix.
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo uplo, Matrix<scalar_t>& orig)
    : BaseMatrix<scalar_t>(orig)
{
    this->uplo_ = uplo;
}

///-------------------------------------------------------------------------
/// Conversion from general matrix, sub-matrix constructor
/// creates shallow copy view of original matrix, A[ i1:i2, j1:j2 ].
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo uplo, Matrix<scalar_t>& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{
    this->uplo_ = uplo;
}

///-------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, j1:j2 ]. Requires i1 == j1. The new view is still a trapezoid
/// matrix, with the same diagonal as the parent matrix.
///
/// @param[in] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    BaseTrapezoidMatrix& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{
    this->uplo_ = orig.uplo_;
    if (i1 != j1) {
        throw std::exception();
    }
}

///-------------------------------------------------------------------------
/// Swap contents of matrices A and B.
template <typename scalar_t>
void swap(BaseTrapezoidMatrix<scalar_t>& A, BaseTrapezoidMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseMatrix<scalar_t>& >(A),
         static_cast< BaseMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank.
// todo: numLocalTiles? use for life as well?
template <typename scalar_t>
int64_t BaseTrapezoidMatrix<scalar_t>::getMaxHostTiles()
{
    int64_t num_tiles = 0;
    if (this->uplo_logical() == Uplo::Lower) {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = j; i < this->mt(); ++i)  // lower
                if (this->tileIsLocal(i, j))
                    ++num_tiles;
    }
    else {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = 0; i <= j && j < this->mt(); ++i)  // upper
                if (this->tileIsLocal(i, j))
                    ++num_tiles;
    }

    return num_tiles;
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank and given device.
// todo: numLocalDeviceTiles
template <typename scalar_t>
int64_t BaseTrapezoidMatrix<scalar_t>::getMaxDeviceTiles(int device)
{
    int64_t num_tiles = 0;
    if (this->uplo_logical() == Uplo::Lower) {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = j; i < this->mt(); ++i)  // lower
                if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                    ++num_tiles;
    }
    else {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = 0; i <= j && j < this->mt(); ++i)  // upper
                if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                    ++num_tiles;
    }
    return num_tiles;
}

//------------------------------------------------------------------------------
/// Allocates batch arrays for all devices.
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::allocateBatchArrays()
{
    int64_t num_tiles = 0;
    for (int device = 0; device < this->num_devices_; ++device) {
        num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
    }
    this->storage_->allocateBatchArrays(num_tiles);
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on host.
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::reserveHostWorkspace()
{
    this->storage_->reserveHostWorkspace(getMaxHostTiles());
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on all GPU devices.
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::reserveDeviceWorkspace()
{
    int64_t num_tiles = 0;
    for (int device = 0; device < this->num_devices_; ++device) {
        num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
    }
    this->storage_->reserveDeviceWorkspace(num_tiles);
}

//------------------------------------------------------------------------------
/// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
/// Primarily for debugging purposes.
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::gather(scalar_t* A, int64_t lda)
{
    // this code assumes the matrix is not transposed
    Op op_save = this->op();
    this->op_ = Op::NoTrans;

    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);

        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);

            if ((uplo() == Uplo::Lower && i >= j) ||
                (uplo() == Uplo::Upper && i <= j)) {
                if (this->mpi_rank_ == 0) {
                    if (! this->tileIsLocal(i, j)) {
                        // erase any existing non-local tile and insert new one
                        this->tileErase(i, j, this->host_num_);
                        this->tileInsert(i, j, this->host_num_,
                                         &A[(size_t)lda*jj + ii], lda);
                        auto Aij = this->at(i, j);
                        Aij.recv(this->tileRank(i, j), this->mpi_comm_);
                    }
                    else {
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
/// Returns off-diagonal sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
/// This version returns a general Matrix, which:
/// - if uplo = Lower, is strictly below the diagonal, or
/// - if uplo = Upper, is strictly above the diagonal.
template <typename scalar_t>
Matrix<scalar_t> BaseTrapezoidMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2, int64_t j1, int64_t j2)
{
    if (this->uplo_logical() == Uplo::Lower) {
        // top-right corner is at or below diagonal
        assert(i1 >= j2);
    }
    else {
        // bottom-left corner is at or above diagonal
        assert(i2 <= j1);
    }
    return Matrix< scalar_t >(*this, i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Returns whether A is physically Lower or Upper storage,
///         ignoring the transposition operation.
/// @see uplo_logical()
template <typename scalar_t>
Uplo BaseTrapezoidMatrix<scalar_t>::uplo() const
{
    return this->uplo_;
}

//------------------------------------------------------------------------------
/// Returns whether op(A) is logically Lower or Upper storage,
///         taking the transposition operation into account.
/// @see uplo()
template <typename scalar_t>
Uplo BaseTrapezoidMatrix<scalar_t>::uplo_logical() const
{
    if ((this->uplo() == Uplo::Lower && this->op() == Op::NoTrans) ||
        (this->uplo() == Uplo::Upper && this->op() != Op::NoTrans))
    {
        return Uplo::Lower;
    }
    else {
        return Uplo::Upper;
    }
}

} // namespace slate

#endif // SLATE_BASE_TRAPEZOID_MATRIX_HH
