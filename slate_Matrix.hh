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

#ifndef SLATE_MATRIX_HH
#define SLATE_MATRIX_HH

#include "slate_BaseMatrix.hh"
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
/// General non-symmetric, m-by-n, distributed, tiled matrices.
template <typename scalar_t>
class Matrix: public BaseMatrix<scalar_t> {
public:
    // constructors
    Matrix();

    static
    Matrix fromLAPACK(int64_t m, int64_t n,
                      scalar_t* A, int64_t lda, int64_t nb,
                      int p, int q, MPI_Comm mpi_comm);

    static
    Matrix fromScaLAPACK(int64_t m, int64_t n,
                         scalar_t* A, int64_t lda, int64_t nb,
                         int p, int q, MPI_Comm mpi_comm);

    // conversion sub-matrix
    Matrix(BaseMatrix<scalar_t>& orig,
           int64_t i1, int64_t i2,
           int64_t j1, int64_t j2);

    // sub-matrix
    Matrix sub(int64_t i1, int64_t i2,
               int64_t j1, int64_t j2);

protected:
    // used by fromLAPACK
    Matrix(int64_t m, int64_t n,
           scalar_t* A, int64_t lda, int64_t nb,
           int p, int q, MPI_Comm mpi_comm);

    // used by fromScaLAPACK
    Matrix(int64_t m, int64_t n,
           scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
           int p, int q, MPI_Comm mpi_comm);

    // used by sub
    Matrix(Matrix& orig,
           int64_t i1, int64_t i2, int64_t j1, int64_t j2);

public:
    template <typename T>
    friend void swap(Matrix<T>& A, Matrix<T>& B);

    int64_t getMaxHostTiles();
    int64_t getMaxDeviceTiles(int device);
    void allocateBatchArrays();
    void reserveHostWorkspace();
    void reserveDeviceWorkspace();
    void gather(scalar_t* A, int64_t lda);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
Matrix<scalar_t>::Matrix():
    BaseMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from LAPACK layout.
/// Construct matrix by wrapping existing memory of an m-by-n LAPACK matrix.
/// The caller must ensure that the memory remains valid for the lifetime
/// of the Matrix object and any shallow copies of it.
/// Input format is an LAPACK-style column-major matrix with leading
/// dimension (column stride) lda >= m, that is replicated across all nodes.
/// Matrix gets tiled with square nb-by-nb tiles.
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
Matrix<scalar_t> Matrix<scalar_t>::fromLAPACK(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    return Matrix<scalar_t>(m, n, A, lda, nb, p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from ScaLAPACK layout.
/// Construct matrix by wrapping existing memory of an m-by-n ScaLAPACK matrix.
/// The caller must ensure that the memory remains valid for the lifetime
/// of the Matrix object and any shallow copies of it.
/// Input format is a ScaLAPACK-style 2D block-cyclic column-major matrix
/// with local leading dimension (column stride) lda,
/// p block rows and q block columns.
/// Matrix gets tiled with square nb-by-nb tiles.
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
Matrix<scalar_t> Matrix<scalar_t>::fromScaLAPACK(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    // note extra nb
    return Matrix<scalar_t>(m, n, A, lda, nb, nb, p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// [internal]
/// @see fromLAPACK
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);
        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (this->tileIsLocal(i, j)) {
                this->tileInsert(i, j, this->host_num_, &A[ ii + jj*lda ], lda);
            }
            ii += ib;
        }
        jj += jb;
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// @see fromScaLAPACK
/// This differs from LAPACK constructor by adding mb.
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    assert(mb == nb);
    // ii, jj are row, col indices
    // ii_loc and jj_loc are the local array indices in a
    // block-cyclic layout (indxg2l)
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);
        // Using Scalapack indxg2l
        int64_t jj_loc = nb*(jj/(nb*q)) + (jj % nb);
        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (this->tileIsLocal(i, j)) {
                // Using Scalapack indxg2l
                int64_t ii_loc = mb*(ii/(mb*p)) + (ii % mb);
                this->tileInsert(i, j, this->host_num_,
                                 &A[ ii_loc + jj_loc*lda ], lda);
            }
            ii += ib;
        }
        jj += jb;
    }
}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
template <typename scalar_t>
Matrix<scalar_t> Matrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    return Matrix<scalar_t>(*this, i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// [internal]
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, j1:j2 ].
/// @see sub().
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    Matrix& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, j1:j2 ].
/// This version is called for conversion from off-diagonal submatrix of
/// TriangularMatrix, SymmetricMatrix, HermitianMatrix, etc.
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    BaseMatrix< scalar_t >& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{}

//------------------------------------------------------------------------------
/// Swap contents of matrices A and B.
//
// (This isn't really needed over BaseMatrix swap, but is here as a reminder
// in case any members are added to Matrix that aren't in BaseMatrix.)
template <typename scalar_t>
void swap(Matrix<scalar_t>& A, Matrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseMatrix<scalar_t>& >(A),
         static_cast< BaseMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank.
//
// todo: numLocalTiles? use for life as well?
template <typename scalar_t>
int64_t Matrix<scalar_t>::getMaxHostTiles()
{
    int64_t num_tiles = 0;
    for (int64_t j = 0; j < this->nt(); ++j)
        for (int64_t i = 0; i < this->mt(); ++i)
            if (this->tileIsLocal(i, j))
                ++num_tiles;

    return num_tiles;
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank and given device.
//
// todo: numLocalDeviceTiles?
template <typename scalar_t>
int64_t Matrix<scalar_t>::getMaxDeviceTiles(int device)
{
    int64_t num_tiles = 0;
    for (int64_t j = 0; j < this->nt(); ++j)
        for (int64_t i = 0; i < this->mt(); ++i)
            if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                ++num_tiles;

    return num_tiles;
}

//------------------------------------------------------------------------------
/// Allocates batch arrays for all devices.
template <typename scalar_t>
void Matrix<scalar_t>::allocateBatchArrays()
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
void Matrix<scalar_t>::reserveHostWorkspace()
{
    this->storage_->reserveHostWorkspace(getMaxHostTiles());
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on all GPU devices.
template <typename scalar_t>
void Matrix<scalar_t>::reserveDeviceWorkspace()
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
void Matrix<scalar_t>::gather(scalar_t* A, int64_t lda)
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
            ii += ib;
        }
        jj += jb;
    }

    this->op_ = op_save;
}

} // namespace slate

#endif // SLATE_MATRIX_HH
