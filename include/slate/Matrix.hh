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

#ifndef SLATE_MATRIX_HH
#define SLATE_MATRIX_HH

#include "slate/internal/BaseMatrix.hh"
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
/// General non-symmetric, m-by-n, distributed, tiled matrices.
template <typename scalar_t>
class Matrix: public BaseMatrix<scalar_t> {
public:
    // constructors
    Matrix();

    Matrix(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

    static
    Matrix fromLAPACK(int64_t m, int64_t n,
                      scalar_t* A, int64_t lda, int64_t nb,
                      int p, int q, MPI_Comm mpi_comm);

    static
    Matrix fromScaLAPACK(int64_t m, int64_t n,
                         scalar_t* A, int64_t lda, int64_t nb,
                         int p, int q, MPI_Comm mpi_comm);

    static
    Matrix fromDevices(int64_t m, int64_t n,
                       scalar_t** Aarray, int num_devices, int64_t lda,
                       int64_t nb, int p, int q, MPI_Comm mpi_comm);

    template <typename out_scalar_t=scalar_t>
    Matrix<out_scalar_t> emptyLike();

    // conversion sub-matrix
    Matrix(BaseMatrix<scalar_t>& orig,
           int64_t i1, int64_t i2,
           int64_t j1, int64_t j2);

    // sub-matrix
    Matrix sub(int64_t i1, int64_t i2,
               int64_t j1, int64_t j2);

    // sliced matrix
    Matrix slice(int64_t row1, int64_t row2,
                 int64_t col1, int64_t col2);

protected:
    // used by fromLAPACK
    Matrix(int64_t m, int64_t n,
           scalar_t* A, int64_t lda, int64_t nb,
           int p, int q, MPI_Comm mpi_comm);

    // used by fromScaLAPACK
    Matrix(int64_t m, int64_t n,
           scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
           int p, int q, MPI_Comm mpi_comm);

    // used by fromDevices
    Matrix(int64_t m, int64_t n,
           scalar_t** Aarray, int num_devices, int64_t lda, int64_t nb,
           int p, int q, MPI_Comm mpi_comm);

    // used by slice
    Matrix(BaseMatrix<scalar_t>& orig,
           typename BaseMatrix<scalar_t>::Slice slice);

public:
    template <typename T>
    friend void swap(Matrix<T>& A, Matrix<T>& B);

    int64_t getMaxHostTiles();
    int64_t getMaxDeviceTiles(int device);
    void allocateBatchArrays();
    void reserveHostWorkspace();
    void reserveDeviceWorkspace();
    void gather(scalar_t* A, int64_t lda);
    void insertLocalTiles(Target origin=Target::Host);
    void insertLocalTiles(bool on_devices);

    // copy local data of op(A).
    void copy(Matrix& A);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
Matrix<scalar_t>::Matrix():
    BaseMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n matrix, with no tiles allocated.
/// Tiles can be added with tileInsert().
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
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
//
// todo: have allocate flag? If true, allocate data; else user will insert tiles?
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
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
///
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
///
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
/// [static]
/// Named constructor returns a new Matrix from data in GPU device memory,
/// distributed 2D block-cyclic across MPI ranks,
/// and 1D block-cyclic across GPU devices within an MPI rank.
///
/// Aarray contains pointers to data on each GPU device in this MPI rank.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] Aarray
///     Array of
///     The local portion of the 2D block cyclic distribution of
///     the m-by-n matrix A, with local leading dimension lda.
///
/// @param[in] num_devices
///     Dimension of Aarray.
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
///
template <typename scalar_t>
Matrix<scalar_t> Matrix<scalar_t>::fromDevices(
    int64_t m, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda,
    int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    return Matrix<scalar_t>(m, n, Aarray, num_devices, lda, nb, p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// Named constructor returns a new, empty Matrix with the same structure
/// (size and distribution) as this matrix. Tiles are not allocated.
///
template <typename scalar_t>
template <typename out_scalar_t>
Matrix<out_scalar_t> Matrix<scalar_t>::emptyLike()
{
    // First create parent matrix, apply op, then return sub-matrix.
    // TODO: currently assumes 2DBC and fixed mb == nb.
    int64_t nb = std::max(this->tileMb(0), this->tileNb(0));
    assert(nb == this->tileMb(0) || this->m() == this->tileMb(0));
    assert(nb == this->tileNb(0) || this->n() == this->tileNb(0));
    int64_t ioffset = this->ioffset();
    int64_t joffset = this->joffset();
    int64_t m = ioffset*nb;
    int64_t n = joffset*nb;
    if (this->op() == Op::NoTrans) {
        m += this->m();
        n += this->n();
    }
    else {
        m += this->n();
        n += this->m();
    }
    int p = this->storage_->p();
    int q = this->storage_->q();
    auto B = Matrix<out_scalar_t>(m, n, nb, p, q, this->mpiComm());
    if (this->op() == Op::Trans) {
        B = transpose( B );
        std::swap(ioffset, joffset);
    }
    else if (this->op() == Op::ConjTrans) {
        B = conj_transpose( B );
        std::swap(ioffset, joffset);
    }
    return B.sub(ioffset, ioffset + this->mt() - 1,
                 joffset, joffset + this->nt() - 1);
}

//------------------------------------------------------------------------------
/// [internal]
/// @see fromLAPACK
///
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
            if (this->tileIsLocal(i, j))
                this->tileInsert(i, j, this->host_num_, &A[ ii + jj*lda ], lda);
            ii += ib;
        }
        jj += jb;
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// @see fromScaLAPACK
/// This differs from LAPACK constructor by adding mb.
///
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    assert(mb == nb);
    // ii, jj are row, col indices
    // ii_local and jj_local are the local array indices in a
    // block-cyclic layout (indxg2l)
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);
        // Using Scalapack indxg2l
        int64_t jj_local = nb*(jj/(nb*q)) + (jj % nb);
        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (this->tileIsLocal(i, j)) {
                // Using Scalapack indxg2l
                int64_t ii_local = mb*(ii/(mb*p)) + (ii % mb);
                this->tileInsert(i, j, this->host_num_,
                                 &A[ ii_local + jj_local*lda ], lda);
            }
            ii += ib;
        }
        jj += jb;
    }
}

//------------------------------------------------------------------------------
/// [internal]
/// @see fromDevices
///
template <typename scalar_t>
Matrix<scalar_t>::Matrix(
    int64_t m, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda,
    int64_t nb, int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    slate_error_if(this->num_devices() != num_devices);

    // ii, jj are row, col indices
    // ii_local and jj_local are the local array indices in a
    // 2D block-cyclic layout.
    // jj_dev is the local array index for the current device in a
    // 1D block-cyclic layout within a node.
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);
        int64_t jj_local = indexGlobal2Local(jj, nb, q);
        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);
            if (this->tileIsLocal(i, j)) {
                int64_t ii_local = indexGlobal2Local(ii, nb, p);
                int dev = this->tileDevice(i, j);
                int64_t jj_dev = indexGlobal2Local(jj_local, nb, num_devices);
                this->tileInsert(i, j, dev,
                                 &Aarray[ dev ][ ii_local + jj_dev*lda ], lda);
            }
            ii += ib;
        }
        jj += jb;
    }
}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, j1:j2 ].
/// This is called from Matrix::sub(i1, i2, j1, j2) and
/// off-diagonal sub(i1, i2, j1, j2) of
/// TriangularMatrix, SymmetricMatrix, HermitianMatrix, etc.
///
/// @param[in] orig
///     Original matrix of which to make sub-matrix.
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
Matrix<scalar_t>::Matrix(
    BaseMatrix< scalar_t >& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{
    this->uplo_ = Uplo::General;
}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
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
Matrix<scalar_t> Matrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    return Matrix<scalar_t>(*this, i1, i2, j1, j2);
}

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
Matrix<scalar_t>::Matrix(
    BaseMatrix<scalar_t>& orig, typename BaseMatrix<scalar_t>::Slice slice)
    : BaseMatrix<scalar_t>(orig, slice)
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
Matrix<scalar_t> Matrix<scalar_t>::slice(
    int64_t row1, int64_t row2,
    int64_t col1, int64_t col2)
{
    return Matrix<scalar_t>(*this,
        typename BaseMatrix<scalar_t>::Slice(row1, row2, col1, col2));
}

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
    for (int device = 0; device < this->num_devices_; ++device)
        num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
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
    for (int device = 0; device < this->num_devices_; ++device)
        num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
    this->storage_->reserveDeviceWorkspace(num_tiles);
}

//------------------------------------------------------------------------------
/// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
/// Primarily for debugging purposes.
///
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
                    Aij.recv(this->tileRank(i, j), this->mpi_comm_, this->layout());
                    tileLayout(i, j, this->layout_);
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
            ii += ib;
        }
        jj += jb;
    }

    this->op_ = op_save;
}

//------------------------------------------------------------------------------
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] target
///     - if target = Devices, inserts tiles on appropriate GPU devices, or
///     - if target = Host, inserts on tiles on CPU host.
///
template <typename scalar_t>
void Matrix<scalar_t>::insertLocalTiles(Target origin)
{
    bool on_devices = (origin == Target::Devices);
    for (int64_t j = 0; j < this->nt(); ++j) {
        for (int64_t i = 0; i < this->mt(); ++i) {
            if (this->tileIsLocal(i, j)) {
                int dev = (on_devices ? this->tileDevice(i, j)
                                      : this->host_num_);
                this->tileInsert(i, j, dev);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// @deprecated
///
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] on_devices
///     If on_devices, inserts tiles on appropriate GPU devices,
///     otherwise inserts tiles on CPU host.
///
template <typename scalar_t>
void Matrix<scalar_t>::insertLocalTiles(bool on_devices)
{
    insertLocalTiles(on_devices ? Target::Devices : Target::Host);
}

//------------------------------------------------------------------------------
/// copy local data of op(A).
/// assumes A has the same distribution, and local tiles are already allocated.
/// TODO this variant copies the Host data only, need to take care of device data
/// TODO handle the op(A) case
template <typename scalar_t>
void Matrix<scalar_t>::copy(Matrix<scalar_t>& A)
{
    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    assert(A_mt <= this->mt());
    assert(A_nt <= this->nt());

    for (int64_t j = 0; j < A_nt; ++j) {
        int64_t jb = A.tileNb(j);
        assert(jb <= this->tileNb(j));

        for (int64_t i = 0; i < A_mt; ++i) {

            if (this->tileIsLocal(i, j)) {
                int64_t ib = A.tileMb(i);
                assert(ib <= this->tileMb(i));

                #pragma omp task
                {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    auto Aij = A.at(i, j);
                    this->tileGetForWriting(i, j, LayoutConvert::None);
                    auto Bij = this->at(i, j);
                    lapack::lacpy(lapack::MatrixType::General, ib, jb,
                                  Aij.data(), Aij.stride(),
                                  Bij.data(), Bij.stride());
                    this->tileLayout(i, j, Aij.layout());
                }
            }
        }
    }
    #pragma omp taskwait
}

} // namespace slate

#endif // SLATE_MATRIX_HH
