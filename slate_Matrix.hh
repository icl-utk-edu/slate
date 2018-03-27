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
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <iostream>

#ifdef SLATE_WITH_CUDA
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
    #include "slate_NoCublas.hh"
#endif

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef _OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

namespace slate {

///=============================================================================
///
template <typename scalar_t>
class Matrix: public BaseMatrix< scalar_t > {
public:
    ///-------------------------------------------------------------------------
    /// Default constructor
    Matrix():
        BaseMatrix< scalar_t >()
    {}

    ///-------------------------------------------------------------------------
    /// Named constructor returns new Matrix.
    /// Construct matrix by wrapping existing memory of an m-by-n LAPACK matrix.
    /// The caller must ensure that the memory remains valid for the lifetime
    /// of the Matrix object and any shallow copies of it.
    /// Input format is an LAPACK-style column-major matrix with leading
    /// dimension (column stride) lda >= m.
    /// Matrix gets tiled with square nb-by-nb tiles.
    static
    Matrix fromLAPACK(int64_t m, int64_t n,
                      scalar_t* A, int64_t lda, int64_t nb,
                      int p, int q, MPI_Comm mpi_comm)
    {
        return Matrix(m, n, A, lda, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// Named constructor returns new Matrix.
    /// Construct matrix by wrapping existing memory of an m-by-n ScaLAPACK matrix.
    /// The caller must ensure that the memory remains valid for the lifetime
    /// of the Matrix object and any shallow copies of it.
    /// Input format is a ScaLAPACK-style 2D block-cyclic column-major matrix
    /// with local leading dimension (column stride) lda,
    /// p block rows and q block columns.
    /// Matrix gets tiled with square nb-by-nb tiles.
    static
    Matrix fromScaLAPACK(int64_t m, int64_t n,
                         scalar_t* A, int64_t lda, int64_t nb,
                         int p, int q, MPI_Comm mpi_comm)
    {
        // note extra nb
        return Matrix(m, n, A, lda, nb, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// @see fromLAPACK
    /// todo: make this protected
    Matrix(int64_t m, int64_t n,
           scalar_t* A, int64_t lda, int64_t nb,
           int p, int q, MPI_Comm mpi_comm):
        BaseMatrix< scalar_t >(m, n, nb, p, q, mpi_comm)
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

    ///-------------------------------------------------------------------------
    /// @see fromScaLAPACK
    /// This differs from LAPACK constructor by adding mb.
    /// todo: make this protected
    Matrix(int64_t m, int64_t n,
           scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
           int p, int q, MPI_Comm mpi_comm):
        BaseMatrix< scalar_t >(m, n, nb, p, q, mpi_comm)
    {
        // ii, jj are row, col indices
        // ii_loc and jj_loc are the local array indices in a block-cyclic layout (indxg2l)
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
                    this->tileInsert(i, j, this->host_num_, &A[ ii_loc + jj_loc*lda ], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, j1:j2 ].
    Matrix(Matrix& orig,
           int64_t i1, int64_t i2, int64_t j1, int64_t j2):
        BaseMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, j1:j2 ].
    /// This version is called for conversion from off-diagonal submatrix of
    /// TriangularMatrix, SymmetricMatrix, HermitianMatrix, etc.
    Matrix(BaseMatrix< scalar_t >& orig,
           int64_t i1, int64_t i2,
           int64_t j1, int64_t j2):
        BaseMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// @return sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    Matrix sub(int64_t i1, int64_t i2,
               int64_t j1, int64_t j2)
    {
        return Matrix(*this, i1, i2, j1, j2);
    }

    ///-------------------------------------------------------------------------
    /// Swap contents of matrices A and B.
    // (This isn't really needed over BaseMatrix swap, but is here as a reminder
    // in case any members are added to Matrix that aren't in BaseMatrix.)
    friend void swap(Matrix& A, Matrix& B)
    {
        using std::swap;
        swap(static_cast< BaseMatrix< scalar_t >& >(A),
             static_cast< BaseMatrix< scalar_t >& >(B));
    }

    ///-------------------------------------------------------------------------
    /// \brief
    /// @return number of local tiles in matrix on this rank.
    // todo: numLocalTiles? use for life as well?
    int64_t getMaxHostTiles()
    {
        int64_t num_tiles = 0;
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = 0; i < this->mt(); ++i)
                if (this->tileIsLocal(i, j))
                    ++num_tiles;

        return num_tiles;
    }

    ///-------------------------------------------------------------------------
    /// \brief
    /// @return number of local tiles in matrix on this rank and given device.
    // todo: numLocalDeviceTiles?
    int64_t getMaxDeviceTiles(int device)
    {
        int64_t num_tiles = 0;
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = 0; i < this->mt(); ++i)
                if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                    ++num_tiles;

        return num_tiles;
    }

    ///-------------------------------------------------------------------------
    /// Allocates batch arrays for all devices.
    void allocateBatchArrays()
    {
        int64_t num_tiles = 0;
        for (int device = 0; device < this->num_devices_; ++device) {
            num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
        }
        this->storage_->allocateBatchArrays(num_tiles);
    }

    ///-------------------------------------------------------------------------
    /// Reserve space for temporary workspace tiles on host.
    void reserveHostWorkspace()
    {
        this->storage_->reserveHostWorkspace(getMaxHostTiles());
    }

    ///-------------------------------------------------------------------------
    /// Reserve space for temporary workspace tiles on all GPU devices.
    void reserveDeviceWorkspace()
    {
        int64_t num_tiles = 0;
        for (int device = 0; device < this->num_devices_; ++device) {
            num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
        }
        this->storage_->reserveDeviceWorkspace(num_tiles);
    }

    ///-------------------------------------------------------------------------
    /// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
    /// Primarily for debugging purposes.
    void gather(scalar_t *A, int64_t lda)
    {
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
    }
};

///=============================================================================
///
template <typename scalar_t>
class BaseTrapezoidMatrix: public BaseMatrix< scalar_t > {
public:
    /// todo: make all constructors protected for Base class?

    ///-------------------------------------------------------------------------
    /// Default constructor
    BaseTrapezoidMatrix():
        BaseMatrix< scalar_t >()
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
    /// dimension (column stride) lda >= m.
    /// Matrix gets tiled with square nb-by-nb tiles.
    BaseTrapezoidMatrix(Uplo in_uplo, int64_t m, int64_t n,
                        scalar_t* A, int64_t lda, int64_t nb,
                        int p, int q, MPI_Comm mpi_comm):
        BaseMatrix< scalar_t >(m, n, nb, p, q, mpi_comm)
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
                        this->tileInsert(i, j, this->host_num_, &A[ ii + jj*lda ], lda);
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
                        this->tileInsert(i, j, this->host_num_, &A[ ii + jj*lda ], lda);
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
    BaseTrapezoidMatrix(Uplo in_uplo, int64_t m, int64_t n,
                        scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                        int p, int q, MPI_Comm mpi_comm):
        BaseMatrix< scalar_t >(m, n, nb, p, q, mpi_comm)
    {
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
                        this->tileInsert(i, j, this->host_num_, &A[ ii_loc + jj_loc*lda ], lda);
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
                        this->tileInsert(i, j, this->host_num_, &A[ ii_loc + jj_loc*lda ], lda);
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
    BaseTrapezoidMatrix(Uplo uplo, Matrix< scalar_t >& orig):
        BaseMatrix< scalar_t >(orig)
    {
        this->uplo_ = uplo;
    }

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix, sub-matrix constructor
    /// creates shallow copy view of original matrix, A[ i1:i2, j1:j2 ].
    BaseTrapezoidMatrix(Uplo uplo, Matrix< scalar_t >& orig,
                        int64_t i1, int64_t i2,
                        int64_t j1, int64_t j2):
        BaseMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {
        this->uplo_ = uplo;
    }

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, j1:j2 ]. Requires i1 == j1. The new view is still a trapezoid
    /// matrix, with the same diagonal as the parent matrix.
    BaseTrapezoidMatrix(BaseTrapezoidMatrix& orig,
                        int64_t i1, int64_t i2,
                        int64_t j1, int64_t j2):
        BaseMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {
        this->uplo_ = orig.uplo_;
        if (i1 != j1) {
            throw std::exception();
        }
    }

    ///-------------------------------------------------------------------------
    /// Swap contents of matrices A and B.
    friend void swap(BaseTrapezoidMatrix& A, BaseTrapezoidMatrix& B)
    {
        using std::swap;
        swap(static_cast< BaseMatrix< scalar_t >& >(A),
             static_cast< BaseMatrix< scalar_t >& >(B));
    }

    ///-------------------------------------------------------------------------
    /// \brief
    /// @return number of local tiles in matrix on this rank.
    // todo: numLocalTiles? use for life as well?
    int64_t getMaxHostTiles()
    {
        int64_t num_tiles = 0;
        if ((this->uplo() == Uplo::Lower && this->op() == Op::NoTrans) ||
            (this->uplo() == Uplo::Upper && this->op() != Op::NoTrans)) {
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

    ///-------------------------------------------------------------------------
    /// \brief
    /// @return number of local tiles in matrix on this rank and given device.
    // todo: numLocalDeviceTiles
    int64_t getMaxDeviceTiles(int device)
    {
        int64_t num_tiles = 0;
        if ((this->uplo() == Uplo::Lower && this->op() == Op::NoTrans) ||
            (this->uplo() == Uplo::Upper && this->op() != Op::NoTrans)) {
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

    ///-------------------------------------------------------------------------
    /// Allocates batch arrays for all devices.
    void allocateBatchArrays()
    {
        int64_t num_tiles = 0;
        for (int device = 0; device < this->num_devices_; ++device) {
            num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
        }
        this->storage_->allocateBatchArrays(num_tiles);
    }

    ///-------------------------------------------------------------------------
    /// Reserve space for temporary workspace tiles on host.
    void reserveHostWorkspace()
    {
        this->storage_->reserveHostWorkspace(getMaxHostTiles());
    }

    ///-------------------------------------------------------------------------
    /// Reserve space for temporary workspace tiles on all GPU devices.
    void reserveDeviceWorkspace()
    {
        int64_t num_tiles = 0;
        for (int device = 0; device < this->num_devices_; ++device) {
            num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
        }
        this->storage_->reserveDeviceWorkspace(num_tiles);
    }

    ///-------------------------------------------------------------------------
    /// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
    /// Primarily for debugging purposes.
    void gather(scalar_t *A, int64_t lda)
    {
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
    }

    ///-------------------------------------------------------------------------
    /// @return off-diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    /// This version returns a general Matrix, which:
    /// - if uplo = Lower, is strictly below the diagonal, or
    /// - if uplo = Upper, is strictly above the diagonal.
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
    {
        if ((this->uplo() == Uplo::Lower && this->op() == Op::NoTrans) ||
            (this->uplo() == Uplo::Upper && this->op() != Op::NoTrans)) {
            // top-right corner is at or below diagonal
            assert(i1 >= j2);
        }
        else {
            // bottom-left corner is at or above diagonal
            assert(i2 <= j1);
        }
        return Matrix< scalar_t >(*this, i1, i2, j1, j2);
    }

    ///-------------------------------------------------------------------------
    /// @return whether the matrix is Lower or Upper storage.
    Uplo uplo() const { return this->uplo_; }
};

///=============================================================================
///
template <typename scalar_t>
class TrapezoidMatrix: public BaseTrapezoidMatrix< scalar_t > {
public:
    ///-------------------------------------------------------------------------
    /// Default constructor
    TrapezoidMatrix():
        BaseTrapezoidMatrix< scalar_t >()
    {}

    ///-------------------------------------------------------------------------
    /// Named constructor returns new TrapezoidMatrix.
    /// Construct matrix by wrapping existing memory of an m-by-n lower
    /// or upper trapezoidal LAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    TrapezoidMatrix fromLAPACK(Uplo uplo, int64_t m, int64_t n,
                               scalar_t* A, int64_t lda, int64_t nb,
                               int p, int q, MPI_Comm mpi_comm)
    {
        return TrapezoidMatrix(uplo, m, n, A, lda, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// Named constructor returns new TrapezoidMatrix.
    /// Construct matrix by wrapping existing memory of an m-by-n lower
    /// or upper trapezoidal ScaLAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    TrapezoidMatrix fromScaLAPACK(Uplo uplo, int64_t m, int64_t n,
                                  scalar_t* A, int64_t lda, int64_t nb,
                                  int p, int q, MPI_Comm mpi_comm)
    {
        // note extra nb
        return TrapezoidMatrix(uplo, m, n, A, lda, nb, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// @see fromLAPACK
    /// todo: make this protected
    TrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                    scalar_t* A, int64_t lda, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, m, n, A, lda, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// @see fromScaLAPACK
    /// This differs from LAPACK constructor by adding mb.
    /// todo: make this protected
    TrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, m, n, A, lda, mb, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
    /// creates a shallow copy view of the original matrix.
    explicit TrapezoidMatrix(BaseTrapezoidMatrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(orig)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
    /// creates a shallow copy view of the original matrix, A[ i1:i2, j1:j2 ].
    TrapezoidMatrix(BaseTrapezoidMatrix< scalar_t >& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix
    /// creates a shallow copy view of the original matrix.
    TrapezoidMatrix(Uplo uplo, Matrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(uplo, orig)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix, sub-matrix constructor
    /// creates shallow copy view of original matrix, A[ i1:i2, j1:j2 ].
    TrapezoidMatrix(Uplo uplo, Matrix< scalar_t >& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2):
        BaseTrapezoidMatrix< scalar_t >(uplo, orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, j1:j2 ]. Requires i1 == j1. The new view is still a trapezoid
    /// matrix, with the same diagonal as the parent matrix.
    TrapezoidMatrix(TrapezoidMatrix& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// @return diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, i1:i2 ].
    /// This version returns a TrapezoidMatrix with the same diagonal as the
    /// parent matrix.
    /// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
    ///                                  int64_t j1, int64_t j2)
    TrapezoidMatrix sub(int64_t i1, int64_t i2)
    {
        return TrapezoidMatrix(*this, i1, i2, i1, i2);
    }

    ///-------------------------------------------------------------------------
    /// @return off-diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    /// This version returns a general Matrix, which:
    /// - if uplo = Lower, is strictly below the diagonal, or
    /// - if uplo = Upper, is strictly above the diagonal.
    /// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
    {
        return BaseTrapezoidMatrix< scalar_t >::sub(i1, i2, j1, j2);
    }

    ///-------------------------------------------------------------------------
    /// Swap contents of matrices A and B.
    // (This isn't really needed over BaseTrapezoidMatrix swap, but is here as a
    // reminder in case any members are added that aren't in BaseTrapezoidMatrix.)
    friend void swap(TrapezoidMatrix& A, TrapezoidMatrix& B)
    {
        using std::swap;
        swap(static_cast< BaseTrapezoidMatrix< scalar_t >& >(A),
             static_cast< BaseTrapezoidMatrix< scalar_t >& >(B));
    }
};

///=============================================================================
///
template <typename scalar_t>
class TriangularMatrix: public TrapezoidMatrix< scalar_t > {
public:
    ///-------------------------------------------------------------------------
    /// Default constructor
    TriangularMatrix():
        TrapezoidMatrix< scalar_t >()
    {}

    ///-------------------------------------------------------------------------
    /// Named constructor returns new TriangularMatrix.
    /// Construct matrix by wrapping existing memory of an n-by-n lower
    /// or upper triangular LAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    TriangularMatrix fromLAPACK(Uplo uplo, int64_t n,
                                scalar_t* A, int64_t lda, int64_t nb,
                                int p, int q, MPI_Comm mpi_comm)
    {
        return TriangularMatrix(uplo, n, A, lda, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// Named constructor returns new TrapezoidMatrix.
    /// Construct matrix by wrapping existing memory of an n-by-n lower
    /// or upper triangular ScaLAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    TriangularMatrix fromScaLAPACK(Uplo uplo, int64_t n,
                                   scalar_t* A, int64_t lda, int64_t nb,
                                   int p, int q, MPI_Comm mpi_comm)
    {
        // note extra nb
        return TriangularMatrix(uplo, n, A, lda, nb, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// @see fromLAPACK
    /// todo: make this protected
    TriangularMatrix(Uplo uplo, int64_t n,
                     scalar_t* A, int64_t lda, int64_t nb,
                     int p, int q, MPI_Comm mpi_comm):
        TrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// @see fromScaLAPACK
    /// This differs from LAPACK constructor by adding mb.
    /// todo: make this protected
    TriangularMatrix(Uplo uplo, int64_t n,
                     scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                     int p, int q, MPI_Comm mpi_comm):
        TrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, mb, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
    /// creates a shallow copy view of the original matrix.
    /// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
    explicit TriangularMatrix(BaseTrapezoidMatrix< scalar_t >& orig):
        TrapezoidMatrix< scalar_t >(orig,
            0, std::min( orig.mt()-1, orig.nt()-1 ),
            0, std::min( orig.mt()-1, orig.nt()-1 ))
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix
    /// creates a shallow copy view of the original matrix.
    /// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
    TriangularMatrix(Uplo uplo, Matrix< scalar_t >& orig):
        TrapezoidMatrix< scalar_t >(uplo, orig,
            0, std::min( orig.mt()-1, orig.nt()-1 ),
            0, std::min( orig.mt()-1, orig.nt()-1 ))
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a triangular matrix, with the
    /// same diagonal as the parent matrix.
    TriangularMatrix(TriangularMatrix& orig,
                     int64_t i1, int64_t i2):
        TrapezoidMatrix< scalar_t >(orig, i1, i2, i1, i2)
    {}

    ///-------------------------------------------------------------------------
    /// @return diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, i1:i2 ].
    /// This version returns a TriangularMatrix with the same diagonal as the
    /// parent matrix.
    /// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
    ///                                  int64_t j1, int64_t j2)
    TriangularMatrix sub(int64_t i1, int64_t i2)
    {
        return TriangularMatrix(*this, i1, i2);
    }

    ///-------------------------------------------------------------------------
    /// @return off-diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    /// This version returns a general Matrix, which:
    /// - if uplo = Lower, is strictly below the diagonal, or
    /// - if uplo = Upper, is strictly above the diagonal.
    /// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
    {
        return BaseTrapezoidMatrix< scalar_t >::sub(i1, i2, j1, j2);
    }

    ///-------------------------------------------------------------------------
    /// Swaps contents of matrices A and B.
    // (This isn't really needed over TrapezoidMatrix swap, but is here as a
    // reminder in case any members are added that aren't in TrapezoidMatrix.)
    friend void swap(TriangularMatrix& A, TriangularMatrix& B)
    {
        using std::swap;
        swap(static_cast< TrapezoidMatrix< scalar_t >& >(A),
             static_cast< TrapezoidMatrix< scalar_t >& >(B));
    }
};

///=============================================================================
template <typename scalar_t>
class SymmetricMatrix: public BaseTrapezoidMatrix< scalar_t > {
public:
    ///-------------------------------------------------------------------------
    /// Default constructor
    SymmetricMatrix():
        BaseTrapezoidMatrix< scalar_t >()
    {}

    ///-------------------------------------------------------------------------
    /// Construct matrix by wrapping existing memory of an n-by-n lower
    /// or upper symmetric LAPACK matrix.
    /// @see BaseTrapezoidMatrix
    static
    SymmetricMatrix fromLAPACK(Uplo uplo, int64_t n,
                               scalar_t* A, int64_t lda, int64_t nb,
                               int p, int q, MPI_Comm mpi_comm)
    {
        return SymmetricMatrix(uplo, n, A, lda, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// Construct matrix by wrapping existing memory of an n-by-n lower
    /// or upper symmetric ScaLAPACK matrix.
    /// @see BaseTrapezoidMatrix
    static
    SymmetricMatrix fromScaLAPACK(Uplo uplo, int64_t n,
                                  scalar_t* A, int64_t lda, int64_t nb,
                                  int p, int q, MPI_Comm mpi_comm)
    {
        // note extra nb
        return SymmetricMatrix(uplo, n, A, lda, nb, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// @see fromLAPACK
    /// todo: make this protected
    SymmetricMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t lda, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// @see fromScaLAPACK
    /// This differs from LAPACK constructor by adding mb.
    /// todo: make this protected
    SymmetricMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, mb, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
    /// creates a shallow copy view of the original matrix.
    /// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
    explicit SymmetricMatrix(BaseTrapezoidMatrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(orig,
            0, std::min( orig.mt()-1, orig.nt()-1 ),
            0, std::min( orig.mt()-1, orig.nt()-1 ))
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix
    /// creates a shallow copy view of the original matrix.
    /// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
    SymmetricMatrix(Uplo uplo, Matrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(uplo, orig,
            0, std::min( orig.mt()-1, orig.nt()-1 ),
            0, std::min( orig.mt()-1, orig.nt()-1 ))
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a symmetric matrix, with the
    /// same diagonal as the parent matrix.
    SymmetricMatrix(SymmetricMatrix& orig,
                    int64_t i1, int64_t i2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2, i1, i2)
    {}

    ///-------------------------------------------------------------------------
    /// @return sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, i1:i2 ].
    /// This version returns a SymmetricMatrix with the same diagonal as the
    /// parent matrix.
    /// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
    ///                                  int64_t j1, int64_t j2)
    SymmetricMatrix sub(int64_t i1, int64_t i2)
    {
        return SymmetricMatrix(*this, i1, i2);
    }

    ///-------------------------------------------------------------------------
    /// @return off-diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    /// This version returns a general Matrix, which:
    /// - if uplo = Lower, is strictly below the diagonal, or
    /// - if uplo = Upper, is strictly above the diagonal.
    /// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
    {
        return BaseTrapezoidMatrix< scalar_t >::sub(i1, i2, j1, j2);
    }

    ///-------------------------------------------------------------------------
    /// Swaps contents of matrices A and B.
    // (This isn't really needed over BaseTrapezoidMatrix swap, but is here as a
    // reminder in case any members are added that aren't in BaseTrapezoidMatrix.)
    friend void swap(SymmetricMatrix& A, SymmetricMatrix& B)
    {
        using std::swap;
        swap(static_cast< BaseTrapezoidMatrix< scalar_t >& >(A),
             static_cast< BaseTrapezoidMatrix< scalar_t >& >(B));
    }
};

///=============================================================================
template <typename scalar_t>
class HermitianMatrix: public BaseTrapezoidMatrix< scalar_t > {
public:
    ///-------------------------------------------------------------------------
    /// Default constructor
    HermitianMatrix():
        BaseTrapezoidMatrix< scalar_t >()
    {}

    ///-------------------------------------------------------------------------
    /// Construct matrix by wrapping existing memory of an n-by-n lower
    /// or upper Hermitian LAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    HermitianMatrix fromLAPACK(Uplo uplo, int64_t n,
                               scalar_t* A, int64_t lda, int64_t nb,
                               int p, int q, MPI_Comm mpi_comm)
    {
        return HermitianMatrix(uplo, n, A, lda, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// Construct matrix by wrapping existing memory of an n-by-n lower
    /// or upper Hermitian ScaLAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    HermitianMatrix fromScaLAPACK(Uplo uplo, int64_t n,
                                  scalar_t* A, int64_t lda, int64_t nb,
                                  int p, int q, MPI_Comm mpi_comm)
    {
        // note extra nb
        return HermitianMatrix(uplo, n, A, lda, nb, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// @see fromLAPACK
    /// todo: make this protected
    HermitianMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t lda, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// @see fromScaLAPACK
    /// This differs from LAPACK constructor by adding mb.
    /// todo: make this protected
    HermitianMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, mb, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
    /// creates a shallow copy view of the original matrix.
    /// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
    explicit HermitianMatrix(BaseTrapezoidMatrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(orig,
            0, std::min( orig.mt()-1, orig.nt()-1 ),
            0, std::min( orig.mt()-1, orig.nt()-1 ))
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix
    /// creates a shallow copy view of the original matrix.
    /// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
    HermitianMatrix(Uplo uplo, Matrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(uplo, orig,
            0, std::min( orig.mt()-1, orig.nt()-1 ),
            0, std::min( orig.mt()-1, orig.nt()-1 ))
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a Hermitian matrix, with the
    /// same diagonal as the parent matrix.
    HermitianMatrix(HermitianMatrix& orig,
                    int64_t i1, int64_t i2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2, i1, i2)
    {}

    ///-------------------------------------------------------------------------
    /// @return sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, i1:i2 ].
    /// This version returns a HermitianMatrix with the same diagonal as the
    /// parent matrix.
    /// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
    ///                                  int64_t j1, int64_t j2)
    HermitianMatrix sub(int64_t i1, int64_t i2)
    {
        return HermitianMatrix(*this, i1, i2);
    }

    ///-------------------------------------------------------------------------
    /// @return off-diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    /// This version returns a general Matrix, which:
    /// - if uplo = Lower, is strictly below the diagonal, or
    /// - if uplo = Upper, is strictly above the diagonal.
    /// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
    {
        return BaseTrapezoidMatrix< scalar_t >::sub(i1, i2, j1, j2);
    }

    ///-------------------------------------------------------------------------
    /// Swaps contents of matrices A and B.
    // (This isn't really needed over BaseTrapezoidMatrix swap, but is here as a
    // reminder in case any members are added that aren't in BaseTrapezoidMatrix.)
    friend void swap(HermitianMatrix& A, HermitianMatrix& B)
    {
        using std::swap;
        swap(static_cast< BaseTrapezoidMatrix< scalar_t >& >(A),
             static_cast< BaseTrapezoidMatrix< scalar_t >& >(B));
    }
};

} // namespace slate

#endif // SLATE_MATRIX_HH
