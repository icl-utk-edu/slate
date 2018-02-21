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
    /// Construct matrix by wrapping existing memory of an m-by-n matrix.
    /// The caller must ensure that the memory remains valid for the lifetime
    /// of the Matrix object and any shallow copies of it.
    /// Input format is an LAPACK-style column-major matrix with leading
    /// dimension (column stride) ld >= m.
    /// Matrix gets tiled with square nb-by-nb tiles.
    Matrix(int64_t m, int64_t n, scalar_t* A, int64_t ld, int64_t nb,
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
                    this->tileInsert(i, j, this->host_num_, &A[ ii + jj*ld ], ld);
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
    /// This version is called for conversion from HermitianMatrix, etc.
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
                        Tile<scalar_t>* tile
                            = this->tileInsert(i, j, &A[(size_t)lda*jj + ii], lda);
                        tile->recv(this->tileRank(i, j), this->mpi_comm_);
                    }
                    // todo: if local, check if need to copy data from tiles to A?
                    // currently assumes local tiles are parts of A.
                    // check pointers and do lacpy if needed.
                }
                else if (this->tileIsLocal(i, j)) {
                    this->at(i, j).send(0, this->mpi_comm_);
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
    ///-------------------------------------------------------------------------
    /// Default constructor
    BaseTrapezoidMatrix():
        BaseMatrix< scalar_t >(),
        uplo_(Uplo::Lower)
    {}

    ///-------------------------------------------------------------------------
    /// Construct matrix by wrapping existing memory of an m-by-n lower
    /// or upper trapezoidal storage matrix. Triangular, symmetric, and
    /// Hermitian matrices all use this storage scheme (with m = n).
    /// The caller must ensure that the memory remains valid for the lifetime
    /// of the Matrix object and any shallow copies of it.
    /// Input format is an LAPACK-style column-major matrix with leading
    /// dimension (column stride) ld >= m.
    /// Matrix gets tiled with square nb-by-nb tiles.
    BaseTrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                        scalar_t* A, int64_t ld, int64_t nb,
                        int p, int q, MPI_Comm mpi_comm):
        BaseMatrix< scalar_t >(m, n, nb, p, q, mpi_comm),
        uplo_(uplo)
    {
        // ii, jj are row, col indices
        // i, j are tile (block row, block col) indices
        if (uplo_ == Uplo::Lower) {
            int64_t jj = 0;
            for (int64_t j = 0; j < this->nt(); ++j) {
                int64_t jb = this->tileNb(j);

                int64_t ii = j*nb;
                for (int64_t i = j; i < this->mt(); ++i) {  // lower
                    int64_t ib = this->tileMb(i);

                    if (this->tileIsLocal(i, j)) {
                        Tile< scalar_t >* tile
                            = this->tileInsert(i, j, this->host_num_, &A[ ii + jj*ld ], ld);
                        if (i == j)
                            tile->uplo(uplo);
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
                        Tile< scalar_t >* tile
                            = this->tileInsert(i, j, this->host_num_, &A[ ii + jj*ld ], ld);
                        if (i == j)
                            tile->uplo(uplo);
                    }
                    ii += ib;
                }
                jj += jb;
            }
        }
    }

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a trapezoid matrix, with the
    /// same diagonal as the parent matrix.
    BaseTrapezoidMatrix(BaseTrapezoidMatrix& orig,
                        int64_t i1, int64_t i2):
        BaseMatrix< scalar_t >(orig, i1, i2, i1, i2),
        uplo_(orig.uplo_)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, j1:j2 ].
    /// This version is called for conversion from general Matrix, etc.
    BaseTrapezoidMatrix(Uplo uplo, BaseMatrix< scalar_t >& orig,
                        int64_t i1, int64_t i2,
                        int64_t j1, int64_t j2):
        BaseMatrix< scalar_t >(orig, i1, i2, j1, j2),
        uplo_(orig.uplo)
    {}

    ///-------------------------------------------------------------------------
    /// Swap contents of matrices A and B.
    friend void swap(BaseTrapezoidMatrix& A, BaseTrapezoidMatrix& B)
    {
        using std::swap;
        swap(static_cast< BaseMatrix< scalar_t >& >(A),
             static_cast< BaseMatrix< scalar_t >& >(B));
        swap(A.uplo_, B.uplo_);
    }

    ///-------------------------------------------------------------------------
    /// \brief
    /// @return number of local tiles in matrix on this rank.
    // todo: numLocalTiles? use for life as well?
    int64_t getMaxHostTiles()
    {
        int64_t num_tiles = 0;
        if (uplo_ == Uplo::Lower) {
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
        if (uplo_ == Uplo::Lower) {
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
        if (uplo_ == Uplo::Lower) {
            int64_t jj = 0;
            for (int64_t j = 0; j < this->nt(); ++j) {
                int64_t jb = this->tileNb(j);

                int64_t ii = 0;
                for (int64_t i = j; i < this->mt(); ++i) {  // lower
                    int64_t ib = this->tileMb(i);

                    if (this->mpi_rank_ == 0) {
                        if (! this->tileIsLocal(i, j)) {
                            Tile<scalar_t>* tile
                                = this->tileInsert(i, j, &A[(size_t)lda*jj + ii], lda);
                            tile->recv(this->tileRank(i, j), this->mpi_comm_);
                        }
                        // todo: if local, check if need to copy data from tiles to A?
                        // currently assumes local tiles are parts of A.
                        // check pointers and do lacpy if needed.
                    }
                    else if (this->tileIsLocal(i, j)) {
                        this->at(i, j).send(0, this->mpi_comm_);
                    }
                    ii += ib;
                }
                jj += jb;
            }
        }
        else {
            int64_t jj = 0;
            for (int64_t j = 0; j < this->nt(); ++j) {
                int64_t jb = this->tileNb(j);

                int64_t ii = 0;
                for (int64_t i = 0; i <= j && i < this->mt(); ++i) {  // upper
                    int64_t ib = this->tileMb(i);

                    if (this->mpi_rank_ == 0) {
                        if (! this->tileIsLocal(i, j)) {
                            Tile<scalar_t>* tile
                                = this->tileInsert(i, j, &A[(size_t)lda*jj + ii], lda);
                            tile->recv(this->tileRank(i, j), this->mpi_comm_);
                        }
                        // todo: if local, check if need to copy data from tiles to A?
                        // currently assumes local tiles are parts of A.
                        // check pointers and do lacpy if needed.
                    }
                    else if (this->tileIsLocal(i, j)) {
                        this->at(i, j).send(0, this->mpi_comm_);
                    }
                    ii += ib;
                }
                jj += jb;
            }
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
        if (this->uplo_ == Uplo::Lower) {
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
    Uplo uplo() const { return uplo_; }

protected:
    Uplo uplo_;
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
    /// Construct matrix by wrapping existing memory of an m-by-n lower
    /// or upper trapezoidal matrix.
    /// @see BaseTrapezoidMatrix
    TrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                    scalar_t* A, int64_t ld, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, m, n, A, ld, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a trapezoid matrix, with the
    /// same diagonal as the parent matrix.
    TrapezoidMatrix(TrapezoidMatrix& orig,
                    int64_t i1, int64_t i2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2)
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
    /// Construct matrix by wrapping existing memory of an n-by-n lower
    /// or upper triangular matrix.
    /// @see BaseTrapezoidMatrix
    TriangularMatrix(Uplo uplo, int64_t n,
                     scalar_t* A, int64_t ld, int64_t nb,
                     int p, int q, MPI_Comm mpi_comm):
        TrapezoidMatrix< scalar_t >(uplo, n, n, A, ld, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a triangular matrix, with the
    /// same diagonal as the parent matrix.
    TriangularMatrix(TriangularMatrix& orig,
                     int64_t i1, int64_t i2):
        TrapezoidMatrix< scalar_t >(orig, i1, i2)
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
///
// todo: transparent conversion between real-symmetric <=> real-Hermitian
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
    /// or upper symmetric matrix.
    /// @see BaseTrapezoidMatrix
    SymmetricMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t ld, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, ld, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a symmetric matrix, with the
    /// same diagonal as the parent matrix.
    SymmetricMatrix(SymmetricMatrix& orig,
                    int64_t i1, int64_t i2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2)
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
///
// todo: transparent conversion between real-symmetric <=> real-Hermitian
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
    /// or upper Hermitian matrix.
    /// @see BaseTrapezoidMatrix
    HermitianMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t ld, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, ld, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, i1:i2 ]. The new view is still a Hermitian matrix, with the
    /// same diagonal as the parent matrix.
    HermitianMatrix(HermitianMatrix& orig,
                    int64_t i1, int64_t i2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2)
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
