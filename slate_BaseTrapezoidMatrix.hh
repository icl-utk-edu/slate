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
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <iostream>

#include "slate_cuda.hh"
#include "slate_cublas.hh"
#include "slate_mpi.hh"
#include "slate_openmp.hh"

namespace slate {

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

    ///-------------------------------------------------------------------------
    /// \brief
    /// @return number of local tiles in matrix on this rank and given device.
    // todo: numLocalDeviceTiles
    int64_t getMaxDeviceTiles(int device)
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

    ///-------------------------------------------------------------------------
    /// @return off-diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    /// This version returns a general Matrix, which:
    /// - if uplo = Lower, is strictly below the diagonal, or
    /// - if uplo = Upper, is strictly above the diagonal.
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
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

    ///-------------------------------------------------------------------------
    /// @return whether the matrix is physically Lower or Upper storage
    ///         (ignoring the transposition operation).
    /// @see uplo_logical()
    Uplo uplo() const { return this->uplo_; }

    ///-------------------------------------------------------------------------
    /// @return whether op(A) is logically Lower or Upper storage,
    ///         taking the transposition operation into account.
    /// @see uplo()
    Uplo uplo_logical() const
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
};

} // namespace slate

#endif // SLATE_BASE_TRAPEZOID_MATRIX_HH
