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
class TriangularBandMatrix: public TriangularMatrix<scalar_t> {
public:
    // constructors
    TriangularBandMatrix();

    TriangularBandMatrix(
        Uplo uplo, Diag diag,
        int64_t n, int64_t kd,
        int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    TriangularBandMatrix(Uplo uplo, Diag diag, BandMatrix<scalar_t>& orig);

public:
    template <typename T>
    friend void swap(TriangularBandMatrix<T>& A, TriangularBandMatrix<T>& B);

    int64_t bandwidth() const;
    void    bandwidth(int64_t kd);

    void    gather(scalar_t* A, int64_t lda);
    void    gatherAll(std::set<int>& rank_set, int tag = 0, int64_t life_factor = 1);
    void    ge2tbGather(Matrix<scalar_t>& A);
    void    insertLocalTiles(Target origin=Target::Host);

    // todo: specialize for band
    // int64_t getMaxHostTiles();
    // int64_t getMaxDeviceTiles(int device);
    // void allocateBatchArrays();
    // void reserveHostWorkspace();
    // void reserveDeviceWorkspace();
    // todo: void tileUpdateAllOrigin();
    // void gather(scalar_t* A, int64_t lda);

protected:
    int64_t kd_;
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix()
    : TriangularMatrix<scalar_t>(),
      kd_(0)
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n band matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
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
    : TriangularMatrix<scalar_t>(uplo, diag, n, nb, p, q, mpi_comm),
      kd_(kd)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// todo:
/// Conversion from general band matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TriangularBandMatrix<scalar_t>::TriangularBandMatrix(
    Uplo uplo, Diag diag, BandMatrix<scalar_t>& orig)
    : TriangularMatrix<scalar_t>(uplo, diag, orig),
      kd_((uplo == Uplo::Lower) == (orig.op() == Op::NoTrans)
            ? orig.lowerBandwidth()
            : orig.upperBandwidth())
{}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(TriangularBandMatrix<scalar_t>& A, TriangularBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< TriangularMatrix<scalar_t>& >(A),
         static_cast< TriangularMatrix<scalar_t>& >(B));
    swap(A.kd_, B.kd_);
}

//------------------------------------------------------------------------------
/// @return number of subdiagonals within band.
template <typename scalar_t>
int64_t TriangularBandMatrix<scalar_t>::bandwidth() const
{
    return kd_;
}

//------------------------------------------------------------------------------
/// Sets number of subdiagonals within band.
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::bandwidth(int64_t kd)
{
    kd_  = kd;
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
    int64_t kdt = ceildiv( this->kd_, this->tileNb(0) );
    for (int64_t j = 0; j < nt; ++j) {
        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = istart; i <= iend; ++i) {

            // If receiving the tile.
            if (! this->tileIsLocal(i, j)) {
                // Create tile to receive data, with life span.
                // If tile already exists, add to its life span.
                LockGuard(this->storage_->getTilesMapLock()); // todo: accessor
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
    int64_t kdt = ceildiv( this->kd_, this->tileNb(0) );
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
//  to TriangularBandMatrix B on MPI rank 0.
/// Primarily for SVD code
///
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::ge2tbGather(Matrix<scalar_t>& A)
{
    Op op_save = this->op();
    this->op_ = Op::NoTrans;
    auto upper = this->uplo() == Uplo::Upper;

    int64_t mt = A.mt();
    int64_t nt = A.nt();
    int64_t kdt = ceildiv( this->kd_, this->tileNb(0) );
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < nt; ++j) {
        int64_t jb = A.tileNb(j);

        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = 0; i < mt; ++i) {
            int64_t ib = A.tileMb(i);
            if (i >= istart && i <= iend) {
                if (this->mpi_rank_ == 0) {
                    if (! A.tileIsLocal(i, j)) {
                        // erase any existing non-local tile and insert new one
                        // A.tileErase(i, j, A.host_num_);
                        this->tileInsert(i, j, this->host_num_);
                        auto Bij = this->at(i, j);
                        Bij.recv(A.tileRank(i, j), this->mpi_comm_, this->layout());
                        //A.tileLayout(i, j, this->layout_);
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

//------------------------------------------------------------------------------
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] target
///     - if target = Devices, inserts tiles on appropriate GPU devices, or
///     - if target = Host, inserts on tiles on CPU host.
///
template <typename scalar_t>
void TriangularBandMatrix<scalar_t>::insertLocalTiles(Target origin)
{
    bool on_devices = (origin == Target::Devices);
    auto upper = this->uplo() == Uplo::Upper;
    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t kdt = ceildiv( this->kd_, this->tileNb(0) );
    for (int64_t j = 0; j < nt; ++j) {
        int64_t istart = upper ? blas::max( 0, j-kdt ) : j;
        int64_t iend   = upper ? j : blas::min( j+kdt, mt-1 );
        for (int64_t i = istart; i <= iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                int dev = (on_devices ? this->tileDevice(i, j)
                                      : this->host_num_);
                this->tileInsert(i, j, dev);
            }
        }
    }
}


} // namespace slate

#endif // SLATE_TRIANGULAR_BAND_MATRIX_HH
