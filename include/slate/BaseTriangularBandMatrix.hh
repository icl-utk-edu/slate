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

#ifndef SLATE_BASE_TRIANGULAR_BAND_MATRIX_HH
#define SLATE_BASE_TRIANGULAR_BAND_MATRIX_HH

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
/// Triangular banded, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class BaseTriangularBandMatrix: public BaseBandMatrix<scalar_t> {
public:
    using ij_tuple = std::tuple<int64_t, int64_t>;

protected:
    // constructors
    BaseTriangularBandMatrix();

    BaseTriangularBandMatrix(Uplo uplo, int64_t n, int64_t kd,
                            std::function<int64_t (int64_t j)>& inTileNb,
                            std::function<int (ij_tuple ij)>& inTileRank,
                            std::function<int (ij_tuple ij)>& inTileDevice,
                            MPI_Comm mpi_comm);

    BaseTriangularBandMatrix(
        Uplo uplo,
        int64_t n, int64_t kd,
        int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    BaseTriangularBandMatrix(Uplo uplo, BaseBandMatrix<scalar_t>& orig);

    BaseTriangularBandMatrix(int64_t kd, BaseMatrix<scalar_t>& orig);

public:
    template <typename T>
    friend void swap(BaseTriangularBandMatrix<T>& A, BaseTriangularBandMatrix<T>& B);

    int64_t bandwidth() const;
    void    bandwidth(int64_t kd);

    void    insertLocalTiles(Target origin=Target::Host);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty band matrix with bandwidth = 0.
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix()
    : BaseBandMatrix<scalar_t>()
{
    this->uplo_ = Uplo::Lower;
}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n band matrix, with no tiles allocated,
/// where tileNb, tileRank, tileDevice are given as functions.
/// Tiles can be added with tileInsert().
///
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    Uplo uplo, int64_t n, int64_t kd,
    std::function<int64_t (int64_t j)>& inTileNb,
    std::function<int (ij_tuple ij)>& inTileRank,
    std::function<int (ij_tuple ij)>& inTileDevice,
    MPI_Comm mpi_comm)
    : BaseBandMatrix<scalar_t>(n, n, kd, kd, inTileNb, inTileRank,
                               inTileDevice, mpi_comm)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;
    this->kl_ = (uplo == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (uplo == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n band matrix, with no tiles allocated,
/// with fixed nb-by-nb tile size and 2D block cyclic distribution.
/// Tiles can be added with tileInsert().
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
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
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    Uplo uplo,
    int64_t n, int64_t kd, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseBandMatrix<scalar_t>(n, n, kd, kd, nb, p, q, mpi_comm)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;
    this->kl_ = (uplo == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (uplo == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// [explicit]
///
/// Conversion from general band matrix
/// creates a shallow copy view of the original matrix.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in,out] orig
///     Original matrix.
///
// todo: should the trans be reflected in kl_ and ku_ values?
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    Uplo uplo, BaseBandMatrix<scalar_t>& orig)
    : BaseBandMatrix<scalar_t>(orig)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;
    this->kl_ = (uplo == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (uplo == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// [explicit]
///
/// Conversion from general matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in] kd
///     Number of sub (if lower) or super (if upper) diagonals within band.
///     kd >= 0.
///
/// @param[in,out] orig
///     Original matrix.
///
// todo: should the trans be reflected in kl_ and ku_ values?
template <typename scalar_t>
BaseTriangularBandMatrix<scalar_t>::BaseTriangularBandMatrix(
    int64_t kd, BaseMatrix<scalar_t>& orig)
    : BaseBandMatrix<scalar_t>(kd, kd, orig)
{
    this->kl_ = (this->uploPhysical() == Uplo::Lower) ? this->kl_ : 0;
    this->ku_ = (this->uploPhysical() == Uplo::Lower) ? 0 : this->ku_;
}

//------------------------------------------------------------------------------
/// Swap contents of band matrices A and B.
template <typename scalar_t>
void swap(BaseTriangularBandMatrix<scalar_t>& A, BaseTriangularBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseBandMatrix<scalar_t>& >(A),
         static_cast< BaseBandMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// @return number of subdiagonals within band.
template <typename scalar_t>
int64_t BaseTriangularBandMatrix<scalar_t>::bandwidth() const
{
    // todo: is this correct? should the trans be accounted for here?
    return (this->uploPhysical() == Uplo::Lower)
            ? this->kl_
            : this->ku_;
}

//------------------------------------------------------------------------------
/// Sets number of subdiagonals within band.
template <typename scalar_t>
void BaseTriangularBandMatrix<scalar_t>::bandwidth(int64_t kd)
{
    // todo: is this correct? should the trans be accounted for here?
    if (this->uploPhysical() == Uplo::Lower)
        this->kl_ = kd;
    else
        this->ku_ = kd;
}

//------------------------------------------------------------------------------
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] target
///     - if target = Devices, inserts tiles on appropriate GPU devices, or
///     - if target = Host, inserts on tiles on CPU host.
///
template <typename scalar_t>
void BaseTriangularBandMatrix<scalar_t>::insertLocalTiles(Target origin)
{
    bool on_devices = (origin == Target::Devices);
    auto upper = this->uplo() == Uplo::Upper;
    int64_t mt = this->mt();
    int64_t nt = this->nt();
    int64_t kdt = ceildiv( this->bandwidth(), this->tileNb(0) );
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

#endif // SLATE_BASE_TRIANGULAR_BAND_MATRIX_HH
