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

#include "slate_BandMatrix.hh"
#include "slate_TriangularMatrix.hh"
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
/// General banded, triangular, n-by-n, distributed, tiled matrices.
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

    // todo: specialize for band
    // int64_t getMaxHostTiles();
    // int64_t getMaxDeviceTiles(int device);
    // void allocateBatchArrays();
    // void reserveHostWorkspace();
    // void reserveDeviceWorkspace();
    // todo: void moveAllToOrigin();
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
/// Constructor creates an n-by-n band matrix, with no tiles allocated.
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
      kd_(uplo == Uplo::Lower ? orig.lowerBandwidth() : orig.upperBandwidth())
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

} // namespace slate

#endif // SLATE_TRIANGULAR_BAND_MATRIX_HH
