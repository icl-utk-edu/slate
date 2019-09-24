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

#ifndef SLATE_HERMITIAN_BAND_MATRIX_HH
#define SLATE_HERMITIAN_BAND_MATRIX_HH

#include "slate/BandMatrix.hh"
#include "slate/HermitianMatrix.hh"
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
/// Hermitian banded, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class HermitianBandMatrix: public HermitianMatrix<scalar_t> {
public:
    // constructors
    HermitianBandMatrix();

    HermitianBandMatrix(
        Uplo uplo,
        int64_t n, int64_t kd,
        int64_t nb, int p, int q, MPI_Comm mpi_comm);

public:
    template <typename T>
    friend void swap(HermitianBandMatrix<T>& A, HermitianBandMatrix<T>& B);

    int64_t bandwidth() const;
    void    bandwidth(int64_t kd);
    void    insertLocalTiles(Target origin=Target::Host);

protected:
    int64_t kd_;
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty Hermitian band matrix with bandwidth = 0.
template <typename scalar_t>
HermitianBandMatrix<scalar_t>::HermitianBandMatrix()
    : HermitianMatrix<scalar_t>(),
      kd_(0)
{}

//------------------------------------------------------------------------------
/// Constructor creates an n-by-n Hermitian band matrix, with no tiles allocated.
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
HermitianBandMatrix<scalar_t>::HermitianBandMatrix(
    Uplo uplo,
    int64_t n, int64_t kd, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : HermitianMatrix<scalar_t>(uplo, n, nb, p, q, mpi_comm),
      kd_(kd)
{}

//------------------------------------------------------------------------------
/// Swap contents of Hermitian band matrices A and B.
template <typename scalar_t>
void swap(HermitianBandMatrix<scalar_t>& A, HermitianBandMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< HermitianMatrix<scalar_t>& >(A),
         static_cast< HermitianMatrix<scalar_t>& >(B));
    swap(A.kd_, B.kd_);
}

//------------------------------------------------------------------------------
/// @return number of subdiagonals within band.
template <typename scalar_t>
int64_t HermitianBandMatrix<scalar_t>::bandwidth() const
{
    return kd_;
}

//------------------------------------------------------------------------------
/// Sets number of subdiagonals within band.
template <typename scalar_t>
void HermitianBandMatrix<scalar_t>::bandwidth(int64_t kd)
{
    kd_  = kd;
}

//------------------------------------------------------------------------------
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] target
///     - if target = Devices, inserts tiles on appropriate GPU devices, or
///     - if target = Host, inserts on tiles on CPU host.
///
template <typename scalar_t>
void HermitianBandMatrix<scalar_t>::insertLocalTiles(Target origin)
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

#endif // SLATE_HERMITIAN_BAND_MATRIX_HH
