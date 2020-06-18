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

#include "slate/slate.hh"
#include "aux/Debug.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename scalar_t>
void heev(lapack::Job jobz,
          HermitianMatrix<scalar_t>& A,
          std::vector<blas::real_type<scalar_t>>& W,
          Matrix<scalar_t>& Z,
          Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;

    int64_t n = A.n();
    bool wantz = (jobz == Job::Vec);

    // MPI_Status status;
    int mpi_rank;

    // Scale matrix to allowable range, if necessary.
    // todo

    // 1. Reduce to band form.
    TriangularFactors<scalar_t> T;
    he2hb(A, T, opts);

    // Copy band.
    // Currently, gathers band matrix to rank 0.
    HermitianBandMatrix<scalar_t> Aband(A.uplo(), n, A.tileNb(0), A.tileNb(0),
                                        1, 1, A.mpiComm());
    Aband.insertLocalTiles();
    Aband.he2hbGather(A);

    // Currently, hb2st and sterf are run on a single node.
    W.resize(n);
    std::vector<real_t> E(n - 1);
    MPI_Comm_rank(A.mpiComm(), &mpi_rank);

    if (mpi_rank == 0) {
        // 2. Reduce band to symmetric tri-diagonal.
        hb2st(Aband, opts);

        // Copy diagonal and super-diagonal to vectors.
        internal::copyhb2st(Aband, W, E);
    }

    // 3. Tri-diagonal eigenvalue solver.
    if (wantz) {
        // Bcast the W and E vectors
        MPI_Bcast( &W[0], n, mpi_type<blas::real_type<scalar_t>>::value, 0, A.mpiComm() );
        MPI_Bcast( &E[0], n-1, mpi_type<blas::real_type<scalar_t>>::value, 0, A.mpiComm() );
        // QR iteration
        steqr2(jobz, W, E, Z);
    }
    else {
        if (mpi_rank == 0) {
            // QR iteration
            sterf<real_t>(W, E, opts);
            // Bcast the vectors of the eigenvalues W
        }
        MPI_Bcast( &W[0], n, mpi_type<blas::real_type<scalar_t>>::value, 0, A.mpiComm() );
    }
    // todo: If matrix was scaled, then rescale eigenvalues appropriately.
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void heev<float>(
    lapack::Job jobz,
    HermitianMatrix<float>& A,
    std::vector<float>& W,
    Matrix<float>& Z,
    Options const& opts);

template
void heev<double>(
    lapack::Job jobz,
    HermitianMatrix<double>& A,
    std::vector<double>& W,
    Matrix<double>& Z,
    Options const& opts);

template
void heev<std::complex<float>>(
    lapack::Job jobz,
    HermitianMatrix<std::complex<float>>& A,
    std::vector<float>& W,
    Matrix<std::complex<float>>& Z,
    Options const& opts);

template
void heev<std::complex<double>>(
    lapack::Job jobz,
    HermitianMatrix<std::complex<double>>& A,
    std::vector<double>& W,
    Matrix<std::complex<double>>& Z,
    Options const& opts);

} // namespace slate
