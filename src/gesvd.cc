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
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Note A is passed by value, so we can transpose if needed
/// without affecting caller.
///
template <typename scalar_t>
void gesvd(Matrix<scalar_t> A,
           std::vector< blas::real_type<scalar_t> >& S,
           Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    using std::swap;

    scalar_t zero = 0;

    int64_t m = A.m();
    int64_t n = A.n();

    bool flip = m < n; // Flip for fat matrix.
    if (flip) {
        slate_not_implemented("m < n not yet supported");
        swap(m, n);
        A = conjTranspose(A);
    }

    // Scale matrix to allowable range, if necessary.
    // todo

    // 0. If m >> n, use QR factorization to reduce to square.
    // Theoretical thresholds based on flops:
    // m >=  5/3 n for no vectors,
    // m >= 16/9 n for QR iteration with vectors,
    // m >= 10/3 n for Divide & Conquer with vectors.
    // Different in practice because stages have different flop rates.
    double threshold = 5/3.;
    bool qr_path = m > threshold*n;
    Matrix<scalar_t> Ahat;
    TriangularFactors<scalar_t> TQ;
    if (qr_path) {
        geqrf(A, TQ, opts);

        auto R_ = A.slice(0, n-1, 0, n-1);
        TriangularMatrix<scalar_t> R(Uplo::Upper, Diag::NonUnit, R_);

        Ahat = R_.emptyLike();
        Ahat.insertLocalTiles();
        set(zero, Ahat);  // todo: only lower

        TriangularMatrix<scalar_t> Ahat_tr(Uplo::Upper, Diag::NonUnit, Ahat);
        copy(R, Ahat_tr);
    }
    else {
        Ahat = A;
    }

    // 1. Reduce to band form.
    TriangularFactors<scalar_t> TU, TV;
    ge2tb(Ahat, TU, TV, opts);

    // Copy band.
    // Currently, gathers band matrix to rank 0.
    TriangularBandMatrix<scalar_t> Aband(Uplo::Upper, Diag::NonUnit,
                                         n, A.tileNb(0), A.tileNb(0),
                                         1, 1, A.mpiComm());
    Aband.insertLocalTiles();
    Aband.ge2tbGather(Ahat);

    // Currently, hb2st and sterf are run on a single node.
    slate::Matrix<scalar_t> U;
    slate::Matrix<scalar_t> VT;

    S.resize(n);
    if (A.mpiRank() == 0) {
        // 2. Reduce band to bi-diagonal.
        tb2bd(Aband, opts);

        // Copy diagonal and super-diagonal to vectors.
        std::vector<real_t> E(n - 1);
        internal::copytb2bd(Aband, S, E);

        // 3. Bi-diagonal SVD solver.
        // QR iteration
        bdsqr<scalar_t>(slate::Job::NoVec, slate::Job::NoVec, S, E, U, VT, opts);
    }

    // If matrix was scaled, then rescale singular values appropriately.
    // todo

    // todo: bcast S.

    if (qr_path) {
        // When initial QR was used.
        // U = Q*U;
    }

    if (flip) {
        // todo: swap(U, V);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesvd<float>(
     Matrix<float> A,
     std::vector<float>& S,
     Options const& opts);

template
void gesvd<double>(
     Matrix<double> A,
     std::vector<double>& S,
     Options const& opts);

template
void gesvd< std::complex<float> >(
     Matrix< std::complex<float> > A,
     std::vector<float>& S,
     Options const& opts);

template
void gesvd< std::complex<double> >(
     Matrix< std::complex<double> > A,
     std::vector<double>& S,
     Options const& opts);

} // namespace slate
