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


template <typename scalar_t>
void gesvd(Matrix<scalar_t>& A,
           std::vector< blas::real_type<scalar_t> >& S,
           const std::map<Option, Value>& opts)
{
    // auto mt = A.mt(), nt = A.nt();
    if (A.m() >= A.n()) {

        auto qr_path =  /* M much greater than N */ false;

        Matrix<scalar_t> Ahat;

        // 0. QR decomposition if needed
        if (qr_path) {
            TriangularFactors<scalar_t> T;
            geqrf(A, T, opts);

            int64_t min_mn = std::min(A.m(), A.n());
            auto R_ = A.slice(0, min_mn-1, 0, min_mn-1);
            auto R = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, R_);

            Ahat = R_.emptyLike();
            Ahat.insertLocalTiles();

            auto Ahat_tr = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, Ahat);
            copy(R, Ahat_tr);
        }
        else {
            Ahat = A;
        }


        // 1. Reduction to bi-diagonal

        // 1.1.1 reduction to band
        slate::TriangularFactors<scalar_t> TU, TV;
        ge2tb(Ahat, TU, TV, opts);

        // 1.1.2 gather general to band
        auto Aband = TriangularBandMatrix<scalar_t>( Uplo::Upper, Diag::NonUnit,
                                                     A.n(), A.tileNb(0), A.tileNb(0),
                                                     1, 1, A.mpiComm());
        Aband.insertLocalTiles();
        Aband.ge2tbGather(Ahat);

        // 1.2.1 triangular band to bidiagonal
        if (A.mpiRank() == 0){
            tb2bd(Aband, opts);
        }

        // 1.2.2 copy triangular band to bi-diagonal (vectors)
        // todo: std::vector< blas::real_type<scalar_t> > E(Aband.n() - 1);
        // todo: S.resize(Aband.n());
        // todo: copy(Aband, S, E);

        // 2. Bi-diagonal SVD (QR iteration)
        if (A.mpiRank() == 0){
            bdsqr(Aband, S, opts);
        }
        // todo: bdsvd(S, E, opts);
    }
    else {
        // todo:
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesvd<float>(
     Matrix<float>& A,
     std::vector<float>& S,
     const std::map<Option, Value>& opts);

template
void gesvd<double>(
     Matrix<double>& A,
     std::vector<double>& S,
     const std::map<Option, Value>& opts);

template
void gesvd< std::complex<float> >(
     Matrix< std::complex<float> >& A,
     std::vector<float>& S,
     const std::map<Option, Value>& opts);

template
void gesvd< std::complex<double> >(
     Matrix< std::complex<double> >& A,
     std::vector<double>& S,
     const std::map<Option, Value>& opts);

} // namespace slate
