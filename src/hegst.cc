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
#include "slate/HermitianMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::hegst from internal::specialization::hegst
// namespace internal {
// namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup posv_specialization
///
template <typename scalar_t>
void hegst(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    const std::map<Option, Value>& opts)
{
    if (itype == 1) {
        if (A.uplo() == Uplo::Lower) { // C = L^-1 * A * L^(-H)
            for (int64_t k = 0; k < A.nt(); ++k) {
                internal::hegst<Target::HostTask>(
                    itype, A.sub(k, k), B.sub(k, k));

                if (k+1 <= A.nt()-1) {
                    auto Bkk = B.sub(k, k);
                    auto Tkk = TriangularMatrix<scalar_t>(Diag::NonUnit, Bkk);
                    internal::trsm<Target::HostTask>(
                        Side::Right,
                        scalar_t(1.0), conj_transpose(Tkk),
                        A.sub(k+1, A.nt()-1, k, k), 1);

                    internal::hemm<Target::HostTask>(
                        Side::Right,
                        scalar_t(-0.5), A.sub(k, k),
                                        B.sub(k+1, B.nt()-1, k, k),
                        scalar_t( 1.0), A.sub(k+1, A.nt()-1, k, k));

                    auto Ak = A.sub(k+1, A.nt()-1, k+1, k+1);
                    auto Hk = HermitianMatrix<scalar_t>(Uplo::Lower, Ak);
                    internal::her2k<Target::HostTask>(
                        scalar_t(-1.0), A.sub(k+1, A.nt()-1, k, k),
                                        B.sub(k+1, B.nt()-1, k, k),
                                  1.0,  std::move(Hk));

                    internal::hemm<Target::HostTask>(
                        Side::Right,
                        scalar_t(-0.5), A.sub(k, k),
                                        B.sub(k+1, B.nt()-1, k, k),
                        scalar_t( 1.0), A.sub(k+1, A.nt()-1, k, k));

                    auto Bk = B.sub(k+1, k+1);
                    auto Tk = TriangularMatrix<scalar_t>(Diag::NonUnit, Bk);
                    internal::trsm<Target::HostTask>(
                        Side::Left,
                        scalar_t(1.0), std::move(Tk),
                        A.sub(k+1, A.nt()-1, k, k), 1);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hegst<float>(
    int64_t itype,
    HermitianMatrix<float>& A,
    HermitianMatrix<float>& B,
    const std::map<Option, Value>& opts);

template
void hegst<double>(
    int64_t itype,
    HermitianMatrix<double>& A,
    HermitianMatrix<double>& B,
    const std::map<Option, Value>& opts);

template
void hegst<std::complex<float>>(
    int64_t itype,
    HermitianMatrix<std::complex<float>>& A,
    HermitianMatrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void hegst<std::complex<double>>(
    int64_t itype,
    HermitianMatrix<std::complex<double>>& A,
    HermitianMatrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

} // namespace slate
