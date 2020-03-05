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
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::hegst from internal::specialization::hegst
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel reduction of a complex Hermitian positive-definite
/// generalized eigenvalue problem to the standard form.
/// Generic implementation for any target.
/// @ingroup hegst_specialization
///
template <Target target, typename scalar_t>
void hegst(slate::internal::TargetType<target>,
           int64_t itype, HermitianMatrix<scalar_t> A,
                          HermitianMatrix<scalar_t> B)
{
    if (A.uplo() == Uplo::Upper) {
        A = conj_transpose(A);
        B = conj_transpose(B);
    }

    const int64_t Ant = A.nt();
    const int64_t Bnt = B.nt();

    const scalar_t half = 0.5;
    const scalar_t cone = 1.0;
    const double   rone = 1.0;

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        for (int64_t k = 0; k < Ant; ++k) {
            auto Akk = A.sub(k, k);
            auto Bkk = B.sub(k, k);
            auto Tkk = TriangularMatrix<scalar_t>(Diag::NonUnit, Bkk);

            if (itype == 1) {
                internal::hegst<Target::HostTask>(
                  itype, std::move(Akk), std::move(Bkk));

                if (k+1 <= Ant-1) {
                    auto Asub = A.sub(k+1, Ant-1, k, k);
                    auto Bsub = B.sub(k+1, Bnt-1, k, k);

                    internal::trsm<Target::HostTask>(
                        Side::Right, cone, conj_transpose(Tkk), std::move(Asub));

                    internal::hemm<Target::HostTask>(
                        Side::Right, -half, std::move(Akk),
                                            std::move(Bsub),
                                      cone, std::move(Asub));

                    auto Ak1 = A.sub(k+1, Ant-1);
                    internal::her2k<Target::HostTask>(
                                     -cone, std::move(Asub),
                                            std::move(Bsub),
                                      rone, std::move(Ak1));

                    internal::hemm<Target::HostTask>(
                        Side::Right, -half, std::move(Akk),
                                            std::move(Bsub),
                                      cone, std::move(Asub));

                    auto Bk1 = B.sub(k+1, Bnt-1);
                    auto Tk1 = TriangularMatrix<scalar_t>(Diag::NonUnit, Bk1);
                    slate::trsm<scalar_t>(Side::Left, cone, Tk1, Asub);
                }
            }
            else if (itype == 2 || itype == 3) {
                if (k >= 1) {
                  auto Asub = A.sub(k, k, 0, k-1);
                  auto Bsub = B.sub(k, k, 0, k-1);

                  auto Bk1 = B.sub(0, k-1);
                  auto Tk1 = TriangularMatrix<scalar_t>(Diag::NonUnit, Bk1);
                  slate::trmm<scalar_t>(Side::Right, cone, Tk1, Asub);

                  internal::hemm<Target::HostTask>(
                      Side::Left,   half, std::move(Akk),
                                          std::move(Bsub),
                                    cone, std::move(Asub));

                  auto Ak1 = A.sub(0, k-1);
                  internal::her2k<Target::HostTask>(
                                    cone, conj_transpose(Asub),
                                          conj_transpose(Bsub),
                                    rone, std::move(Ak1));

                  internal::hemm<Target::HostTask>(
                      Side::Left,   half, std::move(Akk),
                                          std::move(Bsub),
                                    cone, std::move(Asub));

                  internal::trmm<Target::HostTask>(
                      Side::Left, cone, conj_transpose(Tkk), std::move(Asub));
                }

                internal::hegst<Target::HostTask>(
                  itype, std::move(Akk), std::move(Bkk));
            }
            else {
                throw std::runtime_error("itype must be: 1, 2, or 3");
            }
        }
    }

    A.tileUpdateAllOrigin();
    A.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup hegst_specialization
///
template <Target target, typename scalar_t>
void hegst(int64_t itype, HermitianMatrix<scalar_t>& A,
                          HermitianMatrix<scalar_t>& B,
           const std::map<Option, Value>& opts)
{
    internal::specialization::hegst(internal::TargetType<target>(),
                                    itype, A, B);
}

//------------------------------------------------------------------------------
/// Distributed parallel reduction of a complex Hermitian positive-definite
/// generalized eigenvalue problem to the standard form.
///
/// Reduces a complex Hermitian positive-definite generalized eigenvalue problem
/// to standard form, as follows:
///
/// itype      |  Problem
/// ---------- | ----------------------
/// itype = 1  |  $A   x = \lambda B x$
/// itype = 2  |  $A B x = \lambda   x$
/// itype = 3  |  $B A x = \lambda   x$
///
/// Before calling `slate::hegst`, you must call `slate::potrf` to compute the
/// Cholesky factorization: $B = L L^H$ or $B = U^H U$.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] itype
///     - itype = 1: Compute $A   x = \lambda B x$;
///     - itype = 2: Compute $A B x = \lambda   x$;
///     - itype = 3: Compute $A B x = \lambda   x$.
///
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix $A$.
///     On exit, the upper or lower triangle is overwritten by the upper or
///     lower triangle of C, as follows:
///     - itype = 1:
///       - A.uplo() = Uplo::Lower: $C = L^(-1) A L^(-H)$;
///       - A.uplo() = Uplo::Upper: $C = U^(-H) A U^(-1)$.
///     - itype = 2:
///       - A.uplo() = Uplo::Lower: $C = L^H A L$;
///       - A.uplo() = Uplo::Upper: $C = U A U^H$.
///     - itype = 3:
///       - A.uplo() = Uplo::Lower: $C = L^H A L$;
///       - A.uplo() = Uplo::Upper: $C = U A U^H$.
///
/// @param[in] B
///     On entry, the n-by-n Hermitian positive definite matrix $A$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// TODO: return value
///
/// @ingroup hegst_computational
///
template <typename scalar_t>
void hegst(int64_t itype, HermitianMatrix<scalar_t>& A,
                          HermitianMatrix<scalar_t>& B,
           const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            hegst<Target::HostTask>(itype, A, B, opts);
            break;
        case Target::HostNest:
            hegst<Target::HostNest>(itype, A, B, opts);
            break;
        case Target::HostBatch:
            hegst<Target::HostBatch>(itype, A, B, opts);
            break;
        case Target::Devices:
            hegst<Target::Devices>(itype, A, B, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hegst<float>(
    int64_t itype, HermitianMatrix<float>& A,
                   HermitianMatrix<float>& B,
    const std::map<Option, Value>& opts);

template
void hegst<double>(
    int64_t itype, HermitianMatrix<double>& A,
                   HermitianMatrix<double>& B,
    const std::map<Option, Value>& opts);

template
void hegst<std::complex<float>>(
    int64_t itype, HermitianMatrix<std::complex<float>>& A,
                   HermitianMatrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void hegst<std::complex<double>>(
    int64_t itype, HermitianMatrix<std::complex<double>>& A,
                   HermitianMatrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

} // namespace slate
