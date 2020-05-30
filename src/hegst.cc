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
#include "work/work.hh"

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
                          HermitianMatrix<scalar_t> B,
           int64_t lookahead)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;

    if (itype != 1 && itype != 2 && itype != 3) {
        throw Exception("itype must be: 1, 2, or 3");
    }
    slate_assert(A.uplo() == B.uplo());
    slate_assert(A.nt() == B.nt());

    if (A.uplo() == Uplo::Upper) {
        A = conjTranspose(A);
        B = conjTranspose(B);
    }

    const int64_t nt = A.nt();

    const scalar_t half = 0.5;
    const scalar_t one  = 1.0;
    const double   rone = 1.0;

    const int tag_zero        = 0;
    const int life_factor_two = 2;

    const Layout layout = Layout::ColMajor;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> column_vector(nt);
    uint8_t* column = column_vector.data();

    if (target == Target::Devices) {
        const int64_t batch_size_zero = 0;
        const int64_t num_arrays_two  = 2; // Number of kernels without lookahead
        if (itype == 1) {
            A.allocateBatchArrays(batch_size_zero, num_arrays_two);
        }
        else {
            A.allocateBatchArrays();
        }
        A.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        for (int64_t k = 0; k < nt; ++k) {
            auto Akk  = A.sub(k, k);
            auto Bkk  = B.sub(k, k);
            auto TBkk = TriangularMatrix<scalar_t>(Diag::NonUnit, Bkk);

            if (itype == 1) {
                #pragma omp task depend(inout:column[k])
                {
                    internal::hegst<Target::HostTask>(
                        itype, std::move(Akk),
                               std::move(Bkk));
                }

                if (k+1 <= nt-1) {
                    auto Asub = A.sub(k+1, nt-1, k, k);
                    auto Bsub = B.sub(k+1, nt-1, k, k);

                    #pragma omp task depend(inout:column[k])
                    {
                        B.template tileBcast<target>(k, k, Asub, layout);

                        internal::trsm<target>(
                            Side::Right,  one,  conjTranspose(TBkk),
                                                std::move(Asub));
                    }

                    #pragma omp task depend(inout:column[k])
                    {
                        A.tileBcast(
                            k, k, Asub, layout, tag_zero, life_factor_two);

                        BcastList bcast_list;
                        for (int64_t i = k+1; i < nt; ++i) {
                            bcast_list.push_back({i, k, {A.sub(i, i, k+1, i),
                                                         A.sub(i, nt-1, i, i)}});
                        }
                        B.template listBcast<target>(
                            bcast_list, layout, tag_zero, life_factor_two);
                    }

                    #pragma omp task depend(in:column[k]) \
                                     depend(inout:column[k+1]) \
                                     depend(inout:column[nt-1])
                    {
                        internal::hemm<Target::HostTask>(
                            Side::Right, -half, std::move(Akk),
                                                std::move(Bsub),
                                          one,  std::move(Asub));

                        BcastList bcast_list;
                        for (int64_t i = k+1; i < nt; ++i) {
                            bcast_list.push_back({i, k, {A.sub(i, i, k+1, i),
                                                         A.sub(i, nt-1, i, i)}});
                        }
                        A.template listBcast<target>(bcast_list, layout);

                        internal::her2k<target>(
                                         -one,  std::move(Asub),
                                                std::move(Bsub),
                                          rone, A.sub(k+1, nt-1));

                        internal::hemm<Target::HostTask>(
                            Side::Right, -half, std::move(Akk),
                                                std::move(Bsub),
                                          one,  std::move(Asub));

                        auto Bk1  = B.sub(k+1, nt-1);
                        auto TBk1 = TriangularMatrix<scalar_t>(Diag::NonUnit, Bk1);
                        work::trsm<target, scalar_t>(
                            Side::Left,  one,  TBk1,
                                               Asub, column, lookahead);
                    }
                }
            }
            else { //if (itype == 2 || itype == 3)
                if (k >= 1) {
                    auto Asub = A.sub(k, k, 0, k-1);
                    auto Bsub = B.sub(k, k, 0, k-1);

                    #pragma omp task depend(inout:column[0])
                    {
                        A.tileBcast(
                            k, k, Asub, layout, tag_zero, life_factor_two);

                        BcastList bcast_list;
                        for (int64_t i = 0; i < k; ++i) {
                            bcast_list.push_back({k, i, {A.sub(i, k-1, i, i),
                                                         A.sub(i, i,   0, i)}});
                        }
                        B.template listBcast<target>(
                            bcast_list, layout, tag_zero, life_factor_two);

                        B.template tileBcast<target>(k, k, Asub, layout);
                    }

                    #pragma omp task depend(inout:column[0])
                    {
                        auto Bk1  = B.sub(0, k-1);
                        auto TBk1 = TriangularMatrix<scalar_t>(Diag::NonUnit, Bk1);
                        work::trmm<target, scalar_t>(
                            Side::Right, one,  TBk1,
                                               Asub, column, column, lookahead);

                        internal::hemm<Target::HostTask>(
                            Side::Left,  half, std::move(Akk),
                                               std::move(Bsub),
                                         one,  std::move(Asub));

                        BcastList bcast_list;
                        for (int64_t i = 0; i < k; ++i) {
                            bcast_list.push_back({k, i, {A.sub(i, k-1, i, i),
                                                         A.sub(i, i,   0, i)}});
                        }
                        A.template listBcast<target>(bcast_list, layout);

                        internal::her2k<Target::HostTask>(
                                        one,  conjTranspose(Asub),
                                              conjTranspose(Bsub),
                                        rone, A.sub(0, k-1));

                        internal::hemm<Target::HostTask>(
                            Side::Left, half, std::move(Akk),
                                              std::move(Bsub),
                                        one,  std::move(Asub));

                        internal::trmm<Target::HostTask>(
                            Side::Left, one,  conjTranspose(TBkk),
                                              std::move(Asub));
                    }
                }

                #pragma omp task depend(inout:column[0])
                {
                    internal::hegst<Target::HostTask>(
                      itype,  std::move(Akk),
                              std::move(Bkk));
                }
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
           Options const& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range&) {
        lookahead = 1;
    }

    internal::specialization::hegst(internal::TargetType<target>(),
                                    itype, A, B, lookahead);
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
///     - itype = 3: Compute $B A x = \lambda   x$.
///
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix $A$.
///     On exit, the upper or lower triangle is overwritten by the upper or
///     lower triangle of C, as follows:
///     - itype = 1:
///       - A.uplo() = Uplo::Lower: $C = L^{-1} A L^{-H}$;
///       - A.uplo() = Uplo::Upper: $C = U^{-H} A U^{-1}$.
///     - itype = 2 or 3:
///       - A.uplo() = Uplo::Lower: $C = L^H A L$;
///       - A.uplo() = Uplo::Upper: $C = U A U^H$.
///
/// @param[in] B
///     On entry, the triangular factor from the Cholesky factorization of $B$,
///     as returned by |slate::potrf|.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
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
           Options const& opts)
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
    Options const& opts);

template
void hegst<double>(
    int64_t itype, HermitianMatrix<double>& A,
                   HermitianMatrix<double>& B,
    Options const& opts);

template
void hegst<std::complex<float>>(
    int64_t itype, HermitianMatrix<std::complex<float>>& A,
                   HermitianMatrix<std::complex<float>>& B,
    Options const& opts);

template
void hegst<std::complex<double>>(
    int64_t itype, HermitianMatrix<std::complex<double>>& A,
                   HermitianMatrix<std::complex<double>>& B,
    Options const& opts);

} // namespace slate
