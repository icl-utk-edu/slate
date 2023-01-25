// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "work/work.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel reduction of a complex Hermitian positive-definite
/// generalized eigenvalue problem to the standard form.
/// Generic implementation for any target.
/// @ingroup hegv_impl
///
template <Target target, typename scalar_t>
void hegst(
    int64_t itype, HermitianMatrix<scalar_t> A,
                   HermitianMatrix<scalar_t> B,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using real_t = blas::real_type<scalar_t>;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    if (itype != 1 && itype != 2 && itype != 3) {
        throw Exception("itype must be: 1, 2, or 3");
    }
    slate_assert(A.uplo() == B.uplo());
    slate_assert(A.nt() == B.nt());

    if (A.uplo() == Uplo::Upper) {
        A = conjTranspose(A);
        B = conjTranspose(B);
    }

    int64_t nt = A.nt();

    const scalar_t half = 0.5;
    const scalar_t one  = 1.0;
    const real_t r_one  = 1.0;

    const int tag_zero        = 0;
    const int life_factor_two = 2;

    const Layout layout = Layout::ColMajor;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> column_vector(nt);
    uint8_t* column = column_vector.data();

    if (target == Target::Devices) {
        // The work::trsm (itype=1) and work::trmm (itype=2,3)
        // routines use 2 queues (queue 0,1). All other
        // internal::routines here use the default queue (queue 0).
        // So 2 queues need to be allocated.
        A.allocateBatchArrays(0, 2+lookahead); // (batch size, num_queues)
        A.reserveDeviceWorkspace();
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
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
                            -one,  std::move( Asub ),
                                   std::move( Bsub ),
                            r_one, A.sub(k+1, nt-1) );

                        internal::hemm<Target::HostTask>(
                            Side::Right,
                            -half, std::move( Akk  ),
                                   std::move( Bsub ),
                            one,   std::move( Asub ) );

                        auto Bk1  = B.sub(k+1, nt-1);
                        auto TBk1 = TriangularMatrix<scalar_t>(Diag::NonUnit, Bk1);
                        work::trsm<target, scalar_t>(
                            Side::Left,  one,  TBk1,
                                               Asub, column,
                            { {slate::Option::Lookahead, lookahead} });
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
                            one,   conj_transpose( Asub ),
                                   conj_transpose( Bsub ),
                            r_one, A.sub(0, k-1) );

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

} // namespace impl

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
/// @ingroup hegv_computational
///
template <typename scalar_t>
void hegst(
    int64_t itype, HermitianMatrix<scalar_t>& A,
                   HermitianMatrix<scalar_t>& B,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::hegst<Target::HostTask>( itype, A, B, opts );
            break;

        case Target::HostNest:
            impl::hegst<Target::HostNest>( itype, A, B, opts );
            break;

        case Target::HostBatch:
            impl::hegst<Target::HostBatch>( itype, A, B, opts );
            break;

        case Target::Devices:
            impl::hegst<Target::Devices>( itype, A, B, opts );
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
