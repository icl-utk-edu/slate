// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::getri from internal::specialization::getri
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel inverse of a general matrix.
/// Generic implementation for any target.
/// @ingroup gesv_specialization
///
/// todo: This routine is in-place and does not support GPUs.
///       There is another one (out-of-place) that does.
///       What if this one is called with Target::Devices?
///       a) execute on CPUs,
///       b) error out (not supported)?
///
template <Target target, typename scalar_t>
void getri(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A, Pivots& pivots,
           int64_t lookahead)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using ReduceList = typename Matrix<scalar_t>::ReduceList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // auto U = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);
    auto L = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, A);

    #pragma omp parallel
    #pragma omp master
    {
        int64_t k = A.nt()-1;
        {
            auto Akk = A.sub(k, k, k, k);
            auto W = Akk.template emptyLike<scalar_t>();
            W.insertLocalTiles(Target::HostTask);

            // Copy A(k, k) to W.
            // todo: Copy L(k, k) to W.
            internal::copy<Target::HostTask>(std::move(Akk), std::move(W));

            // Zero L(k, k).
            if (L.tileIsLocal(k, k)) {
                auto Lkk = L(k, k);
                tzset(scalar_t(0.0), Lkk);
            }

            // send W down col A(0:nt-1, k)
            W.template tileBcast(
                0, 0, A.sub(0, A.nt()-1, k, k), layout);

            auto Wkk = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, W);
            internal::trsm<Target::HostTask>(
                Side::Right,
                scalar_t(1.0), std::move(Wkk), A.sub(0, A.nt()-1, k, k));
        }
        --k;

        for (; k >= 0; --k) {

            auto Lk = A.sub(k, A.nt()-1, k, k);
            auto W = Lk.template emptyLike<scalar_t>();
            W.insertLocalTiles(Target::HostTask);

            // Copy L(:, k) to W.
            internal::copy<Target::HostTask>(std::move(Lk), std::move(W));

            // Zero L(k, k).
            if (L.tileIsLocal(k, k)) {
                auto Lkk = L(k, k);
                tzset(scalar_t(0.0), Lkk);
            }

            // Zero L(k+1:A_nt-1, k).
            for (int64_t i = k+1; i < A.nt(); ++i) {
                if (L.tileIsLocal(i, k)) {
                    L(i, k).set(0.0);
                }
            }

            // send W across A
            BcastList bcast_list_W;
            for (int64_t i = 1; i < W.mt(); ++i) {
                // send W(i) down column A(0:nt-1, k+i)
                bcast_list_W.push_back({i, 0, {A.sub(0, A.nt()-1, k+i, k+i)}});
            }
            W.template listBcast(bcast_list_W, layout);

            // A(:, k) -= A(:, k+1:nt-1) * W
            internal::gemmA<Target::HostTask>(
                scalar_t(-1.0), A.sub(0, A.nt()-1, k+1, A.nt()-1),
                                W.sub(1, W.mt()-1, 0, 0),
                scalar_t(1.0),  A.sub(0, A.nt()-1, k, k), layout);

            // reduce A(0:nt-1, k)
            ReduceList reduce_list_A;
            for (int64_t i = 0; i < A.nt(); ++i) {
                // recude A(i, k) across A(i, k+1:nt-1)
                reduce_list_A.push_back({i, k,
                                          A.sub(i, i, k, k),
                                          {A.sub(i, i, k+1, A.nt()-1)}
                                        });
            }
            A.template listReduce(reduce_list_A, layout);

            // send W(0, 0) down col A(0:nt-1, k)
            W.tileBcast(0, 0, A.sub(0, A.nt()-1, k, k), layout);

            auto Wkk = W.sub(0, 0, 0, 0);
            auto Tkk = TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Wkk);
            internal::trsm<Target::HostTask>(
                Side::Right,
                scalar_t(1.0), std::move(Tkk), A.sub(0, A.nt()-1, k, k));
        }

        // Apply column pivoting.
        for (int64_t j = A.nt()-1; j >= 0; --j) {
            internal::permuteRows<Target::HostTask>(
                Direction::Backward, transpose(A).sub(j, A.nt()-1, 0, A.nt()-1),
                pivots.at(j), Layout::ColMajor);
        }
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gesv_specialization
///
template <Target target, typename scalar_t>
void getri(Matrix<scalar_t>& A, Pivots& pivots,
           Options const& opts)
{
    slate_assert(A.mt() == A.nt());  // square

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    internal::specialization::getri(internal::TargetType<target>(),
                                    A, pivots, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel LU inversion.
///
/// Computes the inverse of a matrix $A$ using the LU factorization $A = L*U$
/// computed by `getrf`.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     On entry, the factors $L$ and $U$ from the factorization $A = P L U$
///     as computed by getrf.
///     On exit, the inverse of the original matrix $A$.
///
/// @param[in] pivots
///     The pivot indices that define the permutation matrix $P$
///     as computed by getrf.
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
/// @ingroup gesv_computational
///
template <typename scalar_t>
void getri(Matrix<scalar_t>& A, Pivots& pivots,
           Options const& opts)
{
    // triangular inversion
    auto U = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);
    trtri(U, opts);

    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            getri<Target::HostTask>(A, pivots, opts);
            break;
        case Target::HostNest:
            getri<Target::HostNest>(A, pivots, opts);
            break;
        case Target::HostBatch:
            getri<Target::HostBatch>(A, pivots, opts);
            break;
        case Target::Devices:
            getri<Target::Devices>(A, pivots, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getri<float>(
    Matrix<float>& A, Pivots& pivots,
    Options const& opts);

template
void getri<double>(
    Matrix<double>& A, Pivots& pivots,
    Options const& opts);

template
void getri< std::complex<float> >(
    Matrix< std::complex<float> >& A, Pivots& pivots,
    Options const& opts);

template
void getri< std::complex<double> >(
    Matrix< std::complex<double> >& A, Pivots& pivots,
    Options const& opts);

} // namespace slate
