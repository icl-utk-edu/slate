// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian indefinite $LTL^T$ solve.
///
/// Solves a system of linear equations $A X = B$ with a
/// Hermitian matrix $A$ using the factorization $A = U^H T U$ or
/// $A = L T L^H$ computed by hetrf.
///
/// Complexity (in real): $2 n^{2} r$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     Details of the factors $U$ or $L$ as computed by hetrf.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
///
/// @param[out] pivots
///     Details of the interchanges applied to $A$ as computed by hetrf.
///
/// @param[out] T
///     Details of the LU factorization of the band matrix as computed by hetrf.
///
/// @param[out] pivots2
///     Details of the interchanges applied to $T$ as computed by hetrf.
///
/// @param[in,out] B
///     On entry, the n-by-nrhs right hand side matrix $B$.
///     On exit, if return value = 0, the n-by-nrhs solution matrix $X$.
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
/// @ingroup hesv_computational
///
template <typename scalar_t>
void hetrs(HermitianMatrix<scalar_t>& A, Pivots& pivots,
                BandMatrix<scalar_t>& T, Pivots& pivots2,
                    Matrix<scalar_t>& B,
           Options const& opts)
{
    // Constants
    const scalar_t one  = 1;

    // assert(A.mt() == A.nt());
    assert(B.mt() == A.mt());

    auto A_ = A;  // local shallow copy to transpose

    // if upper, change to lower
    if (A_.uplo() == Uplo::Upper)
        A_ = conjTranspose(A_);

    int64_t A_nt = A_.nt();
    int64_t A_mt = A_.mt();
    int64_t B_nt = B.nt();
    int64_t B_mt = B.mt();

    if (A_nt > 1) {
        // pivot right-hand-sides
        for (int64_t k = 1; k < B.mt(); ++k) {
            // swap rows in B(k:mt-1, 0:nt-1)
            internal::permuteRows<Target::HostTask>(
                Direction::Forward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                pivots.at(k), Layout::ColMajor);
        }

        // forward substitution with L from Aasen's
        auto Lkk = TriangularMatrix<scalar_t>( Diag::NonUnit, A_, 1, A_mt-1, 0, A_nt-2 );
        auto Bkk = B.sub(1, B_mt-1, 0, B_nt-1);
        trsm(Side::Left, one, Lkk, Bkk, opts);
    }

    // band solve
    gbtrs(T, pivots2, B, opts);

    if (A_nt > 1) {
        // backward substitution with L^T from Aasen's
        auto Lkk = TriangularMatrix<scalar_t>( Diag::NonUnit, A_, 1, A_mt-1, 0, A_nt-2 );
        auto Bkk = B.sub(1, B_mt-1, 0, B_nt-1);
        Lkk = conjTranspose(Lkk);
        trsm(Side::Left, one, Lkk, Bkk, opts);

        // pivot right-hand-sides
        for (int64_t k = B.mt()-1; k > 0; --k) {
            // swap rows in B(k:mt-1, 0:nt-1)
            internal::permuteRows<Target::HostTask>(
                Direction::Backward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                pivots.at(k), Layout::ColMajor);
        }
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hetrs<float>(
    HermitianMatrix<float>& A, Pivots& pivots,
         BandMatrix<float>& T, Pivots& pivots2,
             Matrix<float>& B,
    Options const& opts);

template
void hetrs<double>(
    HermitianMatrix<double>& A, Pivots& pivots,
         BandMatrix<double>& T, Pivots& pivots2,
             Matrix<double>& B,
    Options const& opts);

template
void hetrs< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A, Pivots& pivots,
         BandMatrix< std::complex<float> >& T, Pivots& pivots2,
             Matrix< std::complex<float> >& B,
    Options const& opts);

template
void hetrs< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A, Pivots& pivots,
         BandMatrix< std::complex<double> >& T, Pivots& pivots2,
             Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
