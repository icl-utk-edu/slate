// Copyright (c) 2022-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GETRF_ADDMOD_HH
#define SLATE_TILE_GETRF_ADDMOD_HH

#include "internal/internal.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"
#include "slate/enums.hh"

#include <blas.hh>
#include <lapack.hh>
#include <vector>

namespace slate {
namespace internal {


template<typename scalar_t>
scalar_t phase(scalar_t z) {
    if constexpr (is_complex<scalar_t>::value) {
        return z == scalar_t(0.0) ? scalar_t(1.0) : z / std::abs(z);
    }
    else {
        return z >= scalar_t(0.0) ? scalar_t(1.0) : scalar_t(-1.0);
    }
}


//------------------------------------------------------------------------------
/// Compute the LU factorization of a tile with additive corrections.
///
/// @param[in,out] A
///     tile to factor
///
/// @param[out] U
///     right singular vectors
///
/// @param[out] singular_values
///     modified singular values
///
/// @param[out] modifications
///     amount the singular values were modified by
///
/// @param[out] modified_indices
///     the singular values that correspond to the elements in modifications
///
///
/// @param[in] ib
///     blocking factor used in the factorization
///
/// @ingroup gesv_tile
///
template <BlockFactor factorType, typename scalar_t>
void getrf_addmod(Tile<scalar_t> A,
                  Tile<scalar_t> U,
                  Tile<scalar_t> VT,
                  std::vector< blas::real_type<scalar_t> >& singular_values,
                  std::vector<scalar_t>& modifications,
                  std::vector<int64_t>& modified_indices,
                  blas::real_type<scalar_t> mod_tol,
                  int64_t ib)
{
    slate_assert(A.layout() == Layout::ColMajor);

    using real_t = blas::real_type<scalar_t>;

    const scalar_t one = 1.0;
    const scalar_t zero = 0.0;
    int64_t mb = A.mb();
    int64_t nb = A.nb();
    int64_t diag_len = std::min(nb, mb);

    scalar_t* A_data = A.data();
    int64_t lda = A.stride();
    scalar_t* U_data = U.data();
    int64_t ldu = U.stride();
    scalar_t* VT_data = VT.data();
    int64_t ldvt = VT.stride();

    std::vector<scalar_t> workspace_vect (ib * std::max(INT64_C(2), std::max(mb, nb)));
    scalar_t* workspace = workspace_vect.data();

    // Used for pivoting
    std::vector<int64_t> iworkspace_vect (2*ib);
    int64_t* iworkspace = iworkspace_vect.data();

    using singular_t = std::conditional_t<factorType == BlockFactor::SVD,
                                          real_t, scalar_t>;

    // Loop over ib-wide blocks..
    for (int64_t k = 0; k < diag_len; k += ib) {
        int64_t kb = std::min(diag_len-k, ib);

        singular_t* s_vals;
        int64_t s_inc;

        scalar_t*  Akk = & A_data[k + k*lda];
        scalar_t*  Ukk = & U_data[k + k*ldu];
        [[maybe_unused]] scalar_t* VTkk = &VT_data[k + k*ldvt];

        if constexpr (factorType == BlockFactor::SVD) {
            // Compute SVD of diagonal block
            //lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, kb, kb,
            //              Akk, lda,
            //              &singular_values[k],
            //              Ukk, ldu,
            //              VTkk, ldvt);
            lapack::gesdd(lapack::Job::AllVec, kb, kb,
                          Akk, lda,
                          &singular_values[k],
                          Ukk, ldu,
                          VTkk, ldvt);
            s_vals = &singular_values[k];
            s_inc = 1;
        }
        else if constexpr (factorType == BlockFactor::QLP) {
            // iworkspace must be zeroed out for geqp3
            for (int64_t i = 0; i < kb; ++i) {
                iworkspace[i] = 0;
            }
            lapack::geqp3(kb, kb, Akk, lda, iworkspace, workspace);
            // copy and clear strictly lower-triangular part of Akk
            for (int64_t j = 0; j < kb; ++j) {
                for (int64_t i = j+1; i < kb; ++i) {
                    Ukk[i + j*ldu] = Akk[i + j*lda];
                    Akk[i + j*lda] = zero;
                }
            }
            // use the non-pivoted LQ since LAPACK doesn't have a pivoted one
            lapack::gelqf(kb, kb, Akk, lda, workspace+kb);

            // build left unitary matrix
            lapack::ungqr(kb, kb, kb, Ukk, ldu, workspace);

            // build right unitary matrix
            lapack::lacpy(lapack::MatrixType::Upper, kb, kb,
                          Akk, lda, VTkk, ldvt);
            lapack::unglq(kb, kb, kb, VTkk, ldvt, workspace+kb);
            // geqp3 provides pivots as the final order, requiring an out-of-place permutation
            for (int64_t h = 0; h < kb; ++h) {
                blas::copy(kb, VTkk+h*ldvt, 1, workspace+(iworkspace[h]-1)*kb, 1);
            }
            lapack::lacpy(lapack::MatrixType::General, kb, kb,
                          workspace, kb, VTkk, ldvt);

            s_vals = Akk;
            s_inc = lda+1;
        }
        else if constexpr (factorType == BlockFactor::QRCP) {
            // iworkspace must be zeroed out for geqp3
            for (int64_t i = 0; i < kb; ++i) {
                iworkspace[i] = 0;
            }
            lapack::geqp3(kb, kb, Akk, lda, iworkspace, workspace);
            lapack::lacpy(lapack::MatrixType::Lower, kb, kb,
                          Akk, lda, Ukk, ldu);
            lapack::ungqr(kb, kb, kb, Ukk, ldu, workspace);

            // TODO just store the pivots instead of building a permutation matrix
            lapack::laset(lapack::MatrixType::General, kb, kb, zero, zero, VTkk, ldvt);
            for (int64_t h = 0; h < kb; ++h) {
                VTkk[h + (iworkspace[h]-1)*ldvt] = one;
            }

            s_vals = Akk;
            s_inc = lda+1;
        }
        else if constexpr (factorType == BlockFactor::QR) {
            lapack::geqrf(kb, kb, Akk, lda, workspace);
            lapack::lacpy(lapack::MatrixType::Lower, kb, kb,
                          Akk, lda, Ukk, ldu);
            lapack::ungqr(kb, kb, kb, Ukk, ldu, workspace);

            s_vals = Akk;
            s_inc = lda+1;
        }
        else {
            // static_assert must depend on factorType
            static_assert(factorType == BlockFactor::SVD, "Not yet implemented");
        }

        // Compute modifications
        // TODO consider optimizing when s_vals guaranteed to be ordered or non-negative
        for (int64_t i = 0; i < kb; ++i) {
            if (std::abs(s_vals[i*s_inc]) <= mod_tol) {
                singular_t target = phase(s_vals[i*s_inc])*mod_tol;
                singular_t mod = target - s_vals[i*s_inc];
                s_vals[i*s_inc] = target;

                modifications.push_back(mod);
                modified_indices.push_back(i);
            }
        }

        // block-column update: A := A R^-1
        if (k+kb < mb) {
            scalar_t* A_panel = &A_data[k+kb + k*lda];
            int64_t ldwork = mb-k-kb;

            if constexpr (factorType == BlockFactor::SVD) {
                blas::gemm(Layout::ColMajor,
                           Op::NoTrans, Op::ConjTrans,
                           mb-k-kb, kb, kb,
                           one,  A_panel, lda,
                                 VTkk, ldvt,
                           zero, workspace, ldwork);

                for (int64_t j = 0; j < kb; ++j) {
                    scalar_t Sj = singular_values[k+j];
                    for (int64_t i = 0; i < mb-k-kb; ++i) {
                        A_panel[i + j*lda] = workspace[i + j*ldwork] / Sj;
                    }
                }
            }
            else if constexpr (factorType == BlockFactor::QLP
                               || factorType == BlockFactor::QRCP) {
                // TODO redo this as a simple permutation for QRCP
                blas::gemm(Layout::ColMajor,
                           Op::NoTrans, Op::ConjTrans,
                           mb-k-kb, kb, kb,
                           one,  A_panel, lda,
                                 VTkk, ldvt,
                           zero, workspace, ldwork);
                lapack::lacpy(lapack::MatrixType::General, mb-k-kb, kb, workspace, ldwork, A_panel, lda);

                auto uplo = factorType == BlockFactor::QLP ? Uplo::Lower : Uplo::Upper;
                blas::trsm(Layout::ColMajor, Side::Right, uplo,
                           Op::NoTrans, Diag::NonUnit, mb-k-kb, kb,
                           one, Akk, lda, A_panel, lda);
            }
            else if constexpr (factorType == BlockFactor::QR) {
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper,
                           Op::NoTrans, Diag::NonUnit, mb-k-kb, kb,
                           one, Akk, lda, A_panel, lda);
            }
            else {
                // static_assert must depend on factorType
                static_assert(factorType == BlockFactor::SVD, "Not yet implemented");
            }
        }

        // block-row update: A := L^-1 A
        if (k+kb < nb) {
            scalar_t* A_panel = &A_data[k + (k+kb)*lda];

            if constexpr (factorType == BlockFactor::SVD
                          || factorType == BlockFactor::QLP
                          || factorType == BlockFactor::QRCP
                          || factorType == BlockFactor::QR) {
                for (int64_t j = 0; j < nb-k-kb; ++j) {
                    for (int64_t i = 0; i < kb; ++i) {
                        workspace[i + j*kb] = A_panel[i + j*lda];
                    }
                }

                blas::gemm(Layout::ColMajor,
                           Op::ConjTrans, Op::NoTrans,
                           kb, nb-k-kb, kb,
                           one,  Ukk, ldu,
                                 workspace, kb,
                           zero, A_panel, lda);
            }
            else {
                // static_assert must depend on factorType
                static_assert(factorType == BlockFactor::SVD, "Not yet implemented");
            }
        }

        // trailing matrix update
        if (k+kb < mb && k+kb < nb) {
            blas::gemm(Layout::ColMajor,
                       Op::NoTrans, Op::NoTrans,
                       mb-k-kb, nb-k-kb, kb,
                       -one, &A_data[k+kb + (k   )*lda], lda,
                             &A_data[k    + (k+kb)*lda], lda,
                       one,  &A_data[k+kb + (k+kb)*lda], lda);
        }
    }
}

} // namespace internal
} // namespace slate

#endif // SLATE_TILE_GETRF_ADDMOD_HH
