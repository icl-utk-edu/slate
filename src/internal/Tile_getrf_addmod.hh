// Copyright (c) 2022-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_GETRF_ADDMOD_HH
#define SLATE_TILE_GETRF_ADDMOD_HH

#include "internal/internal.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"

#include <blas.hh>
#include <lapack.hh>
#include <vector>

namespace slate {
namespace internal {
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
template <typename scalar_t>
void getrf_addmod(Tile<scalar_t> A,
                  Tile<scalar_t> U,
                  Tile<scalar_t> VT,
                  std::vector< blas::real_type<scalar_t> >& singular_values,
                  std::vector< blas::real_type<scalar_t> >& modifications,
                  std::vector<int64_t>& modified_indices,
                  blas::real_type<scalar_t> mod_tol,
                  int64_t ib)
{
    slate_assert(A.layout() == Layout::ColMajor);

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

    std::vector<scalar_t> workspace_vect (ib * std::max(mb, nb));
    scalar_t* workspace = workspace_vect.data();

    // Loop over ib-wide blocks..
    for (int64_t k = 0; k < diag_len; k += ib) {
        int64_t kb = std::min(diag_len-k, ib);

        // Compute SVD of diagonal block
        //lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, kb, kb,
        //              &A_data[k + k*lda], lda,
        //              &singular_values[k],
        //              &U_data[k + k*ldu], ldu,
        //              VT_data[k + k*ldvt], ldvt);
        lapack::gesdd(lapack::Job::AllVec, kb, kb,
                      &A_data[k + k*lda], lda,
                      &singular_values[k],
                      &U_data[k + k*ldu], ldu,
                      &VT_data[k + k*ldvt], ldvt);

        // Compute modifications
        {
            // Note: gesvd puts singular values in decreasing order;
            // singular values are positive reals
            int64_t i = 0;
            while (i < kb && singular_values[k+i] >= mod_tol) {
                i++;
            }
            if (i < kb) {
                int64_t prev_mods = modifications.size();
                int64_t new_mods = kb-i;
                modifications.resize(prev_mods + new_mods);
                modified_indices.resize(prev_mods + new_mods);

                for (int64_t j = 0; j < new_mods; ++j, ++i) {
                    modifications[prev_mods+j] = mod_tol - singular_values[k+i];
                    modified_indices[prev_mods+j] = k+i;
                    singular_values[k+i] = mod_tol;
                }
            }
        }

        // block-column update: A := A V^H^H S^-1
        if (k+kb < mb) {
            int64_t ldwork = mb-k-kb;

            blas::gemm(Layout::ColMajor,
                       Op::NoTrans, Op::ConjTrans,
                       mb-k-kb, kb, kb,
                       one,  &A_data[k+kb + k*lda], lda,
                             &VT_data[k + k*ldvt], ldvt,
                       zero, &workspace[0], ldwork);

            for (int64_t j = 0; j < kb; ++j) {
                scalar_t Sj = singular_values[k+j];
                for (int64_t i = 0; i < mb-k-kb; ++i) {
                    A_data[(k+kb+i) + (k+j)*lda] = workspace[i + j*ldwork] / Sj;
                }
            }
        }

        // block-row update: A := U^H A
        if (k+kb < nb) {
            for (int64_t j = 0; j < nb-k-kb; ++j) {
                for (int64_t i = 0; i < kb; ++i) {
                    workspace[i + j*kb] = A_data[(k+i) + (k+kb+j)*lda];
                }
            }

            blas::gemm(Layout::ColMajor,
                       Op::ConjTrans, Op::NoTrans,
                       kb, nb-k-kb, kb,
                       one,  &U_data[k + k*ldu], ldu,
                             &workspace[0], kb,
                       zero, &A_data[k + (k+kb)*lda], lda);
        }

        // trailing matrix update
        if (k+kb < mb && k+kb < nb) {
            blas::gemm(blas::Layout::ColMajor,
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
