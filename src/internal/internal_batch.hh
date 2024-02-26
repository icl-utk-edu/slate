// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
/// Provides various helper functions for batched routines.
///
/// Provides simple precision-independent wrappers around MKL batch
/// routines. Eventually to be replaced by BLAS++ batch routines.
///
/// Provides routines to build the batch regions for device batched kernels.
#ifndef SLATE_INTERNAL_BATCH_HH
#define SLATE_INTERNAL_BATCH_HH

#include "slate/Exception.hh"
#include "slate/BaseMatrix.hh"

#include <blas.hh>

#ifdef BLAS_HAVE_MKL
    #include <mkl_cblas.h>
#endif

#include <complex>
#include <set>

namespace slate {
namespace internal {

#ifdef BLAS_HAVE_MKL

//------------------------------------------------------------------------------
inline CBLAS_TRANSPOSE cblas_trans_const(Op op)
{
    switch (op) {
        case Op::NoTrans:   return CblasNoTrans;
        case Op::Trans:     return CblasTrans;
        case Op::ConjTrans: return CblasConjTrans;
        default: slate_error("unknown op");
    }
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const float* alpha_array,
    const float** A_array,
    const int* lda_array,
    const float** B_array,
    const int* ldb_array,
    const float* beta_array,
    float** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_sgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, A_array, lda_array,
                                   B_array, ldb_array,
                      beta_array,  C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const double* alpha_array,
    const double** A_array,
    const int* lda_array,
    const double** B_array,
    const int* ldb_array,
    const double* beta_array,
    double** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_dgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, A_array, lda_array,
                                   B_array, ldb_array,
                      beta_array,  C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const std::complex<float>* alpha_array,
    const std::complex<float>** A_array,
    const int* lda_array,
    const std::complex<float>** B_array,
    const int* ldb_array,
    const std::complex<float>* beta_array,
    std::complex<float>** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_cgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, (const void**) A_array, lda_array,
                                   (const void**) B_array, ldb_array,
                      beta_array,  (void**)       C_array, ldc_array,
                      group_count, group_size);
}

//------------------------------------------------------------------------------
inline void cblas_gemm_batch(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE* transA_array,
    const CBLAS_TRANSPOSE* transB_array,
    const int* m_array,
    const int* n_array,
    const int* k_array,
    const std::complex<double>* alpha_array,
    const std::complex<double>** A_array,
    const int* lda_array,
    const std::complex<double>** B_array,
    const int* ldb_array,
    const std::complex<double>* beta_array,
    std::complex<double>** C_array,
    const int* ldc_array,
    const int group_count,
    const int* group_size)
{
    cblas_zgemm_batch(layout, transA_array, transB_array,
                      m_array, n_array, k_array,
                      alpha_array, (const void**) A_array, lda_array,
                                   (const void**) B_array, ldb_array,
                      beta_array,  (void**)       C_array, ldc_array,
                      group_count, group_size);
}
#endif // BLAS_HAVE_MKL


// Utilities for computing device batch regions

//------------------------------------------------------------------------------
/// Computes the range of tiles with either the same mb or the same nb.
///
/// @param[in] dim
///     Whether to compute the row ranges or the column ranges
///
/// @param[in] A
///     The matrix to get tile sizes from
///
/// @return The ranges of uniform tile sizes
///
template<typename scalar_t>
std::vector<int64_t> device_regions_range( RowCol dim, BaseMatrix<scalar_t>& A )
{
    bool want_rows = dim == RowCol::Row;

    int64_t kt = want_rows ? A.mt() : A.nt();

    std::vector< int64_t > range;
    int64_t last = -1;
    for (int64_t k = 0; k < kt; ++k) {
        int64_t kb = want_rows ? A.tileMb( k ) : A.tileNb( k );
        if (kb != last) {
            last = kb;
            range.push_back( k );
        }
    }
    range.push_back( kt );
    return range;
}

//------------------------------------------------------------------------------
/// Helper class to store the information on a device region.
///
/// @tparam store_diag
///     Wheather the diagonal tiles may need to be special cased
///
/// @tparam mat_count
///     The number of matrices used by the kernel
///
template< bool store_diag, int mat_count >
struct device_regions_params {
    int64_t count, mb, nb;
    int64_t ld[mat_count];

private:
    // When store_diag is false, we don't want to allocate any memory for is_diagonal
    struct Empty {};
public:
    std::conditional_t< store_diag, bool, Empty > is_diagonal;

    device_regions_params()
            : count(0), mb(0), nb(0), ld{0}
    {
        if constexpr (store_diag) {
            is_diagonal = false;
        }
    }
};

//------------------------------------------------------------------------------
/// @copydoc device_regions_build(std::array< std::reference_wrapper<BaseMatrix<scalar_t>>, mat_count >, std::array< scalar_t**, mat_count >, int64_t, std::function<void(int64_t, int64_t, int64_t)>)
///
/// @params[in] irange
///     The ranges of tiles with a uniform number of rows
///
/// @params[in] jrange
///     The ranges of tiles with a uniform number of columns
///
template< bool store_diag, int mat_count, typename scalar_t, bool diag_same=!store_diag >
std::vector< device_regions_params<store_diag, mat_count> > device_regions_build(
        std::array< std::reference_wrapper<BaseMatrix<scalar_t>>, mat_count > mats,
        std::array< scalar_t**, mat_count > mats_array_host,
        int64_t device,
        std::function<void(int64_t, int64_t, int64_t)> extra_setup,
        std::vector<int64_t>& irange,
        std::vector<int64_t>& jrange)
{
    // The first two arguments should be valid targets for brace-initialization
    // reference_wrapper works around fact that C++ doesn't allow array of references

    using Params = device_regions_params<store_diag, mat_count>;

    auto& A = mats[0].get();

    // Trapezoidal matrices always need special treatment for diagonal tiles
    assert( !diag_same || A.uplo() == Uplo::General );

    static_assert( diag_same || store_diag,
                   "Can't special case the diagonal when is_diagonal is not allocated" );

    // Size 1 dimensions get broadcast to allow setting up GEMM et al.
    // i_step[m]=0 results in only accessing row 0 of matrix m (likewise for j)
    // The first matrix is always indexed normally since it determines the loops
    int64_t i_step[mat_count];
    int64_t j_step[mat_count];
    i_step[0] = 1;
    j_step[0] = 1;
    for (int m = 1; m < mat_count; ++m) {
        i_step[m] = (mats[ m ].get().mt() > 1);
        j_step[m] = (mats[ m ].get().nt() > 1);
    }

    int64_t batch_count = 0;
    int64_t mt = A.mt();
    std::vector<Params> group_params;
    // loop over regions
    for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
    for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
        // Loop over the tiles in this region,
        // save any that should be computed on this process & device
        Params group;
        group.mb = A.tileMb( irange[ ii ] );
        group.nb = A.tileNb( jrange[ jj ] );
        for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
            // This is a column major loop.  So,
            // * Lower matrices start at j+1
            // * Upper matrices end at j
            // * General matrices run the whole range
            int64_t istart = std::max(irange[ ii ], (A.uplo() == Uplo::Lower ? j+1 : 0));
            int64_t iend   = std::min(irange[ ii+1 ], (A.uplo() == Uplo::Upper ? j : mt));
            for (int64_t i = istart; i < iend; ++i) {
                if ((diag_same || i != j)
                    && A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )) {

                    // Add tiles to current group
                    for (int m = 0; m < mat_count; ++m) {
                        auto Mij = mats[ m ].get()( i*i_step[m], j*j_step[m], device );
                        mats_array_host[ m ][ batch_count ] = Mij.data();
                        if (group.count == 0) {
                            group.ld[m] = Mij.stride();
                        }
                        else {
                            assert( group.ld[m] == Mij.stride() );
                        }
                    }
                    if (extra_setup) {
                        extra_setup( group_params.size(), i, j );
                    }
                    ++group.count;
                    ++batch_count;
                }
            } // for i
        } // for j
        // If any tiles in the region should be computed here, save the group
        if (group.count > 0) {
            group_params.push_back( group );
        }

        // If the diagonal tiles need special treatment, build those groups
        if constexpr (store_diag && !diag_same) {
            // Loop over the diagonal tiles in this region.  If any should be
            // computed on this process & device, save them.
            group = Params();
            group.is_diagonal = true;
            group.mb = A.tileMb( irange[ ii ] );
            group.nb = A.tileNb( jrange[ jj ] );
            // Diagonal tiles only in the intersection of irange and jrange
            int64_t ijstart = std::max(irange[ ii   ], jrange[ jj   ]);
            int64_t ijend   = std::min(irange[ ii+1 ], jrange[ jj+1 ]);
            for (int64_t ij = ijstart; ij < ijend; ++ij) {
                if (A.tileIsLocal( ij, ij )
                    && device == A.tileDevice( ij, ij )) {

                    // Add tiles to current group
                    // This logic matches that of above
                    for (int m = 0; m < mat_count; ++m) {
                        auto Mij = mats[ m ].get()( ij*i_step[m], ij*j_step[m], device );
                        mats_array_host[ m ][ batch_count ] = Mij.data();
                        if (group.count == 0) {
                            group.ld[m] = Mij.stride();
                        }
                        else {
                            assert( group.ld[m] == Mij.stride() );
                        }
                    }
                    if (extra_setup) {
                        extra_setup( group_params.size(), ij, ij );
                    }
                    ++group.count;
                    ++batch_count;
                }
            } // for ij
            // If any tiles in the region should be computed here, save the group
            if (group.count > 0) {
                group_params.push_back( group );
            }
        } // if store_diag && !diag_same
    }} // for jj, ii
    return group_params;
}

//------------------------------------------------------------------------------
/// Computes and populates the regions for the given matrices.
///
/// @tparam store_diag
///     Whether the diagonal tiles may need to be special cased
///
/// @tparam mat_count
///     The number of matrices used by the kernel
///
/// @tparam scalar_t
///     The type of the matrices
///
/// @tparam[in] diag_same
///     Whether to include the diagonal tiles in the off-diagonal groups
///     If false, store_diag must be true
//------------------------------------------------------------------------------
/// @param[in] mats
///     An array of the matrices to build regions for
///
/// @param[in] mats_array_host
///     An array of the arrays to fill with pointers to device data
///
/// @param[in] device
///     The device to build regions for
///
/// @param[in] extra_setup
///     Callback that is called whenever a tile is added to a group.
///     The group index and the tile indices are passed as arguments
///
/// @return A list of batches with identical size.
///
template< bool store_diag, int mat_count, typename scalar_t, bool diag_same=!store_diag >
std::vector< device_regions_params<store_diag, mat_count> > device_regions_build(
        std::array< std::reference_wrapper<BaseMatrix<scalar_t>>, mat_count > mats,
        std::array< scalar_t**, mat_count > mats_array_host,
        int64_t device,
        std::function<void(int64_t, int64_t, int64_t)> extra_setup = {})
{
    // Find ranges of matching mb's and ranges of matching nb's.
    auto irange = device_regions_range( RowCol::Row, mats[0].get() );
    auto jrange = device_regions_range( RowCol::Col, mats[0].get() );

    return device_regions_build< store_diag, mat_count, scalar_t, diag_same >(
                                 mats, mats_array_host, device, extra_setup,
                                 irange, jrange );
}


} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_BATCH_HH
