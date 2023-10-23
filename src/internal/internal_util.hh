// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_INTERNAL_UTIL_HH
#define SLATE_INTERNAL_UTIL_HH

#include "slate/internal/mpi.hh"
#include "slate/Matrix.hh"
#include "slate/BaseTrapezoidMatrix.hh"

#include <cmath>
#include <complex>

#include <blas.hh>

namespace slate {
namespace internal {

template <typename T>
T pow(T base, T exp);

void mpi_max_nan(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

//------------------------------------------
inline float real(float val) { return val; }
inline double real(double val) { return val; }
inline float real(std::complex<float> val) { return val.real(); }
inline double real(std::complex<double> val) { return val.real(); }

inline float imag(float val) { return 0.0; }
inline double imag(double val) { return 0.0; }
inline float imag(std::complex<float> val) { return val.imag(); }
inline double imag(std::complex<double> val) { return val.imag(); }

//--------------------------
template <typename scalar_t>
scalar_t make(blas::real_type<scalar_t> real, blas::real_type<scalar_t> imag);

template <>
inline float make<float>(float real, float imag) { return real; }

template <>
inline double make<double>(double real, double imag) { return real; }

template <>
inline std::complex<float> make<std::complex<float>>(float real, float imag)
{
    return std::complex<float>(real, imag);
}

template <>
inline std::complex<double> make<std::complex<double>>(double real, double imag)
{
    return std::complex<double>(real, imag);
}

//------------------------------------------------------------------------------
/// Helper function to sort by second element of a pair.
/// Used to sort rank_rows by row (see ttqrt, ttmqr), and rank_cols by col.
/// @return True if a.second < b.second.
template <typename T1, typename T2>
inline bool compareSecond(
    std::pair<T1, T2> const& a,
    std::pair<T1, T2> const& b)
{
    return a.second < b.second;
}


//------------------------------------------------------------------------------
/// Helper function to check convergence in iterative methods
template <typename scalar_t>
bool iterRefConverged(std::vector<scalar_t>& colnorms_R,
                      std::vector<scalar_t>& colnorms_X,
                      scalar_t cte)
{
    assert(colnorms_X.size() == colnorms_R.size());
    bool value = true;
    int64_t size = colnorms_X.size();

    for (int64_t i = 0; i < size; i++) {
        if (colnorms_R[i] > colnorms_X[i] * cte) {
            value = false;
            break;
        }
    }

    return value;
}

//------------------------------------------------------------------------------
/// Helper function to allocate a krylov basis
template<typename scalar_t>
slate::Matrix<scalar_t> alloc_basis(slate::BaseMatrix<scalar_t>& A, int64_t n,
                                    Target target)
{
    auto mpiComm = A.mpiComm();
    auto tileMbFunc = A.tileMbFunc();
    auto tileNbFunc = A.tileNbFunc();
    auto tileRankFunc = A.tileRankFunc();
    auto tileDeviceFunc = A.tileDeviceFunc();
    Matrix<scalar_t> V(A.m(), n, tileMbFunc, tileNbFunc,
                       tileRankFunc, tileDeviceFunc, mpiComm);
    V.insertLocalTiles(target);
    return V;
}


// Utilities for device batch regions

//------------------------------------------------------------------------------
/// Computes the range of tiles with either the same mb or the same nb
///
/// @param[in] want_rows
///     If true, compute the row-ranges.  Else, compute the column-ranges.
///
/// @param[in] A
///     The matrix to get tile sizes from
///
/// @return The ranges of uniform tile sizes
///
template<typename scalar_t>
std::vector<int64_t> device_regions_range( bool want_rows, BaseMatrix<scalar_t>& A )
{
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
/// Helper class to store the information on a device region
///
/// @tparam has_diag
///     Wheather the diagonal tiles may need to be special cased
///
/// @tparam mat_count
///     The number of matrices used by the kernel
///
template< bool has_diag, int mat_count >
struct device_regions_params {
    int64_t count, mb, nb;
    int64_t ld[mat_count];

private:
    // When has_diag is false, we don't want to allocate any memory for is_diagonal
    struct Empty {};
public:
    std::conditional_t< has_diag, bool, Empty > is_diagonal;

    device_regions_params()
            : count(0), mb(0), nb(0)
    {
        for (int i = 0; i < mat_count; ++i) {
            ld[i] = 0;
        }
        if constexpr (has_diag) {
            is_diagonal = false;
        }
    }
};

//------------------------------------------------------------------------------
/// Computes and populates the regions for the given matrices.
///
/// @params[in] mats
///     An array of the matrices to build regions for
///
/// @params[in] mats_array_host
///     An array of the arrays to fill with pointers to device data
///
/// @params[in] device
///     The device to build regions for
///
/// @params[in] diag_same
///     Whether to treat the diagonal tiles as normal tiles in spite of has_diag
///     Ignored when has_diag is false.
///
template< bool has_diag, int mat_count, typename scalar_t>
std::vector< device_regions_params<has_diag, mat_count> > device_regions_build(
        std::array< std::reference_wrapper<BaseMatrix<scalar_t>>, mat_count > mats,
        std::array< scalar_t**, mat_count > mats_array_host,
        int64_t device,
        bool diag_same = true)
{
    // The first two arguments should be valid targets for brace-initialization
    // reference_wrapper works around fact that C++ doesn't allow array of references

    using Params = device_regions_params<has_diag, mat_count>;

    auto& A = mats[0].get();

    // Find ranges of matching mb's and ranges of matching nb's.
    std::vector< int64_t > irange = device_regions_range( true, A );
    std::vector< int64_t > jrange = device_regions_range( false, A );

    // Trapezoidal matrices always need special treatment for diagonal tiles
    diag_same &= A.uplo() == Uplo::General;

    // Can't treat diagonals special when we can't store the diagonal status
    assert( diag_same || has_diag );
    diag_same |= !has_diag; // Ensure the compiler can propagate this assertion

    // Single dimensions are always indexed as 0. This allows setting up GEMM et al.
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
    for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
    for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
        Params group;
        group.mb = A.tileMb( irange[ ii ] );
        group.nb = A.tileNb( jrange[ jj ] );
        for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
            // Lower matrices start at j+1
            // Upper matrices end at j
            // General matrices run the whole range
            int istart = std::max(irange[ ii ], (A.uplo() == Uplo::Lower ? j+1 : 0));
            int iend   = std::min(irange[ ii+1 ], (A.uplo() == Uplo::Upper ? j : mt));
            for (int64_t i = istart; i < iend; ++i) {
                if ((!has_diag || diag_same || i != j)
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
                    ++group.count;
                    ++batch_count;
                }
            } // for i
        } // for j
        if (group.count > 0) {
            group_params.push_back( group );
        }

        // If the diagonal tiles need special treatment, build those groups
        if constexpr (has_diag) if (!diag_same) {
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
                        auto Mij = mats[ m ].get()( ij, ij, device );
                        mats_array_host[ m ][ batch_count ] = Mij.data();
                        if (group.count == 0) {
                            group.ld[m] = Mij.stride();
                        }
                        else {
                            assert( group.ld[m] == Mij.stride() );
                        }
                    }
                    ++group.count;
                    ++batch_count;
                }
            } // for ij
            if (group.count > 0) {
                group_params.push_back( group );
            }
        } // if has_diag && !diag_same
    }} // for jj, ii
    return group_params;
}


} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_UTIL_HH
