// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
/// An helper function to find each rank's first (top-most) row in panel k for
/// the QR-family of routines.
///
/// @param[in] A_panel
///     Current panel, which is a sub of the input matrix $A$.
///
/// @param[in] k
///     Index of the current panel in the input matrix $A$.
///
/// @param[out] first_indices
///     The array of computed indices.
///
/// @ingroup geqrf_impl
///
template <typename scalar_t>
std::vector< int64_t > geqrf_compute_first_indices(
    Matrix<scalar_t>& A_panel, int64_t k)
{
    // Find ranks in this column.
    std::set<int> ranks_set;
    A_panel.getRanks(&ranks_set);
    assert(ranks_set.size() > 0);

    // Find each rank's first (top-most) row in this panel,
    // where the triangular tile resulting from local geqrf panel
    // will reside.
    std::vector< int64_t > first_indices;
    first_indices.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t i = 0; i < A_panel.mt(); ++i) {
            if (A_panel.tileRank(i, 0) == r) {
                first_indices.push_back(i+k);
                break;
            }
        }
    }
    return first_indices;
}

//------------------------------------------------------------------------------
/// An helper function to find each rank's first (left-most) row in panel k for
/// the LQ-family of routines.
///
/// @param[in] A_panel
///     Current panel, which is a sub of the input matrix $A$.
///
/// @param[in] k
///     Index of the current panel in the input matrix $A$.
///
/// @param[out] first_indices
///     The array of computed indices.
///
/// @ingroup gelqf_impl
///
template <typename scalar_t>
std::vector< int64_t > gelqf_compute_first_indices(
    Matrix<scalar_t> A_panel, int64_t k)
{
    // Find ranks in this column.
    std::set<int> ranks_set;
    A_panel.getRanks(&ranks_set);
    assert(ranks_set.size() > 0);

    // Find each rank's first (top-most) row in this panel,
    // where the triangular tile resulting from local geqrf panel
    // will reside.
    std::vector< int64_t > first_indices;
    first_indices.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t j = 0; j < A_panel.nt(); ++j) {
            if (A_panel.tileRank(0, j) == r) {
                first_indices.push_back(j+k);
                break;
            }
        }
    }
    return first_indices;
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



} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_UTIL_HH
