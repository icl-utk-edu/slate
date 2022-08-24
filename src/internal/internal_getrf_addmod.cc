// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/Tile_getrf_addmod.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// LU factorization of single tile with additive corrections.
/// Dispatches to target implementations.
/// @ingroup gesv_internal
///
template <Target target, typename scalar_t>
void getrf_addmod(Matrix< scalar_t >&& A,
                  Matrix< scalar_t >&& U,
                  std::vector< blas::real_type<scalar_t> >&& singular_values,
                  std::vector< blas::real_type<scalar_t> >&& modifications,
                  std::vector<int64_t>&& modified_indices,
                  blas::real_type<scalar_t> mod_tol,
                  int64_t ib)
{
    getrf_addmod(internal::TargetType<target>(),
                 A, U, singular_values, modifications, modified_indices,
                 mod_tol, ib);
}

//------------------------------------------------------------------------------
/// LU factorization of single tile with additive corrections, host implementation.
/// @ingroup gesv_internal
///
template <typename scalar_t>
void getrf_addmod(internal::TargetType<Target::HostTask>,
                  Matrix<scalar_t>& A,
                  Matrix<scalar_t>& U,
                  std::vector< blas::real_type<scalar_t> >& singular_values,
                  std::vector< blas::real_type<scalar_t> >& modifications,
                  std::vector<int64_t>& modified_indices,
                  blas::real_type<scalar_t> mod_tol,
                  int64_t ib)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);

    if (A.tileIsLocal(0, 0)) {
        A.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
        U.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
        getrf_addmod(A(0, 0), U(0, 0), singular_values, modifications, modified_indices,
                     mod_tol, ib);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf_addmod<Target::HostTask, float>(
    Matrix<float>&& A,
    Matrix<float>&& U,
    std::vector<float>&& singular_values,
    std::vector<float>&& modifications,
    std::vector<int64_t>&& modified_indices,
    float mod_tol,
    int64_t ib);

// ----------------------------------------
template
void getrf_addmod<Target::HostTask, double>(
    Matrix<double>&& A,
    Matrix<double>&& U,
    std::vector<double>&& singular_values,
    std::vector<double>&& modifications,
    std::vector<int64_t>&& modified_indices,
    double mod_tol,
    int64_t ib);

// ----------------------------------------
template
void getrf_addmod< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& U,
    std::vector<float>&& singular_values,
    std::vector<float>&& modifications,
    std::vector<int64_t>&& modified_indices,
    float mod_tol,
    int64_t ib);

// ----------------------------------------
template
void getrf_addmod< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& U,
    std::vector<double>&& singular_values,
    std::vector<double>&& modifications,
    std::vector<int64_t>&& modified_indices,
    double mod_tol,
    int64_t ib);

} // namespace internal
} // namespace slate
