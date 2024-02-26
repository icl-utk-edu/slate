// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/Tile_getrf_nopiv.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// LU factorization of single tile without pivoting.
/// Dispatches to target implementations.
/// @ingroup gesv_internal
///
template <Target target, typename scalar_t>
void getrf_nopiv(
    Matrix< scalar_t >&& A,
    int64_t ib, int priority, int64_t* info )
{
    getrf_nopiv( internal::TargetType<target>(), A, ib, priority, info );
}

//------------------------------------------------------------------------------
/// LU factorization of single tile without pivoting, host implementation.
///
/// @param[in,out] info
///     Exit status.
///     * 0: successful exit
///     * i > 0: U(i,i) is exactly zero (1-based index). The factorization
///       will have NaN due to division by zero.
///
/// @ingroup gesv_internal
///
template <typename scalar_t>
void getrf_nopiv(
    internal::TargetType<Target::HostTask>,
    Matrix<scalar_t>& A,
    int64_t ib, int priority, int64_t* info )
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);

    *info = 0;

    if (A.tileIsLocal(0, 0)) {
        A.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
        tile::getrf_nopiv( A( 0, 0 ), ib, info );
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf_nopiv<Target::HostTask, float>(
    Matrix<float>&& A,
    int64_t ib,
    int priority,
    int64_t* info );

// ----------------------------------------
template
void getrf_nopiv<Target::HostTask, double>(
    Matrix<double>&& A,
    int64_t ib,
    int priority,
    int64_t* info );

// ----------------------------------------
template
void getrf_nopiv< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    int64_t ib,
    int priority,
    int64_t* info );

// ----------------------------------------
template
void getrf_nopiv< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    int64_t ib,
    int priority,
    int64_t* info );

} // namespace internal
} // namespace slate
