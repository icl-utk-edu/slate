// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_lapack.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Cholesky factorization of single tile.
/// Dispatches to target implementations.
/// @ingroup posv_internal
///
template <Target target, typename scalar_t>
int64_t potrf(
    HermitianMatrix< scalar_t >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info)
{
    return potrf( internal::TargetType<target>(), A, priority,
                  queue_index, device_info );
}

//------------------------------------------------------------------------------
/// Cholesky factorization of single tile, host implementation.
/// @ingroup posv_internal
///
template <typename scalar_t>
int64_t potrf(
    internal::TargetType<Target::HostTask>,
    HermitianMatrix<scalar_t>& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);

    int64_t info = 0;
    if (A.tileIsLocal( 0, 0 )) {
        A.tileGetForWriting( 0, 0, LayoutConvert::ColMajor );
        info = potrf( A( 0, 0 ) );
    }
    return info;
}

//------------------------------------------------------------------------------
/// Cholesky factorization of single tile, device implementation.
/// @ingroup posv_internal
///
template <typename scalar_t>
int64_t potrf(
    internal::TargetType<Target::Devices>,
    HermitianMatrix<scalar_t>& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);

    int64_t info = 0;
    if (A.tileIsLocal(0, 0)) {
        int device = A.tileDevice( 0, 0 );
        A.tileGetForWriting(0, 0, device, LayoutConvert::ColMajor);
        lapack::Queue* queue = A.compute_queue( device, queue_index );
        auto A00 = A( 0, 0, device );
        lapack::potrf(
            A00.uploPhysical(), A00.mb(), A00.data(),
            A00.stride(), device_info, *queue );
        lapack::device_info_int host_info;
        blas::device_memcpy( device_info, &host_info, 1, *queue );
        queue->sync();
        info = int64_t( host_info );
    }
    return info;
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
int64_t potrf<Target::HostTask, float>(
    HermitianMatrix<float>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
int64_t potrf<Target::HostTask, double>(
    HermitianMatrix<double>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
int64_t potrf< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
int64_t potrf< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

template
int64_t potrf<Target::Devices, float>(
    HermitianMatrix<float>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
int64_t potrf<Target::Devices, double>(
    HermitianMatrix<double>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
int64_t potrf< Target::Devices, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
int64_t potrf< Target::Devices, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

} // namespace internal
} // namespace slate
