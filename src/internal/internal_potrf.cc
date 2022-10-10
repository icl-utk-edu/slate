// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
void potrf(HermitianMatrix< scalar_t >&& A,
           int priority, int64_t queue_index,
           lapack::device_info_int* device_info)
{
    potrf(internal::TargetType<target>(), A, priority,
          queue_index, device_info);
}

//------------------------------------------------------------------------------
/// Cholesky factorization of single tile, host implementation.
/// @ingroup posv_internal
///
template <typename scalar_t>
void potrf(internal::TargetType<Target::HostTask>,
           HermitianMatrix<scalar_t>& A,
           int priority, int64_t queue_index,
           lapack::device_info_int* device_info)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);

    if (A.tileIsLocal(0, 0))
        {
            A.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
            potrf(A(0, 0));
        }
}

//------------------------------------------------------------------------------
/// Cholesky factorization of single tile, device implementation.
/// @ingroup posv_internal
///
template <typename scalar_t>
void potrf(internal::TargetType<Target::Devices>,
           HermitianMatrix<scalar_t>& A,
           int priority, int64_t queue_index,
           lapack::device_info_int* device_info)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);

    if (A.tileIsLocal(0, 0)) {
        int device = A.tileDevice( 0, 0 );
        A.tileGetForWriting(0, 0, device, LayoutConvert::ColMajor);
        lapack::Queue* queue = A.compute_queue( device, queue_index );
        auto A00 = A( 0, 0, device );
        lapack::potrf(
            A00.uploPhysical(), A00.mb(), A00.data(),
            A00.stride(), device_info, *queue );
        queue->sync();
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void potrf<Target::HostTask, float>(
    HermitianMatrix<float>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
void potrf<Target::HostTask, double>(
    HermitianMatrix<double>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
void potrf< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
void potrf< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

template
void potrf<Target::Devices, float>(
    HermitianMatrix<float>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
void potrf<Target::Devices, double>(
    HermitianMatrix<double>&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
void potrf< Target::Devices, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

// ----------------------------------------
template
void potrf< Target::Devices, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    int priority, int64_t queue_index,
    lapack::device_info_int* device_info);

} // namespace internal
} // namespace slate
