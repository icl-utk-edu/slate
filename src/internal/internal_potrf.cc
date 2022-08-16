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
void potrf(HermitianMatrix< scalar_t >&& A, int priority)
{
    potrf(internal::TargetType<target>(), A, priority);
}

//------------------------------------------------------------------------------
/// Cholesky factorization of single tile, host implementation.
/// @ingroup posv_internal
///
template <typename scalar_t>
void potrf(internal::TargetType<Target::HostTask>,
           HermitianMatrix<scalar_t>& A, int priority)
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
           HermitianMatrix<scalar_t>& A, int priority)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);
    using lapack::device_info_int;
    typedef long long lld;

    if (A.tileIsLocal(0, 0))
    {
        int device = A.tileDevice( 0, 0 );
        blas::set_device(device);
        A.tileGetForWriting(0, 0, device, LayoutConvert::ColMajor);
        lapack::Queue* queue = A.compute_queue( device, 0 );
        device_info_int* d_info = blas::device_malloc< device_info_int >( 1 ); // todo Kadir avoid this allocation
        lapack::potrf(
            A(0, 0).uplo(), A(0, 0).mb(), A(0, 0, device).data(),
            A(0, 0, device).stride(), d_info, *queue );
        queue->sync();

        // Copy result back to CPU.
        device_info_int info_tst;
        blas::device_memcpy( &info_tst, d_info, 1, *queue );
        queue->sync();

        if (info_tst != 0) {
            fprintf( stderr, "lapack::potrf returned error %lld\n", (lld) info_tst );
        }

        // Cleanup GPU memory.
        blas::device_free( d_info );
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void potrf<Target::HostTask, float>(
    HermitianMatrix<float>&& A,
    int priority);

// ----------------------------------------
template
void potrf<Target::HostTask, double>(
    HermitianMatrix<double>&& A,
    int priority);

// ----------------------------------------
template
void potrf< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    int priority);

// ----------------------------------------
template
void potrf< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    int priority);

template
void potrf<Target::Devices, float>(
    HermitianMatrix<float>&& A,
    int priority);

// ----------------------------------------
template
void potrf<Target::Devices, double>(
    HermitianMatrix<double>&& A,
    int priority);

// ----------------------------------------
template
void potrf< Target::Devices, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    int priority);

// ----------------------------------------
template
void potrf< Target::Devices, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    int priority);

} // namespace internal
} // namespace slate
