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

} // namespace internal
} // namespace slate
