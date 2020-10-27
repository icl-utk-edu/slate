// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_lapack.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
///
/// todo: update docs: multiply not inverse
///
/// Triangular inversion of single tile.
/// Dispatches to target implementations.
/// @ingroup tr_internal
///
template <Target target, typename scalar_t>
void trtrm(TriangularMatrix< scalar_t >&& A, int priority)
{
    trtrm(internal::TargetType<target>(), A, priority);
}

//------------------------------------------------------------------------------
/// Triangular inversion of single tile, host implementation.
/// @ingroup tr_internal
///
template <typename scalar_t>
void trtrm(internal::TargetType<Target::HostTask>,
           TriangularMatrix<scalar_t>& A, int priority)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);

    if (A.tileIsLocal(0, 0))
        #pragma omp task shared(A) priority(priority)
        {
            A.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
            trtrm(A(0, 0));
        }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trtrm<Target::HostTask, float>(
    TriangularMatrix<float>&& A,
    int priority);

// ----------------------------------------
template
void trtrm<Target::HostTask, double>(
    TriangularMatrix<double>&& A,
    int priority);

// ----------------------------------------
template
void trtrm< Target::HostTask, std::complex<float> >(
    TriangularMatrix< std::complex<float> >&& A,
    int priority);

// ----------------------------------------
template
void trtrm< Target::HostTask, std::complex<double> >(
    TriangularMatrix< std::complex<double> >&& A,
    int priority);

} // namespace internal
} // namespace slate
