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
/// Reduces a complex Hermitian positive-definite generalized eigenvalue problem
/// to the standard form of single tile.
/// Dispatches to target implementations.
/// @ingroup hegv_internal
///
template <Target target, typename scalar_t>
void hegst(int64_t itype, HermitianMatrix< scalar_t >&& A,
                          HermitianMatrix< scalar_t >&& B)
{
    hegst(internal::TargetType<target>(), itype, A, B);
}

//------------------------------------------------------------------------------
/// Reduces a complex Hermitian positive-definite generalized eigenvalue problem
/// to the standard form of single tile, host implementation.
/// @ingroup hegv_internal
///
template <typename scalar_t>
void hegst(internal::TargetType<Target::HostTask>,
           int64_t itype, HermitianMatrix<scalar_t>& A,
                          HermitianMatrix<scalar_t>& B)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(B.nt() == 1);

    if (A.tileIsLocal(0, 0)) {
        {
            A.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
            B.tileGetForReading(0, 0, LayoutConvert::ColMajor);
            hegst(itype, A(0, 0), B(0, 0));
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void hegst<Target::HostTask, float>(
    int64_t itype, HermitianMatrix<float>&& A,
                   HermitianMatrix<float>&& B);

// ----------------------------------------
template
void hegst<Target::HostTask, double>(
    int64_t itype, HermitianMatrix<double>&& A,
                   HermitianMatrix<double>&& B);

// ----------------------------------------
template
void hegst<Target::HostTask, std::complex<float>>(
    int64_t itype, HermitianMatrix<std::complex<float>>&& A,
                   HermitianMatrix<std::complex<float>>&& B);

// ----------------------------------------
template
void hegst<Target::HostTask, std::complex<double>>(
    int64_t itype, HermitianMatrix<std::complex<double>>&& A,
                   HermitianMatrix<std::complex<double>>&& B);

} // namespace internal
} // namespace slate
