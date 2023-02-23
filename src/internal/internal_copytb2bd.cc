// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Copy bi-diagonal TriangularBand matrix to two vectors.
/// Dispatches to target implementations.
/// @ingroup copy_internal
///
template <Target target, typename scalar_t>
void copytb2bd(TriangularBandMatrix<scalar_t>& A,
               std::vector< blas::real_type<scalar_t> >& D,
               std::vector< blas::real_type<scalar_t> >& E)
{
    copytb2bd(internal::TargetType<target>(),
               A,
               D, E);
}

//------------------------------------------------------------------------------
/// Copy bi-diagonal TriangularBand matrix to two vectors.
/// Host OpenMP task implementation.
/// @ingroup copy_internal
///
template <typename scalar_t>
void copytb2bd(internal::TargetType<Target::HostTask>,
               TriangularBandMatrix<scalar_t> A,
               std::vector< blas::real_type<scalar_t> >& D,
               std::vector< blas::real_type<scalar_t> >& E)
{
    trace::Block trace_block("slate::copytb2bd");
    using blas::real;

    // If lower, change to upper.
    if (A.uplo() == Uplo::Lower) {
        A = conj_transpose( A );
    }

    // Make sure it is a bi-diagonal matrix.
    slate_assert(A.bandwidth() == 1);

    int64_t nt = A.nt();
    int64_t n = A.n();
    D.resize(n);
    E.resize(n - 1);

    // Copy diagonal & super-diagonal.
    int64_t D_index = 0;
    int64_t E_index = 0;
    for (int64_t i = 0; i < nt; ++i) {
        // Copy 1 element from super-diagonal tile to E.
        if (i > 0) {
            auto T = A(i-1, i);
            E[E_index] = real( T(T.mb()-1, 0) );
            E_index += 1;
            A.tileTick(i-1, i);
        }

        // Copy main diagonal to D.
        auto T = A(i, i);
        slate_assert(T.mb() == T.nb()); // square diagonal tile
        auto len = T.nb();
        for (int j = 0; j < len; ++j) {
            D[D_index + j] = real( T(j, j) );
        }
        D_index += len;

        // Copy super-diagonal to E.
        for (int j = 0; j < len-1; ++j) {
            E[E_index + j] = real( T(j, j+1) );
        }
        E_index += len-1;
        A.tileTick(i, i);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void copytb2bd<Target::HostTask, float>(
    TriangularBandMatrix<float>& A,
    std::vector<float>& D,
    std::vector<float>& E);

// ----------------------------------------
template
void copytb2bd<Target::HostTask, double>(
    TriangularBandMatrix<double>& A,
    std::vector<double>& D,
    std::vector<double>& E);

// ----------------------------------------
template
void copytb2bd< Target::HostTask, std::complex<float> >(
    TriangularBandMatrix< std::complex<float> >& A,
    std::vector<float>& D,
    std::vector<float>& E);

// ----------------------------------------
template
void copytb2bd< Target::HostTask, std::complex<double> >(
    TriangularBandMatrix< std::complex<double> >& A,
    std::vector<double>& D,
    std::vector<double>& E);

} // namespace internal
} // namespace slate
