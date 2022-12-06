// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
// #include "auxiliary/Debug.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Computes all eigenvalues of a symmetric tridiagonal matrix
/// using the Pal-Walker-Kahan variant of the QL or QR algorithm.
///
/// ATTENTION: only host computation supported for now
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void sterf(
    std::vector< scalar_t >& D,
    std::vector< scalar_t >& E,
    Options const& opts )
{
    trace::Block trace_block("lapack::sterf");

    lapack::sterf( D.size(), &D[0], &E[0] );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void sterf<float>(
    std::vector<float>& D,
    std::vector<float>& E,
    Options const& opts);

template
void sterf<double>(
    std::vector<double>& D,
    std::vector<double>& E,
    Options const& opts);

} // namespace slate
