// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal
/// matrix in parallel, using the divide and conquer algorithm.
///
//------------------------------------------------------------------------------
/// @tparam real_t
///     One of float, double.
//------------------------------------------------------------------------------
/// @param[in,out] D
///     On entry, the diagonal elements of the tridiagonal matrix.
///     On exit, the eigenvalues in ascending order.
///     D is duplicated on all MPI ranks.
///
/// @param[in,out] E
///     On entry, the subdiagonal elements of the tridiagonal matrix.
///     On exit, E has been destroyed.
///     E is duplicated on all MPI ranks.
///
/// @param[out] Q
///     On exit, Q contains the orthonormal eigenvectors of the
///     symmetric tridiagonal matrix.
///     Code currently requires that Q has a 2D block-cyclic distribution,
///     with column-major grid order, which is the default for a SLATE Matrix.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup heev_computational
///
template <typename real_t>
void stedc(
    std::vector<real_t>& D, std::vector<real_t>& E,
    Matrix<real_t>& Q,
    Options const& opts )
{
    using lapack::MatrixType;

    int64_t n = D.size();

    const real_t zero = 0.0;
    const real_t one  = 1.0;

    real_t Anorm = lapack::lanst( Norm::Max, n, &D[0], &E[0] );
    if (Anorm == zero)
        return;
    if (std::isinf( Anorm ) || std::isnan( Anorm ))
        throw std::domain_error( "Input matrix contains Inf or NaN" );

    // Scale if necessary.
    // todo: steqr scales only if Anorm > sfmax or Anorm < sfmin,
    // but stedc always scales. Is that right?
    lapack::lascl( MatrixType::General, 0, 0, Anorm, one, n,   1, &D[0], n   );
    lapack::lascl( MatrixType::General, 0, 0, Anorm, one, n-1, 1, &E[0], n-1 );

    // For now, the algorithm is CPU-only.
    // Move Q to the CPU and reset target.
    // todo: the MOSI API doesn't have a way to do Hold + Modified in one call.
    Q.tileGetAndHoldAll( HostNum, LayoutConvert::ColMajor ); // get for reading
    Q.tileGetAllForWriting( HostNum, LayoutConvert::ColMajor );
    Options opts_local( opts );
    opts_local[ Option::Target ] = Target::HostTask;

    // Allocate workspace matrices W and U needed in stedc_merge.
    auto W = Q.emptyLike();
    W.insertLocalTiles();

    auto U = Q.emptyLike();
    U.insertLocalTiles();

    // Q = Identity.
    bool sort = true;  // todo pass via opts
    if (sort) {
        // Computing eigenvectors in W and sorting into Q saves a copy.
        set( zero, one, W, opts_local );
        stedc_solve( D, E, W, Q, U, opts_local );
        stedc_sort( D, W, Q, opts_local );
    }
    else {
        // Compute eigenvectors directly in Q.
        set( zero, one, Q, opts_local );
        stedc_solve( D, E, Q, W, U, opts_local );
    }

    // Scale eigenvalues back.
    lapack::lascl( MatrixType::General, 0, 0, one, Anorm, n, 1, &D[0], n );

    Q.tileUnsetHoldAll( HostNum );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Only real, not complex.
template
void stedc<float>(
    std::vector<float>& D, std::vector<float>& E,
    Matrix<float>& Q,
    Options const& opts );

template
void stedc<double>(
    std::vector<double>& D, std::vector<double>& E,
    Matrix<double>& Q,
    Options const& opts );

} // namespace slate
