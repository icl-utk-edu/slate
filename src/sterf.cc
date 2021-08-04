// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
// #include "aux/Debug.hh"
#include "slate/Tile_blas.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

#include <atomic>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::sterf from internal::specialization::sterf
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// computes all eigenvalues of a symmetric tridiagonal matrix
/// using the Pal-Walker-Kahan variant of the QL or QR algorithm.
/// Generic implementation for any target.
/// @ingroup svd_specialization
///
// ATTENTION: only host computation supported for now
//
template <Target target, typename scalar_t>
void sterf(slate::internal::TargetType<target>,
           std::vector< scalar_t >& D,
           std::vector< scalar_t >& E)
{
    trace::Block trace_block("lapack::sterf");

    lapack::sterf(D.size(), &D[0], &E[0]);
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup svd_specialization
///
template <Target target, typename scalar_t>
void sterf(std::vector< scalar_t >& D,
           std::vector< scalar_t >& E,
           Options const& opts)
{
    internal::specialization::sterf<target, scalar_t>(
                                    internal::TargetType<target>(),
                                    D, E);
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
void sterf(std::vector< scalar_t >& D,
           std::vector< scalar_t >& E,
           Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    // only HostTask implementation is provided, since it calls LAPACK only
    switch (target) {
        case Target::Host:
        case Target::HostTask:
        case Target::HostNest:
        case Target::HostBatch:
        case Target::Devices:
            sterf<Target::HostTask, scalar_t>(D, E, opts);
            break;
    }
    // todo: return value for errors?
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
