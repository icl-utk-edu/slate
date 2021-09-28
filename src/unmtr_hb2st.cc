// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel unmtr_hb2st.
/// Generic implementation for any target.
/// @ingroup heev_specialization
///
template <Target target, typename scalar_t>
void unmtr_hb2st(slate::internal::TargetType<target>,
                 Side side, Op op,
                 Matrix<scalar_t>& V,
                 Matrix<scalar_t>& C,
                 const std::map<Option, Value>& opts)
{
    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        #pragma omp task
        {
            internal::unmtr_hb2st<target>(side, op, V, C, opts);
        }
        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }
}
} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup heev_computational
///
template <Target target, typename scalar_t>
void unmtr_hb2st(Side side, Op op,
                 Matrix<scalar_t>& V,
                 Matrix<scalar_t>& C,
                 const std::map<Option, Value>& opts)
{
    internal::specialization::unmtr_hb2st(internal::TargetType<target>(),
                                          side, op, V, C, opts);
}

//------------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `slate::hb2st` as
/// follows:
///
/// op              |  side = Left  |  side = Right (not supported)
/// --------------- | ------------- | --------------
/// op = NoTrans    |  $Q C  $      |  $C Q  $
/// op = ConjTrans  |  $Q^H C$      |  $C Q^H$
///
/// where $Q$ is a unitary matrix defined as the product of k
/// elementary reflectors
/// \[
///     Q = H(1) H(2) . . . H(k)
/// \]
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///     - Side::Left:  apply $Q$ or $Q^H$ from the left;
///     - Side::Right: apply $Q$ or $Q^H$ from the right (not supported).
///
/// @param[in] op
///     - Op::NoTrans    apply $Q$;
///     - Op::ConjTrans: apply $Q^H$;
///     - Op::Trans:     apply $Q^T$ (only if real).
///       In the real case, Op::Trans is equivalent to Op::ConjTrans.
///       In the complex case, Op::Trans is not allowed.
///
/// @param[in] V
///     Householder vectors as returned by `slate::hb2st`.
///
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, $C$ is overwritten by $Q C$, $Q^H C$, $C Q$, or $C Q^H$.
///     C must be distributed 1D block column (cyclic or non-cyclic).
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
template <typename scalar_t>
void unmtr_hb2st(Side side, Op op,
                 Matrix<scalar_t>& V,
                 Matrix<scalar_t>& C,
                 const std::map<Option, Value>& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            internal::unmtr_hb2st<Target::HostTask>( side, op, V, C, opts);
            break;
        case Target::HostNest:
            break;
        case Target::HostBatch:
            break;
        case Target::Devices:
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmtr_hb2st<float>(
    Side side, Op op,
    Matrix<float>& V,
    Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<double>(
    Side side, Op op,
    Matrix<double>& V,
    Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st< std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& V,
    Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st< std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& V,
    Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

} // namespace slate
