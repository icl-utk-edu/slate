// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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

//------------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `slate::he2hb` as
/// follows:
///
/// op              |  side = Left  |  side = Right
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
///     - Side::Right: apply $Q$ or $Q^H$ from the right.
///
/// @param[in] op
///     - Op::NoTrans    apply $Q$;
///     - Op::ConjTrans: apply $Q^H$;
///     - Op::Trans:     apply $Q^T$ (only if real).
///       In the real case, Op::Trans is equivalent to Op::ConjTrans.
///       In the complex case, Op::Trans is not allowed.
///
/// @param[in] A
///     On entry, the n-by-n Hermitian matrix $A$, as returned by
///     `slate::he2hb`.
///
/// @param[in] T
///     On entry, triangular matrices of the elementary
///     reflector H(i), as returned by `slate::he2hb`.
///
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, $C$ is overwritten by $Q C$, $Q^H C$, $C Q$, or $C Q^H$.
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
/// @ingroup svd_computational
///
template <typename scalar_t>
void unmbr_ge2tb(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t> T,
    Matrix<scalar_t>& C,
    Options const& opts)
{
    if (side == Side::Left) {
        // Form UC, where U's representation is in lower part of A and TU.
        unmqr(side, Op::NoTrans, A, T, C, opts);
    }

    else if (side == Side::Right) {
        // Form (UC)V^H, where V's representation is above band in A and TV.
        auto Asub =  A.sub(0, A.mt()-1, 1, A.nt()-1);
        auto Csub =  C.sub(0, C.mt()-1, 1, C.nt()-1);
        slate::TriangularFactors<scalar_t> Tsub = {
            T[0].sub(0, T[0].mt()-1, 1, T[0].nt()-1),
            T[1].sub(0, T[1].mt()-1, 1, T[1].nt()-1)
        };

        slate::unmlq(slate::Side::Right, op,
                Asub, Tsub, Csub, opts);
    }

    // Note V^H == Q, not Q^H.
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmbr_ge2tb<float>(
    Side side, Op op,
    Matrix<float>& A,
    TriangularFactors<float> T,
    Matrix<float>& C,
    Options const& opts);

template
void unmbr_ge2tb<double>(
    Side side, Op op,
    Matrix<double>& A,
    TriangularFactors<double> T,
    Matrix<double>& C,
    Options const& opts);

template
void unmbr_ge2tb<std::complex<float>>(
    Side side, Op op,
    Matrix<std::complex<float>>& A,
    TriangularFactors<std::complex<float> > T,
    Matrix< std::complex<float> >& C,
    Options const& opts);

template
void unmbr_ge2tb<std::complex<double>>(
    Side side, Op op,
    Matrix<std::complex<double>>& A,
    TriangularFactors<std::complex<double>> T,
    Matrix<std::complex<double>>& C,
    Options const& opts);

} // namespace slate
