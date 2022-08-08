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
/// @ingroup heev_computational
///
template <typename scalar_t>
void unmtr_he2hb(
    Side side, Op op,
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t> T,
    Matrix<scalar_t>& C,
    Options const& opts)
{
    slate::TriangularFactors<scalar_t> T_sub = {
        T[ 0 ].sub( 1, A.nt()-1, 0, A.nt()-1 ),
        T[ 1 ].sub( 1, A.nt()-1, 0, A.nt()-1 )
    };

    if (A.uplo() == Uplo::Upper) {
        // todo: never tested.
        auto A_sub = slate::Matrix<scalar_t>(A, 0, A.nt()-1, 1, A.nt()-1);
        slate::unmlq(side, op, A_sub, T_sub, C, opts);
    }
    else { // uplo == Uplo::Lower
        auto A_sub = slate::Matrix<scalar_t>(A, 1, A.nt()-1, 0,  A.nt()-1);

        int64_t i0 = (side == Side::Left) ? 1 : 0;
        int64_t i1 = (side == Side::Left) ? 0 : 1;

        auto C_cub = C.sub(i0, A.nt()-1, i1, A.nt()-1);

        slate::unmqr(side, op, A_sub, T_sub, C_cub, opts);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmtr_he2hb<float>(
    Side side, Op op,
    HermitianMatrix<float>& A,
    TriangularFactors<float> T,
    Matrix<float>& C,
    Options const& opts);

template
void unmtr_he2hb<double>(
    Side side, Op op,
    HermitianMatrix<double>& A,
    TriangularFactors<double> T,
    Matrix<double>& C,
    Options const& opts);

template
void unmtr_he2hb<std::complex<float>>(
    Side side, Op op,
    HermitianMatrix<std::complex<float>>& A,
    TriangularFactors<std::complex<float> > T,
    Matrix< std::complex<float> >& C,
    Options const& opts);

template
void unmtr_he2hb<std::complex<double>>(
    Side side, Op op,
    HermitianMatrix<std::complex<double>>& A,
    TriangularFactors<std::complex<double>> T,
    Matrix<std::complex<double>>& C,
    Options const& opts);

} // namespace slate
