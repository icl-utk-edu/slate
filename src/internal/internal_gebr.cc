// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/Matrix.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"
#include "internal/internal_householder.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Generates a Householder reflector $H = I - \tau v v^H$ using the first
/// column of the matrix $A$, i.e., a reflector that zeroes $A[1:m-1, 0]$.
/// Stores $\tau$ in $v[0]$.
///
/// @param[in] A
///     The m-by-n matrix A.
///
/// @param[out] v
///     The vector v in the representation of H.
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void gerfg(Matrix<scalar_t>& A, int64_t n, scalar_t* v)
{
    using blas::conj;

    assert(n == A.m());

    // v <- A[:, 0]
    scalar_t* vi = v;
    for (int64_t i = 0; i < A.mt(); ++i) {
        auto tile = A(i, 0);
        if (tile.op() == Op::ConjTrans || tile.op() == Op::Trans) {
            int64_t mb = tile.mb();
            scalar_t* t_ptr = tile.data();
            for (int64_t j = 0; j < mb; ++j) {
                vi[j] = conj( t_ptr[ j * tile.stride() ] );
            }
        }
        else {
            blas::copy(tile.mb(), tile.data(),
                       tile.op() == Op::NoTrans ? 1 : tile.stride(),
                       vi, 1);
        }
        vi += tile.mb();
    }

    // Compute the reflector in v.
    // Store tau in v[0].
    scalar_t tau;
    lapack::larfg(n, v, &v[1], 1, &tau);
    v[0] = tau;
}

//------------------------------------------------------------------------------
/// Applies a Houselolder reflector $H = I - \tau v v^H$ to the matrix $A$
/// from the left. Takes the $\tau$ factor from $v[0]$.
///
/// @param[in] n
///     Length of vector v.
///
/// @param[in] v
///     The vector v in the representation of H.
///     Modified but restored on exit.
///
/// @param[in,out] A
///     The m-by-n matrix A.
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void gerf(int64_t n, scalar_t* v, Matrix<scalar_t>& A)
{
    using blas::conj;

    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    // Replace tau with 1.0 in v[0].
    scalar_t tau = v[0];
    v[0] = one;

    // w = A^H v
    auto AH = conjTranspose(A);
    std::vector<scalar_t> w(AH.m());

    scalar_t* wi = w.data();
    for (int64_t i = 0; i < AH.mt(); ++i) {
        scalar_t* vj = v;
        scalar_t beta = zero;
        for (int64_t j = 0; j < AH.nt(); ++j) {
            tile::gemv( one, AH(i, j), vj, beta, wi );
            beta = one;
            vj += AH.tileNb(j);
        }
        wi += AH.tileMb(i);
    }

    // A = A - v w^H
    scalar_t* vi = v;
    for (int64_t i = 0; i < A.mt(); ++i) {
        scalar_t* wj = w.data();
        for (int64_t j = 0; j < A.nt(); ++j) {
            tile::ger( -tau, vi, wj, A(i, j) );
            wj += A.tileNb(j);
        }
        vi += A.tileMb(i);
    }

    // Restore v[0].
    v[0] = tau;
}

//------------------------------------------------------------------------------
/// Implements task 1 in the bidiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
/// @ingroup svd_computational
///
template <Target target, typename scalar_t>
void gebr1(Matrix<scalar_t>&& A,
           int64_t n1, scalar_t* v1,
           int64_t n2, scalar_t* v2,
           int priority)
{
    gebr1(internal::TargetType<target>(),
          A, n1, v1, n2, v2, priority);
}

//------------------------------------------------------------------------------
/// Implements task 1 in the bidiagonal bulge chasing algorithm.
/// (see https://doi.org/10.1137/17M1117732
/// and http://www.icl.utk.edu/publications/swan-013)
///
/// @param[in,out] A
///     The first block of a sweep.
///
/// @param[out] v1
///     The Householder reflector to zero A[0, 1:n-1].
///
/// @param[out] v2
///     The Householder reflector to zero A[2:m-1, 0].
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void gebr1(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A,
           int64_t n1, scalar_t* v1,
           int64_t n2, scalar_t* v2,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::gebr1");

    // Zero A[0, 1:n-1].
    // Apply A Q^H => becomes Q A^H.
    auto A1 = conjTranspose(A);
    gerfg(A1, n1, v1);
    gerf(n1, v1, A1);

    // Zero A[2:m-1, 0].
    // Apply Q^H A => conjugate tau.
    auto A2 = A.slice(1, A.m()-1, 0, A.n()-1);
    gerfg(A2, n2, v2);
    v2[0] = conj( v2[0] );
    gerf(n2, v2, A2);
}

//------------------------------------------------------------------------------
/// Implements task 2 in the bidiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
/// @ingroup svd_computational
///
template <Target target, typename scalar_t>
void gebr2(int64_t n1, scalar_t* v1,
           Matrix<scalar_t>&& A,
           int64_t n2, scalar_t* v2,
           int priority)
{
    gebr2(internal::TargetType<target>(),
          n1, v1, A, n2, v2, priority);
}

//------------------------------------------------------------------------------
/// Implements task 2 in the bidiagonal bulge chasing algorithm.
///
/// @param[in] v1
///     The second Householder reflector produced by task 1.
///
/// @param[in,out] A
///     An off-diagonal block in a sweep.
///
/// @param[out] v2
///     The Householder reflector to zero A[0, 1:n-1].
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void gebr2(internal::TargetType<Target::HostTask>,
           int64_t n1, scalar_t* v1,
           Matrix<scalar_t>& A,
           int64_t n2, scalar_t* v2,
           int priority)
{
    trace::Block trace_block("internal::gebr2");

    // Apply the second reflector from task 1: Q^H A.
    gerf(n1, v1, A);

    // Zero A[0, 1:n-1].
    // Apply A Q^H => becomes Q A^H.
    auto AH = conjTranspose(A);
    gerfg(AH, n2, v2);
    gerf(n2, v2, AH);
}

//------------------------------------------------------------------------------
/// Implements task 3 in the bidiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
/// @ingroup svd_computational
///
template <Target target, typename scalar_t>
void gebr3(int64_t n1, scalar_t* v1,
           Matrix<scalar_t>&& A,
           int64_t n2, scalar_t* v2,
           int priority)
{
    gebr3(internal::TargetType<target>(),
          n1, v1, A, n2, v2, priority);
}

//------------------------------------------------------------------------------
/// Implements task 3 in the bidiagonal bulge chasing algorithm.
///
/// @param[in] v1
///     The Householder reflector produced by task 2.
///
/// @param[in,out] A
///     A diagonal block in a sweep.
///
/// @param[out] v2
///     The Householder reflector to zero A[1:m-1, 0].
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void gebr3(internal::TargetType<Target::HostTask>,
           int64_t n1, scalar_t* v1,
           Matrix<scalar_t>& A,
           int64_t n2, scalar_t* v2,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::gebr3");

    // Apply the reflector from task 2: Q A.
    auto AH = conjTranspose(A);
    gerf(n1, v1, AH);

    // Zero A[1:m-1, 0].
    // Apply Q^H A => conjugate tau.
    gerfg(A, n2, v2);
    v2[0] = conj( v2[0] );
    gerf(n2, v2, A);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void gerfg(Matrix<float>& A, int64_t n, float* v);

template
void gerfg(Matrix<double>& A, int64_t n, double* v);

template
void gerfg(Matrix<std::complex<float>>& A, int64_t n, std::complex<float>* v);

template
void gerfg(Matrix<std::complex<double>>& A, int64_t n, std::complex<double>* v);

// ----------------------------------------
template
void gerf(int64_t n, float* v, Matrix<float>& A);

template
void gerf(int64_t n, double* v, Matrix<double>& A);

template
void gerf(int64_t n, std::complex<float>* v, Matrix<std::complex<float>>& A);

template
void gerf(int64_t n, std::complex<double>* v, Matrix<std::complex<double>>& A);

// ----------------------------------------
template
void gebr1<Target::HostTask, float>(
    Matrix<float>&& A,
    int64_t n1, float* v1,
    int64_t n2, float* v2,
    int priority);

template
void gebr1<Target::HostTask, double>(
    Matrix<double>&& A,
    int64_t n1, double* v1,
    int64_t n2, double* v2,
    int priority);

template
void gebr1< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    int64_t n1, std::complex<float>* v1,
    int64_t n2, std::complex<float>* v2,
    int priority);

template
void gebr1< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    int64_t n1, std::complex<double>* v1,
    int64_t n2, std::complex<double>* v2,
    int priority);

// ----------------------------------------
template
void gebr2<Target::HostTask, float>(
    int64_t n1, float* v1,
    Matrix<float>&& A,
    int64_t n2, float* v2,
    int priority);

template
void gebr2<Target::HostTask, double>(
    int64_t n1, double* v1,
    Matrix<double>&& A,
    int64_t n2, double* v2,
    int priority);

template
void gebr2< Target::HostTask, std::complex<float> >(
    int64_t n1, std::complex<float>* v1,
    Matrix< std::complex<float> >&& A,
    int64_t n2, std::complex<float>* v2,
    int priority);

template
void gebr2< Target::HostTask, std::complex<double> >(
    int64_t n1, std::complex<double>* v1,
    Matrix< std::complex<double> >&& A,
    int64_t n2, std::complex<double>* v2,
    int priority);

// ----------------------------------------
template
void gebr3<Target::HostTask, float>(
    int64_t n1, float* v1,
    Matrix<float>&& A,
    int64_t n2, float* v2,
    int priority);

template
void gebr3<Target::HostTask, double>(
    int64_t n1, double* v1,
    Matrix<double>&& A,
    int64_t n2, double* v2,
    int priority);

template
void gebr3< Target::HostTask, std::complex<float> >(
    int64_t n1, std::complex<float>* v1,
    Matrix< std::complex<float> >&& A,
    int64_t n2, std::complex<float>* v2,
    int priority);

template
void gebr3< Target::HostTask, std::complex<double> >(
    int64_t n1, std::complex<double>* v1,
    Matrix< std::complex<double> >&& A,
    int64_t n2, std::complex<double>* v2,
    int priority);

} // namespace internal
} // namespace slate
