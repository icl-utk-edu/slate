// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Generates a Householder reflector $v$ using the first column
/// of the matrix $A$, i.e., a reflector that zeroes $A[1:m-1, 0]$.
/// Stores $tau$ in $v[0]$.
///
/// @param[in] A
///     The m-by-n matrix A.
///
/// @param[out] v
///     The Householder reflector that zeroes $A[1:m-1, 0]$.
///
template <typename scalar_t>
void gerfg(Matrix<scalar_t>& A, std::vector<scalar_t>& v)
{
    using blas::conj;

    // v <- A[:, 0]
    v.resize(A.m());
    scalar_t* v_ptr = v.data();
    for (int64_t i = 0; i < A.mt(); ++i) {
        auto tile = A(i, 0);
        if (tile.op() == Op::ConjTrans ||
            tile.op() == Op::Trans) {
            int64_t mb = tile.mb();
            scalar_t* t_ptr = tile.data();
            for (int64_t j = 0; j < mb; ++j) {
                v_ptr[j] = conj( t_ptr[ j * tile.stride() ] );
            }
        }
        else {
            blas::copy(tile.mb(), tile.data(),
                       tile.op() == Op::NoTrans ? 1 : tile.stride(),
                       v_ptr, 1);
        }
        v_ptr += tile.mb();
    }

    // Compute the reflector in v.
    // Store tau in v[0].
    scalar_t tau;
    lapack::larfg(v.size(), v.data(), v.data()+1, 1, &tau);
    v[0] = tau;
}

//------------------------------------------------------------------------------
/// Applies a Houselolder reflector $v$ to the matrix $A$ from the left.
/// Takes the $tau$ factor from $v[0]$.
///
/// @param[in] in_v
///     The Householder reflector to apply.
///
/// @param[in,out] A
///     The m-by-n matrix A.
///
template <typename scalar_t>
void gerf(std::vector<scalar_t> const& in_v, Matrix<scalar_t>& A)
{
    using blas::conj;
    scalar_t one  = 1;
    scalar_t zero = 0;

    // Replace tau with 1.0 in v[0].
    auto v = in_v;
    scalar_t tau = v[0];
    v[0] = one;

    // w = C^H * v
    auto AH = conjTranspose(A);
    std::vector<scalar_t> w(AH.m());

    scalar_t* w_ptr = w.data();
    for (int64_t i = 0; i < AH.mt(); ++i) {
        scalar_t* v_ptr = v.data();
        scalar_t beta = zero;
        for (int64_t j = 0; j < AH.nt(); ++j) {
            gemv(one, AH(i, j), v_ptr, beta, w_ptr);
            beta = one;
            v_ptr += AH.tileNb(j);
        }
        w_ptr += AH.tileMb(i);
    }

    // C = C - v * w^H
    scalar_t* v_ptr = v.data();
    for (int64_t i = 0; i < A.mt(); ++i) {
        w_ptr = w.data();
        for (int64_t j = 0; j < A.nt(); ++j) {
            ger(-tau, v_ptr, w_ptr, A(i, j));
            w_ptr += A.tileNb(j);
        }
        v_ptr += A.tileMb(i);
    }
}

//------------------------------------------------------------------------------
/// Implements task 1 in the bidiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
template <Target target, typename scalar_t>
void gebr1(Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v1,
           std::vector<scalar_t>& v2,
           int priority)
{
    gebr1(internal::TargetType<target>(),
          A, v1, v2, priority);
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
template <typename scalar_t>
void gebr1(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A,
           std::vector<scalar_t>& v1,
           std::vector<scalar_t>& v2,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::gebr1");

    // Zero A[0, 1:n-1].
    // Apply A*Q^H => becomes Q*A^H.
    auto A1 = conjTranspose(A);
    gerfg(A1, v1);
    gerf(v1, A1);

    // Zero A[2:m-1, 0].
    // Apply Q^H*A => conjugate tau.
    auto A2 = A.slice(1, A.m()-1, 0, A.n()-1);
    gerfg(A2, v2);
    v2[0] = conj( v2[0] );
    gerf(v2, A2);
}

//------------------------------------------------------------------------------
/// Implements task 2 in the bidiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
template <Target target, typename scalar_t>
void gebr2(std::vector<scalar_t> const& v1,
           Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v2,
           int priority)
{
    gebr2(internal::TargetType<target>(),
          v1, A, v2, priority);
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
template <typename scalar_t>
void gebr2(internal::TargetType<Target::HostTask>,
           std::vector<scalar_t> const& v1,
           Matrix<scalar_t>& A,
           std::vector<scalar_t>& v2,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::gebr2");

    // Apply the second reflector from task 1: Q^H*A.
    gerf(v1, A);

    // Zero A[0, 1:n-1].
    // Apply A*Q^H => becomes Q*A^H.
    auto AH = conjTranspose(A);
    gerfg(AH, v2);
    gerf(v2, AH);
}

//------------------------------------------------------------------------------
/// Implements task 3 in the bidiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
template <Target target, typename scalar_t>
void gebr3(std::vector<scalar_t> const& v1,
           Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v2,
           int priority)
{
    gebr3(internal::TargetType<target>(),
          v1, A, v2, priority);
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
template <typename scalar_t>
void gebr3(internal::TargetType<Target::HostTask>,
           std::vector<scalar_t> const& v1,
           Matrix<scalar_t>& A,
           std::vector<scalar_t>& v2,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::gebr3");

    // Apply the reflector from task 2: Q*A.
    auto AH = conjTranspose(A);
    gerf(v1, AH);

    // Zero A[1:m-1, 0].
    // Apply Q^H*A => conjugate tau.
    gerfg(A, v2);
    v2[0] = conj( v2[0] );
    gerf(v2, A);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void gebr1<Target::HostTask, float>(
    Matrix<float>&& A,
    std::vector<float>& v1,
    std::vector<float>& v2,
    int priority);

template
void gebr1<Target::HostTask, double>(
    Matrix<double>&& A,
    std::vector<double>& v1,
    std::vector<double>& v2,
    int priority);

template
void gebr1< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    std::vector< std::complex<float> >& v1,
    std::vector< std::complex<float> >& v2,
    int priority);

template
void gebr1< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A,
    std::vector< std::complex<double> >& v1,
    std::vector< std::complex<double> >& v2,
    int priority);

// ----------------------------------------
template
void gebr2<Target::HostTask, float>(
    std::vector<float> const& v1,
    Matrix<float>&& A,
    std::vector<float>& v2,
    int priority);

template
void gebr2<Target::HostTask, double>(
    std::vector<double> const& v1,
    Matrix<double>&& A,
    std::vector<double>& v2,
    int priority);

template
void gebr2< Target::HostTask, std::complex<float> >(
    std::vector< std::complex<float> > const& v1,
    Matrix< std::complex<float> >&& A,
    std::vector< std::complex<float> >& v2,
    int priority);

template
void gebr2< Target::HostTask, std::complex<double> >(
    std::vector< std::complex<double> > const& v1,
    Matrix< std::complex<double> >&& A,
    std::vector< std::complex<double> >& v2,
    int priority);

// ----------------------------------------
template
void gebr3<Target::HostTask, float>(
    std::vector<float> const& v1,
    Matrix<float>&& A,
    std::vector<float>& v2,
    int priority);

template
void gebr3<Target::HostTask, double>(
    std::vector<double> const& v1,
    Matrix<double>&& A,
    std::vector<double>& v2,
    int priority);

template
void gebr3< Target::HostTask, std::complex<float> >(
    std::vector< std::complex<float> > const& v1,
    Matrix< std::complex<float> >&& A,
    std::vector< std::complex<float> >& v2,
    int priority);

template
void gebr3< Target::HostTask, std::complex<double> >(
    std::vector< std::complex<double> > const& v1,
    Matrix< std::complex<double> >&& A,
    std::vector< std::complex<double> >& v2,
    int priority);

} // namespace internal
} // namespace slate
