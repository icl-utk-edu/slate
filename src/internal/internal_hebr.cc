// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/HermitianMatrix.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"
#include "internal/internal_householder.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Applies a Householder reflector $H = I - \tau v v^H$ to the Hermitian
/// matrix $A$ on the left and right. Takes the $\tau$ factor from $v[0]$.
///
/// @param[in] n
///     Length of vector v.
///
/// @param[in] v
///     The vector v in the representation of H.
///     Modified but restored on exit.
///
/// @param[in,out] A
///     The n-by-n Hermitian matrix A.
///
/// @ingroup heev_computational
///
template <typename scalar_t>
void herf(int64_t n, scalar_t* v, HermitianMatrix<scalar_t>& A)
{
    using blas::conj;

    const scalar_t one  = 1.0;
    const scalar_t zero = 0.0;
    const scalar_t half = 0.5;

    // todo: seems odd to conj tau here. Maybe gerfg isn't generating tau right?
    // Replace tau with 1.0 in v[0].
    scalar_t tau = conj(v[0]);
    v[0] = one;

    scalar_t *wi, *vi;

    // w = A v
    // todo: HermitianMatrix::at(i, j) can be extended to support access
    // to the (nonexistent) symmetric part by returning transpose(at(j, i)).
    // This will allow removing the if/else condition.
    // The first call to gemv() will support both cases.
    std::vector<scalar_t> w_vec(A.n());
    scalar_t* w = w_vec.data();
    wi = w;
    for (int64_t i = 0; i < A.nt(); ++i) {
        scalar_t* vj = v;
        scalar_t beta = zero;
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (i == j) {
                tile::hemv( one, A(i, j), vj, beta, wi );
            }
            else {
                auto Aij = i > j
                         ? A(i, j)
                         : conjTranspose(A(j, i));
                tile::gemv( one, Aij, vj, beta, wi );
            }
            beta = one;
            vj += A.tileNb(j);
        }
        wi += A.tileMb(i);
    }

    // todo: should this be switched, v^H w instead of w^H v?
    // w = A v - 0.5 tau ((A v)^H v) v
    scalar_t alpha = -half * tau * blas::dot(A.n(), w, 1, v, 1);
    blas::axpy(A.n(), alpha, v, 1, w, 1);

    // A = A - tau v w^H - conj(tau) w v^H, lower triangle
    vi = v;
    wi = w;
    for (int64_t i = 0; i < A.nt(); ++i) {
        scalar_t* vj = v;
        scalar_t* wj = w;
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (i > j) {  // lower
                tile::ger( -tau,       vi, wj, A(i, j) );
                tile::ger( -conj(tau), wi, vj, A(i, j) );
            }
            else if (i == j) {  // diag
                tile::her2( -tau, vi, wj, A(i, j) );
            }
            vj += A.tileNb(j);
            wj += A.tileNb(j);
        }
        vi += A.tileMb(i);
        wi += A.tileMb(i);
    }

    // Restore v[0].
    v[0] = conj(tau);
}

//------------------------------------------------------------------------------
/// Implements task type 1 in the tridiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
/// @ingroup heev_computational
///
template <Target target, typename scalar_t>
void hebr1(int64_t n, scalar_t* v,
           HermitianMatrix<scalar_t>&& A,
           int priority)
{
    hebr1(internal::TargetType<target>(),
          n, v, A, priority);
}

//------------------------------------------------------------------------------
/// Implements task type 1 in the tridiagonal bulge chasing algorithm,
/// bringing the first column & row of A to tridiagonal.
/// See https://doi.org/10.1145/2063384.2063394
/// and http://www.icl.utk.edu/publications/swan-013
/// Here, the first block starts at $(0, 0)$, not at $(1, 0)$.
/// todo: as compared to SVD?
///
/// @param[in] n
///     Length of vector v.
///
/// @param[out] v
///     The Householder reflector to zero A[2:n-1, 0].
///
/// @param[in,out] A
///     The first block of a sweep.
///
template <typename scalar_t>
void hebr1(internal::TargetType<Target::HostTask>,
           int64_t n, scalar_t* v,
           HermitianMatrix<scalar_t>& A,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::hebr1");

    // Zero A[2:n-1, 0].
    auto A1 = A.slice(1, A.m()-1, 0, 0);
    gerfg(A1, n, v);

    // todo: this is silly; we already know it zeros the column.
    v[0] = conj(v[0]);
    gerf(n, v, A1);
    v[0] = conj(v[0]);

    // Apply the 2-sided transformation to A[1:n-1, 1:n-1].
    auto A2 = A.slice(1, A.n()-1);
    herf(n, v, A2);
}

//------------------------------------------------------------------------------
/// Implements task type 2 in the tridiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
/// @ingroup heev_computational
///
template <Target target, typename scalar_t>
void hebr2(int64_t n1, scalar_t* v1,
           int64_t n2, scalar_t* v2,
           Matrix<scalar_t>&& A,
           int priority)
{
    hebr2(internal::TargetType<target>(),
          n1, v1, n2, v2, A, priority);
}

//------------------------------------------------------------------------------
/// Implements task type 2 in the tridiagonal bulge chasing algorithm,
/// updating an off-diagonal block, which creates a bulge, then bringing its
/// first column back to the original bandwidth.
///
/// @param[in] n1
///     Length of vector v1.
///
/// @param[in] v1
///     The Householder reflector produced by task type 1 or 2.
///
/// @param[in] n2
///     Length of vector v2.
///
/// @param[out] v2
///     The Householder reflector to zero A[1:n-1, 0].
///
/// @param[in,out] A
///     An off-diagonal block in a sweep.
///
template <typename scalar_t>
void hebr2(internal::TargetType<Target::HostTask>,
           int64_t n1, scalar_t* v1,
           int64_t n2, scalar_t* v2,
           Matrix<scalar_t>& A,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::hebr2");

    // Apply the reflector from task type 1 or 2.
    auto AH = conjTranspose(A);
    gerf(n1, v1, AH);

    // Zero A[1:n-1, 0].
    gerfg(A, n2, v2);
    v2[0] = conj(v2[0]);
    gerf(n2, v2, A);
    v2[0] = conj(v2[0]);
}

//------------------------------------------------------------------------------
/// Implements task type 3 in the tridiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
/// @ingroup heev_computational
///
template <Target target, typename scalar_t>
void hebr3(int64_t n, scalar_t* v,
           HermitianMatrix<scalar_t>&& A,
           int priority)
{
    hebr3(internal::TargetType<target>(),
          n, v, A, priority);
}

//------------------------------------------------------------------------------
/// Implements task type 3 in the tridiagonal bulge chasing algorithm,
/// updating a diagonal block with a 2-sided Householder transformation.
///
/// @param[in] n
///     Length of vector v.
///
/// @param[in] v
///     The Householder reflector produced by task type 2.
///
/// @param[in,out] A
///     A diagonal block in a sweep.
///
/// @ingroup heev_computational
///
template <typename scalar_t>
void hebr3(internal::TargetType<Target::HostTask>,
           int64_t n, scalar_t* v,
           HermitianMatrix<scalar_t>& A,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::hebr3");

    // Apply the reflector from task type 2.
    herf(n, v, A);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void hebr1<Target::HostTask, float>(
    int64_t n, float* v1,
    HermitianMatrix<float>&& A,
    int priority);

template
void hebr1<Target::HostTask, double>(
    int64_t n, double* v1,
    HermitianMatrix<double>&& A,
    int priority);

template
void hebr1< Target::HostTask, std::complex<float> >(
    int64_t n, std::complex<float>* v1,
    HermitianMatrix< std::complex<float> >&& A,
    int priority);

template
void hebr1< Target::HostTask, std::complex<double> >(
    int64_t n, std::complex<double>* v1,
    HermitianMatrix< std::complex<double> >&& A,
    int priority);

// ----------------------------------------
template
void hebr2<Target::HostTask, float>(
    int64_t n1, float* v1,
    int64_t n2, float* v2,
    Matrix<float>&& A,
    int priority);

template
void hebr2<Target::HostTask, double>(
    int64_t n1, double* v1,
    int64_t n2, double* v2,
    Matrix<double>&& A,
    int priority);

template
void hebr2< Target::HostTask, std::complex<float> >(
    int64_t n1, std::complex<float>* v1,
    int64_t n2, std::complex<float>* v2,
    Matrix< std::complex<float> >&& A,
    int priority);

template
void hebr2< Target::HostTask, std::complex<double> >(
    int64_t n1, std::complex<double>* v1,
    int64_t n2, std::complex<double>* v2,
    Matrix< std::complex<double> >&& A,
    int priority);

// ----------------------------------------
template
void hebr3<Target::HostTask, float>(
    int64_t n, float* v,
    HermitianMatrix<float>&& A,
    int priority);

template
void hebr3<Target::HostTask, double>(
    int64_t n, double* v,
    HermitianMatrix<double>&& A,
    int priority);

template
void hebr3< Target::HostTask, std::complex<float> >(
    int64_t n, std::complex<float>* v,
    HermitianMatrix< std::complex<float> >&& A,
    int priority);

template
void hebr3< Target::HostTask, std::complex<double> >(
    int64_t n, std::complex<double>* v,
    HermitianMatrix< std::complex<double> >&& A,
    int priority);

} // namespace internal
} // namespace slate
