//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "slate/internal/util.hh"
#include "slate/HermitianMatrix.hh"
#include "internal/Tile_lapack.hh"
#include "slate/types.hh"

namespace slate {
namespace internal {

// todo: These functions are defined in internal_gebr.cc.
// It would be better to move the declarations to a header file.
template <typename scalar_t>
void gerfg(Matrix<scalar_t>& A, std::vector<scalar_t>& v);

template <typename scalar_t>
void gerf(std::vector<scalar_t> const& in_v, Matrix<scalar_t>& A);

//------------------------------------------------------------------------------
/// Applies a Houselolder reflector $v$ to the Hermitian matrix $A$
/// from the left. Takes the $tau$ factor from $v[0]$.
///
/// @param[in] in_v
///     The Householder reflector to apply.
///
/// @param[in,out] A
///     The n-by-n Hermitian matrix A.
///
template <typename scalar_t>
void herf(std::vector<scalar_t> const& in_v, HermitianMatrix<scalar_t>& A)
{
    using blas::conj;
    scalar_t one  = 1;
    scalar_t zero = 0;

    // Replace tau with 1.0 in v[0].
    auto v = in_v;
    scalar_t tau = conj(v[0]);
    v[0] = one;

    // w = C * v
    // todo: HermitianMatrix::at(i, j) can be extended to support access
    // to the (nonexistent) symmetric part by returning transpose(at(j, i)).
    // This will allow to remove the if/else condition.
    // The first call to gemv() will support both cases.
    std::vector<scalar_t> w(A.n());
    scalar_t* w_ptr = w.data();
    for (int64_t i = 0; i < A.nt(); ++i) {
        scalar_t* v_ptr = v.data();
        scalar_t beta = zero;
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (i == j) {
                hemv(one, A(i, j), v_ptr, beta, w_ptr);
            }
            else {
                if (i > j) {
                    gemv(one, A(i, j), v_ptr, beta, w_ptr);
                }
                else {
                    gemv(one, conjTranspose(A(j, i)), v_ptr, beta, w_ptr);
                }
            }
            beta = one;
            v_ptr += A.tileNb(j);
        }
        w_ptr += A.tileMb(i);
    }

    scalar_t alpha =
        scalar_t(-0.5)*tau*blas::dot(A.n(), w.data(), 1, v.data(), 1);
    blas::axpy(A.n(), alpha, v.data(), 1, w.data(), 1);

    // todo: In principle the entire update of C could be done in one pass
    // instead of three passes.

    // C = C - v * w^H (excluding diagonal tiles)
    scalar_t* v_ptr = v.data();
    for (int64_t i = 0; i < A.nt(); ++i) {
        w_ptr = w.data();
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (i > j) {
                ger(-tau, v_ptr, w_ptr, A(i, j));
            }
            w_ptr += A.tileNb(j);
        }
        v_ptr += A.tileMb(i);
    }

    // C = C - w * v^H (excluding diagonal tiles)
    w_ptr = w.data();
    for (int64_t i = 0; i < A.nt(); ++i) {
        v_ptr = v.data();
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (i > j) {
                ger(-conj(tau), w_ptr, v_ptr, A(i, j));
            }
            v_ptr += A.tileNb(j);
        }
        w_ptr += A.tileMb(i);
    }

    // C = C - v * w^H - w * v^H (diagonal tiles)
    v_ptr = v.data();
    for (int64_t i = 0; i < A.mt(); ++i) {
        w_ptr = w.data();
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (i == j) {
                her2(-tau, v_ptr, w_ptr, A(i, j));
            }
            w_ptr += A.tileNb(j);
        }
        v_ptr += A.tileMb(i);
    }
}

//------------------------------------------------------------------------------
/// Implements task 1 in the tridiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
template <Target target, typename scalar_t>
void hebr1(HermitianMatrix<scalar_t>&& A,
           std::vector<scalar_t>& v,
           int priority)
{
    hebr1(internal::TargetType<target>(),
          A, v, priority);
}

//------------------------------------------------------------------------------
/// Implements task 1 in the tridiagonal bulge chasing algorithm.
/// (see https://doi.org/10.1145/2063384.2063394
/// and http://www.icl.utk.edu/publications/swan-013)
/// Here, the first block starts at $(0, 0)$, not at $(1, 0)$.
///
/// @param[in,out] A
///     The first block of a sweep.
///
/// @param[out] v
///     The Householder reflector to zero A[2:n-1, 0].
///
template <typename scalar_t>
void hebr1(internal::TargetType<Target::HostTask>,
           HermitianMatrix<scalar_t>& A,
           std::vector<scalar_t>& v,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::hebr1");

    // Zero A[2:n-1, 0].
    auto A1 = A.slice(1, A.m()-1, 0, 0);
    gerfg(A1, v);
    scalar_t tmp = v[0];
    v[0] = conj(v[0]);
    gerf(v, A1);

    // Apply the transformations to A[1:n-1, 1:n-1].
    v[0] = tmp;
    auto A2 = A.slice(1, A.n()-1);
    herf(v, A2);
}

//------------------------------------------------------------------------------
/// Implements task 2 in the tridiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
template <Target target, typename scalar_t>
void hebr2(std::vector<scalar_t>& v1,
           Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v2,
           int priority)
{
    hebr2(internal::TargetType<target>(),
          v1, A, v2, priority);
}

//------------------------------------------------------------------------------
/// Implements task 2 in the tridiagonal bulge chasing algorithm.
///
/// @param[in] v1
///     The Householder reflector produced by task 1.
///
/// @param[in,out] A
///     An off-diagonal block in a sweep.
///
/// @param[out] v2
///     The Householder reflector to zero A[1:n-1, 0].
///
template <typename scalar_t>
void hebr2(internal::TargetType<Target::HostTask>,
           std::vector<scalar_t>& v1,
           Matrix<scalar_t>& A,
           std::vector<scalar_t>& v2,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::hebr2");

    // Apply the reflector from task 1.
    auto AH = conjTranspose(A);
    gerf(v1, AH);

    // Zero A[1:n-1, 0].
    gerfg(A, v2);
    v2[0] = conj(v2[0]);
    gerf(v2, A);
}

//------------------------------------------------------------------------------
/// Implements task 3 in the tridiagonal bulge chasing algorithm.
/// Dispatches to target implementations.
///
template <Target target, typename scalar_t>
void hebr3(std::vector<scalar_t>& v,
           HermitianMatrix<scalar_t>&& A,
           int priority)
{
    hebr3(internal::TargetType<target>(),
          v, A, priority);
}

//------------------------------------------------------------------------------
/// Implements task 3 in the tridiagonal bulge chasing algorithm.
///
/// @param[in] v
///     The Householder reflector produced by task 2.
///
/// @param[in,out] A
///     A diagonal block in a sweep.
///
template <typename scalar_t>
void hebr3(internal::TargetType<Target::HostTask>,
           std::vector<scalar_t>& v,
           HermitianMatrix<scalar_t>& A,
           int priority)
{
    using blas::conj;
    trace::Block trace_block("internal::hebr3");

    // Apply the reflector from task 2.
    v[0] = conj( v[0] );
    herf(v, A);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void hebr1<Target::HostTask, float>(
    HermitianMatrix<float>&& A,
    std::vector<float>& v1,
    int priority);

template
void hebr1<Target::HostTask, double>(
    HermitianMatrix<double>&& A,
    std::vector<double>& v1,
    int priority);

template
void hebr1< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    std::vector< std::complex<float> >& v1,
    int priority);

template
void hebr1< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    std::vector< std::complex<double> >& v1,
    int priority);

// ----------------------------------------
template
void hebr2<Target::HostTask, float>(
    std::vector<float>& v1,
    Matrix<float>&& A,
    std::vector<float>& v2,
    int priority);

template
void hebr2<Target::HostTask, double>(
    std::vector<double>& v1,
    Matrix<double>&& A,
    std::vector<double>& v2,
    int priority);

template
void hebr2< Target::HostTask, std::complex<float> >(
    std::vector< std::complex<float> >& v1,
    Matrix< std::complex<float> >&& A,
    std::vector< std::complex<float> >& v2,
    int priority);

template
void hebr2< Target::HostTask, std::complex<double> >(
    std::vector< std::complex<double> >& v1,
    Matrix< std::complex<double> >&& A,
    std::vector< std::complex<double> >& v2,
    int priority);

// ----------------------------------------
template
void hebr3<Target::HostTask, float>(
    std::vector<float>& v,
    HermitianMatrix<float>&& A,
    int priority);

template
void hebr3<Target::HostTask, double>(
    std::vector<double>& v,
    HermitianMatrix<double>&& A,
    int priority);

template
void hebr3< Target::HostTask, std::complex<float> >(
    std::vector< std::complex<float> >& v,
    HermitianMatrix< std::complex<float> >&& A,
    int priority);

template
void hebr3< Target::HostTask, std::complex<double> >(
    std::vector< std::complex<double> >& v,
    HermitianMatrix< std::complex<double> >&& A,
    int priority);

} // namespace internal
} // namespace slate
