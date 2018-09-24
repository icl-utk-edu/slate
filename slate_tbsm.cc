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

#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_Matrix.hh"
#include "slate_TriangularBandMatrix.hh"
#include "slate_internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::tbsm from internal::specialization::tbsm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel triangular matrix solve.
/// Generic implementation for any target.
/// Note A and B are passed by value, so we can transpose if needed
/// (for side = right) without affecting caller.
/// @ingroup tbsm
template <Target target, typename scalar_t>
void tbsm(slate::internal::TargetType<target>,
          Side side,
          scalar_t alpha,
          TriangularBandMatrix<scalar_t> A, Pivots& pivots,
                        Matrix<scalar_t> B,
          int64_t lookahead)
{
    using namespace blas;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // if on right, change to left by (conj)-transposing A and B to get
    // op(B) = op(A)^{-1} * op(B)
    if (side == Side::Right) {
        if (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans) {
            A = conj_transpose(A);
            B = conj_transpose(B);
            alpha = conj(alpha);
        }
        else {
            A = transpose(A);
            B = transpose(B);
        }
    }

    // B is mt-by-nt, A is mt-by-mt (assuming side = left)
    assert(A.mt() == B.mt());
    assert(A.nt() == B.mt());

    int64_t mt = B.mt();
    int64_t nt = B.nt();

    if (target == Target::Devices) {
        B.allocateBatchArrays();
        B.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> row_vector(A.nt());
    uint8_t* row = row_vector.data();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t kdt = ceildiv( A.bandwidth(), A.tileNb(0) );

    const scalar_t one = 1.0;

    #pragma omp parallel
    #pragma omp master
    {
        if (alpha != one) {
            // Scale B = alpha B.
            // Due to the band, this can't be done in trsm & gemm tasks
            // (at least, not without splitting gemms).
            int64_t B_mt = B.mt();
            int64_t B_nt = B.nt();
            for (int64_t i = 0; i < B_mt; ++i) {
                #pragma omp task depend(inout:row[i]) priority(1)
                {
                    // No batched routine; use host-nest implementation.
                    // todo: make internal::scale routine and device implementation.
                    #pragma omp parallel for schedule(dynamic, 1)
                    for (int64_t j = 0; j < B_nt; ++j) {
                        if (B.tileIsLocal(i, j)) {
                            B.tileMoveToHost(i, j, B.tileDevice(i, j));
                            scale(alpha, B(i, j));
                        }
                    }
                    #pragma omp taskwait
                }
            }
        }

        if (A.uplo_logical() == Uplo::Lower) {
            // ----------------------------------------
            // Lower/NoTrans or Upper/Trans, Left case
            // Forward sweep
            for (int64_t k = 0; k < mt; ++k) {
                // A( k:i_end-1, k ) is the panel
                // Compared to trsm, i_end replaces mt.
                // "end" in the usual C++ sense of entry after the last entry.
                int64_t i_end = min(k + kdt + 1, mt);

                if (! pivots.empty()) {
                    // swap rows in B(k:mt-1, 0:nt-1)
                    // Pivots need to lock the whole rest of the B matrix.
                    #pragma omp taskwait
                    internal::swap<Target::HostTask>(
                        Direction::Forward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                        pivots.at(k));
                    #pragma omp taskwait
                }

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // send A(k, k) to ranks owning block row B(k, :)
                    A.template tileBcast(k, k, B.sub(k, k, 0, nt-1));

                    // solve A(k, k) B(k, :) = B(k, :)
                    internal::trsm<Target::HostTask>(
                        Side::Left,
                        one, A.sub(k, k),
                             B.sub(k, k, 0, nt-1), 1);

                    // send A(i=k+1:i_end-1, k) to ranks owning block row B(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = k+1; i < i_end; ++i)
                        bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                    A.template listBcast<target>(bcast_list_A);

                    // send B(k, j=0:nt-1) to ranks owning
                    // block col B(k+1:i_end-1, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < nt; ++j) {
                        bcast_list_B.push_back(
                            {k, j, {B.sub(k+1, i_end-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B);
                }

                // lookahead update, B(k+1:k+la, :) -= A(k+1:k+la, k) B(k, :)
                for (int64_t i = k+1; i < k+1+lookahead && i < i_end; ++i) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[i]) priority(1)
                    {
                        internal::gemm<Target::HostTask>(
                            -one, A.sub(i, i, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(i, i, 0, nt-1), 1);
                    }
                }

                // trailing update,
                // B(k+1+la:i_end-1, :) -= A(k+1+la:i_end-1, k) B(k, :)
                // Updates rows k+1+la to i_end-1, but two depends are sufficient:
                // depend on k+1+la is all that is needed in next iteration;
                // depend on mt-1 daisy chains all the trailing updates.
                if (k+1+lookahead < i_end) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[k+1+lookahead]) \
                                     depend(inout:row[mt-1])
                    {
                        internal::gemm<target>(
                            -one, A.sub(k+1+lookahead, i_end-1, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(k+1+lookahead, i_end-1, 0, nt-1));
                    }
                }
            }
        }
        else {
            // ----------------------------------------
            // Upper/NoTrans or Lower/Trans, Left case
            // Backward sweep
            for (int64_t k = mt-1; k >= 0; --k) {
                // A( i_begin:k, k ) is the panel
                int64_t i_begin = max(k - kdt, 0);  // todo: was: min(k - kdt, mt);

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // send A(k, k) to ranks owning block row B(k, :)
                    A.template tileBcast(k, k, B.sub(k, k, 0, nt-1));

                    // solve A(k, k) B(k, :) = B(k, :)
                    internal::trsm<Target::HostTask>(
                        Side::Left,
                        one, A.sub(k, k),
                             B.sub(k, k, 0, nt-1), 1);

                    // send A(i=k-kdt:k-1, k) to ranks owning block row B(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = i_begin; i < k; ++i)
                        bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                    A.template listBcast<target>(bcast_list_A);

                    // send B(k, j=0:nt-1) to ranks owning block col B(k-kdt:k-1, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < nt; ++j)
                        bcast_list_B.push_back({k, j, {B.sub(i_begin, k-1, j, j)}});
                    B.template listBcast<target>(bcast_list_B);
                }

                if (! pivots.empty()) {
                    // swap rows in B(k:mt-1, 0:nt-1)
                    // Swaps need to lock the whole rest of the B matrix.
                    #pragma omp taskwait
                    internal::swap<Target::HostTask>(
                        Direction::Backward, B.sub(k, B.mt()-1, 0, B.nt()-1),
                        pivots.at(k));
                    #pragma omp taskwait
                }

                // lookahead update, B(k-la:k-1, :) -= A(k-la:k-1, k) B(k, :)
                for (int64_t i = k-1; i > k-1-lookahead && i >= i_begin; --i) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[i]) priority(1)
                    {
                        internal::gemm<Target::HostTask>(
                            -one, A.sub(i, i, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(i, i, 0, nt-1), 1);
                    }
                }

                // trailing update, B(0:k-1-la, :) -= A(0:k-1-la, k) B(k, :)
                // Updates rows 0 to k-1-la, but two depends are sufficient:
                // depend on k-1-la is all that is needed in next iteration;
                // depend on 0 daisy chains all the trailing updates.
                if (k-1-lookahead >= i_begin) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[k-1-lookahead]) \
                                     depend(inout:row[0])
                    {
                        internal::gemm<target>(
                            -one, A.sub(i_begin, k-1-lookahead, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(i_begin, k-1-lookahead, 0, nt-1));
                    }
                }
            }
        }
    }

    B.moveAllToOrigin();
    B.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup tbsm
template <Target target, typename scalar_t>
void tbsm(blas::Side side,
          scalar_t alpha,
          TriangularBandMatrix<scalar_t>& A, Pivots& pivots,
                        Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::tbsm(internal::TargetType<target>(),
                                   side,
                                   alpha, A, pivots,
                                          B,
                                   lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel triangular band matrix-matrix solve.
/// Solves one of the triangular matrix equations
/// \[
///     A X = \alpha B,
/// \]
/// or
/// \[
///     X A = \alpha B,
/// \]
/// where alpha is a scalar, B is an m-by-n matrix and A is a unit or non-unit,
/// upper or lower triangular band matrix. The matrix X overwrites B.
/// Pivoting from tbtrf is applied during the solve.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     slate::tbsm( Side::Left, alpha, AT, pivots, B );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether A appears on the left or on the right of X:
///         - Side::Left:  solve $A X = \alpha B$
///         - Side::Right: solve $X A = \alpha B$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m triangular matrix A;
///         - if side = right, the n-by-n triangular matrix A.
///
/// @param[in] pivots
///         Pivot information from gbtrf.
///         If pivots is an empty vector, no pivoting is applied.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result X.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of panels to overlap with matrix updates.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup tbsm
template <typename scalar_t>
void tbsm(blas::Side side,
          scalar_t alpha,
          TriangularBandMatrix<scalar_t>& A, Pivots& pivots,
                        Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            tbsm<Target::HostTask>(side, alpha, A, pivots, B, opts);
            break;
        case Target::HostNest:
            tbsm<Target::HostNest>(side, alpha, A, pivots, B, opts);
            break;
        case Target::HostBatch:
            tbsm<Target::HostBatch>(side, alpha, A, pivots, B, opts);
            break;
        case Target::Devices:
            tbsm<Target::Devices>(side, alpha, A, pivots, B, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tbsm<float>(
    blas::Side side,
    float alpha,
    TriangularBandMatrix<float>& A, Pivots& pivots,
                  Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void tbsm<double>(
    blas::Side side,
    double alpha,
    TriangularBandMatrix<double>& A, Pivots& pivots,
                  Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void tbsm< std::complex<float> >(
    blas::Side side,
    std::complex<float> alpha,
    TriangularBandMatrix< std::complex<float> >& A, Pivots& pivots,
                  Matrix< std::complex<float> >& B,
    const std::map<Option, Value>& opts);

template
void tbsm< std::complex<double> >(
    blas::Side side,
    std::complex<double> alpha,
    TriangularBandMatrix< std::complex<double> >& A, Pivots& pivots,
                  Matrix< std::complex<double> >& B,
    const std::map<Option, Value>& opts);

} // namespace slate
