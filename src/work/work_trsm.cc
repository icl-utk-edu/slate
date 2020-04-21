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

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "work/work.hh"

namespace slate {
namespace work {

//------------------------------------------------------------------------------
/// Triangular solve matrix (multiple right-hand sides).
///
/// @tparam target
///         One of HostTask, HostNest, HostBatch, Devices.
///
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
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result X.
///
/// @param[in] row
///         A raw pointer to a dummy vector data. The dummy vector is used for
///         OpenMP dependencies tracking, not based on the actual data. Entries
///         in the dummy vector represent each row of matrix $B$. The size of
///         row should be number of block columns of matrix $A$.
///
/// @param[in] lookahead
///         Number of blocks to overlap communication and computation.
///         lookahead >= 0. Default 1.
///
/// @ingroup trsm_work
///
template <Target target, typename scalar_t>
void trsm(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                               Matrix<scalar_t> B,
          uint8_t* row, int64_t lookahead)
{
    using blas::conj;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

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

    const int64_t mt = B.mt();
    const int64_t nt = B.nt();

    const int priority_one  = 1;
    const int priority_zero = 0;

    const int64_t batch_arrays_index_zero = 0;
    const int64_t batch_arrays_index_one  = 1;

    if (A.uplo() == Uplo::Lower) {
        // ----------------------------------------
        // Lower/NoTrans or Upper/Trans, Left case
        // Forward sweep
        for (int64_t k = 0; k < mt; ++k) {
            scalar_t alph = k == 0 ? alpha : scalar_t(1.0);

            // panel (Akk tile)
            #pragma omp task depend(inout:row[k]) priority(1)
            {
                // send A(k, k) to ranks owning block row B(k, :)
                A.template tileBcast(k, k, B.sub(k, k, 0, nt-1), layout);

                // solve A(k, k) B(k, :) = alpha B(k, :)
                internal::trsm<target>(
                    Side::Left,
                    alph, A.sub(k, k),
                          B.sub(k, k, 0, nt-1),
                    priority_one, layout, batch_arrays_index_one);

                // send A(i=k+1:mt-1, k) to ranks owning block row B(i, :)
                BcastList bcast_list_A;
                for (int64_t i = k+1; i < mt; ++i)
                    bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // send B(k, j=0:nt-1) to ranks owning
                // block col B(k+1:mt-1, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < nt; ++j) {
                    bcast_list_B.push_back(
                        {k, j, {B.sub(k+1, mt-1, j, j)}});
                }
                B.template listBcast<target>(bcast_list_B, layout);
            }

            // lookahead update, B(k+1:k+la, :) -= A(k+1:k+la, k) B(k, :)
            for (int64_t i = k+1; i < k+1+lookahead && i < mt; ++i) {
                #pragma omp task depend(in:row[k]) \
                                 depend(inout:row[i]) priority(1)
                {
                    // TODO: execute lookahead on devices
                    internal::gemm<Target::HostTask>(
                        scalar_t(-1.0), A.sub(i, i, k, k),
                                        B.sub(k, k, 0, nt-1),
                        alph,           B.sub(i, i, 0, nt-1),
                        layout, priority_one);
                }
            }

            // trailing update,
            // B(k+1+la:mt-1, :) -= A(k+1+la:mt-1, k) B(k, :)
            // Updates rows k+1+la to mt-1, but two depends are sufficient:
            // depend on k+1+la is all that is needed in next iteration;
            // depend on mt-1 daisy chains all the trailing updates.
            if (k+1+lookahead < mt) {
                #pragma omp task depend(in:row[k]) \
                                 depend(inout:row[k+1+lookahead]) \
                                 depend(inout:row[mt-1])
                {
                    internal::gemm<target>(
                        scalar_t(-1.0),
                                    A.sub(k+1+lookahead, mt-1, k, k),
                                    B.sub(k, k, 0, nt-1),
                        alph,       B.sub(k+1+lookahead, mt-1, 0, nt-1),
                        layout, priority_zero, batch_arrays_index_zero);
                }
            }
        }
    }
    else {
        // ----------------------------------------
        // Upper/NoTrans or Lower/Trans, Left case
        // Backward sweep
        for (int64_t k = mt-1; k >= 0; --k) {
            scalar_t alph = k == (mt-1) ? alpha : scalar_t(1.0);

            // panel (Akk tile)
            #pragma omp task depend(inout:row[k]) priority(1)
            {
                // send A(k, k) to ranks owning block row B(k, :)
                A.template tileBcast(k, k, B.sub(k, k, 0, nt-1), layout);

                // solve A(k, k) B(k, :) = alpha B(k, :)
                internal::trsm<target>(
                    Side::Left,
                    alph, A.sub(k, k),
                          B.sub(k, k, 0, nt-1),
                    priority_one, layout, batch_arrays_index_one);

                // send A(i=0:k-1, k) to ranks owning block row B(i, :)
                BcastList bcast_list_A;
                for (int64_t i = 0; i < k; ++i)
                    bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // send B(k, j=0:nt-1) to ranks owning block col B(0:k-1, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < nt; ++j)
                    bcast_list_B.push_back({k, j, {B.sub(0, k-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }

            // lookahead update, B(k-la:k-1, :) -= A(k-la:k-1, k) B(k, :)
            for (int64_t i = k-1; i > k-1-lookahead && i >= 0; --i) {
                #pragma omp task depend(in:row[k]) \
                                 depend(inout:row[i]) priority(1)
                {
                    // TODO: execute lookahead on devices
                    internal::gemm<Target::HostTask>(
                        scalar_t(-1.0), A.sub(i, i, k, k),
                                        B.sub(k, k, 0, nt-1),
                        alph,           B.sub(i, i, 0, nt-1),
                        layout, priority_one);
                }
            }

            // trailing update, B(0:k-1-la, :) -= A(0:k-1-la, k) B(k, :)
            // Updates rows 0 to k-1-la, but two depends are sufficient:
            // depend on k-1-la is all that is needed in next iteration;
            // depend on 0 daisy chains all the trailing updates.
            if (k-1-lookahead >= 0) {
                #pragma omp task depend(in:row[k]) \
                                 depend(inout:row[k-1-lookahead]) \
                                 depend(inout:row[0])
                {
                    internal::gemm<target>(
                        scalar_t(-1.0),
                                      A.sub(0, k-1-lookahead, k, k),
                                      B.sub(k, k, 0, nt-1),
                        alph,         B.sub(0, k-1-lookahead, 0, nt-1),
                        layout, priority_zero, batch_arrays_index_zero);
                }
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trsm<Target::HostTask, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, int64_t lookahead);

// ----------------------------------------
template
void trsm<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, int64_t lookahead);

// ----------------------------------------
template
void trsm<Target::HostTask, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostNest, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostBatch, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::Devices, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

// ----------------------------------------
template
void trsm<Target::HostTask, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostNest, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::HostBatch, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

template
void trsm<Target::Devices, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

} // namespace work
} // namespace slate
