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
/// Triangular matrix multiply.
///
/// @tparam target
///         One of HostTask, HostNest, HostBatch, Devices.
///
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether A appears on the left or on the right of B:
///         - Side::Left:  $B = \alpha A B$
///         - Side::Right: $B = \alpha B A$
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
///         On exit, overwritten by the result $\alpha A B$ or $\alpha B A$.
///
/// @param[in] bcast
///         A raw pointer to a dummy vector data.. The dummy vector is used for
///         OpenMP dependencies tracking, not based on the actual data. Entries
///         in the dummy vector represent each column of matrix $A$ and each row
///         of matrix $B$. The size of bcast should be number of block columns of
///         matrix $A$.
///
/// @param[in] gemm
///         A raw pointer to a dummy vector data. The dummy vector is used for
///         OpenMP dependencies tarcking, not based on the actual data. Entries
///         in the dummy vector represent each column of matrix $A$ and each row
///         of matrix $B$. The size of gemm should be number of block columns of
///         matrix $A$.
///
/// @param[in] lookahead
///         Number of blocks to overlap communication and computation.
///         lookahead >= 0. Default 1.
///
/// @ingroup trmm_work
///
template <Target target, typename scalar_t>
void trmm(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                               Matrix<scalar_t> B,
          uint8_t* bcast, uint8_t* gemm, int64_t lookahead)
{
    using blas::conj;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // if on right, change to left by (conj)-transposing A and B to get
    // op(B) = op(A)*op(B)
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

    if (A.uplo() == Uplo::Upper) {
        // ----------------------------------------
        // Left, Upper/NoTrans or Lower/Trans case
        // Forward sweep

        // send 1st block col of A and block row of B
        #pragma omp task depend(out:bcast[0])
        {
            // broadcast A(i, 0) to ranks owning block row B(i, :),
            // for i = 0
            A.template tileBcast<target>(0, 0, B.sub(0, 0, 0, nt-1), layout);

            // broadcast B(0, j) to ranks owning block col B(0:0, j)
            // todo: nowhere to send?
            BcastList bcast_list_B;
            for (int64_t j = 0; j < nt; ++j)
                bcast_list_B.push_back({0, j, {B.sub(0, 0, j, j)}});
            B.template listBcast<target>(bcast_list_B, layout);
        }

        // send next lookahead block cols of A and block rows of B
        for (int64_t k = 1; k < lookahead+1 && k < mt; ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                // broadcast A(i, k) to ranks owning block row B(i, :)
                BcastList bcast_list_A;
                for (int64_t i = 0; i <= k; ++i) // upper
                    bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // broadcast B(k, j) to ranks owning block col B(0:k, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < nt; ++j)
                    bcast_list_B.push_back({k, j, {B.sub(0, k, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }
        }

        // multiply alpha A(:, 0) B(0, :), which is:
        // B(0, :) = alpha [ A(0, 0) B(0, :) ]  trmm
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        {
            internal::trmm<Target::HostTask>(
                Side::Left,
                alpha, A.sub(0, 0),
                       B.sub(0, 0, 0, nt-1));
        }
        for (int64_t k = 1; k < mt; ++k) {

            // send next block col of A and block row of B
            if (k+lookahead < mt) {
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    // broadcast A(i, k+la) to ranks owning
                    // block row B(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = 0; i <= k+lookahead; ++i) {  // upper
                        bcast_list_A.push_back(
                            {i, k+lookahead, {B.sub(i, i, 0, nt-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k+la, j) to ranks owning
                    // block col B(0:k+la, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < nt; ++j) {
                        bcast_list_B.push_back(
                            {k+lookahead, j,
                             {B.sub(0, k+lookahead, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, k) B(k, :), which is:
            // B(0:k-1, :) += alpha [ A(0:k-1, k) B(k, :) ]  gemm
            // B(k, :)      = alpha [ A(k, k)     B(k, :) ]  trmm
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            {
                internal::gemm<target>(
                    alpha,         A.sub(0, k-1, k, k),
                                   B.sub(k, k, 0, nt-1),
                    scalar_t(1.0), B.sub(0, k-1, 0, nt-1),
                    layout);

                // todo: target? needs batch trmm
                internal::trmm<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(k, k),
                           B.sub(k, k, 0, nt-1));
            }
        }
    }
    else {
        // ----------------------------------------
        // Left, Lower/NoTrans or Upper/Trans case
        // Backward sweep

        // send 1st block col of A and block row of B
        #pragma omp task depend(out:bcast[mt-1])
        {
            // broadcast A(i, 0) to ranks owning block row B(i, :),
            // for i = m-1
            A.template tileBcast<target>(
                mt-1, mt-1, B.sub(mt-1, mt-1, 0, nt-1), layout);

            // broadcast B(m-1, j) to ranks owning block col B(m-1:m-1, j)
            // todo: nowhere to send?
            BcastList bcast_list_B;
            for (int64_t j = 0; j < nt; ++j) {
                bcast_list_B.push_back(
                    {mt-1, j, {B.sub(mt-1, mt-1, j, j)}});
            }
            B.template listBcast<target>(bcast_list_B, layout);
        }

        // send next lookahead block cols of A and block rows of B
        for (int64_t k = mt-2; k >= mt-1-lookahead && k >= 0; --k) {
            #pragma omp task depend(in:bcast[k+1]) \
                             depend(out:bcast[k])
            {
                // broadcast A(i, k) to ranks owning block row B(i, :)
                BcastList bcast_list_A;
                for (int64_t i = k; i < mt; ++i)  // lower
                    bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // broadcast B(k, j) to ranks owning block col B(k:m-1, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < nt; ++j)
                    bcast_list_B.push_back({k, j, {B.sub(k, mt-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }
        }

        // multiply B = alpha A(:, mt-1) B(mt-1, :), which is:
        // B(mt-1, :) = alpha [ A(mt-1, mt-1) B(mt-1, :) ]  trmm
        #pragma omp task depend(in:bcast[mt-1]) \
                         depend(out:gemm[mt-1])
        {
            internal::trmm<Target::HostTask>(
                Side::Left,
                alpha, A.sub(mt-1, mt-1),
                       B.sub(mt-1, mt-1, 0, nt-1));
        }

        for (int64_t k = mt-2; k >= 0; --k) {

            // send next block col of A and block row of B
            if (k-lookahead >= 0) {
                #pragma omp task depend(in:gemm[k+1]) \
                                 depend(in:bcast[k-lookahead+1]) \
                                 depend(out:bcast[k-lookahead])
                {
                    // broadcast A(i, k-la) to ranks
                    // owning block row B(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = k-lookahead; i < mt; ++i) {  // lower
                        bcast_list_A.push_back(
                            {i, k-lookahead, {B.sub(i, i, 0, nt-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k-la, j) to ranks
                    // owning block col B(k-la:m-1, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < nt; ++j) {
                        bcast_list_B.push_back(
                            {k-lookahead, j,
                             {B.sub(k-lookahead, mt-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, k) B(k, :), which is:
            // B(k+1:m-1, :) += alpha [ A(k+1:m-1, k) B(k, :) ]  gemm
            // B(k, :)        = alpha [ A(k, k)       B(k, :) ]  trmm
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k+1]) \
                             depend(out:gemm[k])
            {
                internal::gemm<target>(
                    alpha,         A.sub(k+1, mt-1, k, k),
                                   B.sub(k, k, 0, nt-1),
                    scalar_t(1.0), B.sub(k+1, mt-1, 0, nt-1),
                    layout);

                // todo: target? needs batch trmm
                internal::trmm<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(k, k),
                           B.sub(k, k, 0, nt-1));
            }
        }
    } // end Lower/NoTrans

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trmm<Target::HostTask, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

// ----------------------------------------
template
void trmm<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

// ----------------------------------------
template
void trmm<Target::HostTask, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostNest, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostBatch, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::Devices, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

// ----------------------------------------
template
void trmm<Target::HostTask, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostNest, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::HostBatch, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

template
void trmm<Target::Devices, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* bcast, uint8_t* gemm, int64_t lookahead);

} // namespace work
} // namespace slate
