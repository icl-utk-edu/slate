// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "work/work.hh"

namespace slate {
namespace work {

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Note A and B are passed by value, so we can transpose if needed
/// (for side = right) without affecting caller.
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
///         OpenMP dependencies tracking, not based on the actual data. Entries
///         in the dummy vector represent each column of matrix $A$ and each row
///         of matrix $B$. The size of gemm should be number of block columns of
///         matrix $A$.
///
/// @param[in] lookahead
///         Number of blocks to overlap communication and computation.
///         lookahead >= 0. Default 1.
///
/// @ingroup trmm_internal
///
template <Target target, typename scalar_t>
void trmm(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                               Matrix<scalar_t> B,
          uint8_t* bcast, uint8_t* gemm, int64_t lookahead)
{
    using blas::conj;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    const scalar_t one = 1.0;

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

    const int priority_0 = 0;
    const int priority_1 = 1;

    // Requires at least 2 queues
    if (target == Target::Devices)
        assert(B.numComputeQueues() >= 2);
    const int64_t queue_0 = 0;
    const int64_t queue_1 = 1;

    if (A.uplo() == Uplo::Upper) {
        // ----------------------------------------
        // Left, Upper/NoTrans or Lower/Trans case
        // Forward sweep

        // send 1st block col of A and block row of B
        #pragma omp task depend(out:bcast[0]) priority(1)
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
                             depend(out:bcast[k]) priority(1)
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
                         depend(out:gemm[0]) priority(1)
        {
            internal::trmm<target>(
                Side::Left,
                alpha, A.sub(0, 0),
                       B.sub(0, 0, 0, nt-1), priority_1, queue_1);
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
                    alpha, A.sub(0, k-1, k, k),
                           B.sub(k, k, 0, nt-1),
                    one,   B.sub(0, k-1, 0, nt-1),
                    layout, priority_0, queue_0);

                internal::trmm<target>(
                    Side::Left,
                    alpha, A.sub(k, k),
                           B.sub(k, k, 0, nt-1),
                    priority_0, queue_1);
            }
        }
    }
    else {
        // ----------------------------------------
        // Left, Lower/NoTrans or Upper/Trans case
        // Backward sweep

        // send 1st block col of A and block row of B
        #pragma omp task depend(out:bcast[mt-1]) priority(1)
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
                             depend(out:bcast[k]) priority(1)
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
                         depend(out:gemm[mt-1]) priority(1)
        {
            internal::trmm<target>(
                Side::Left,
                alpha, A.sub(mt-1, mt-1),
                       B.sub(mt-1, mt-1, 0, nt-1), priority_1, queue_1);
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
                    alpha, A.sub(k+1, mt-1, k, k),
                           B.sub(k, k, 0, nt-1),
                    one,   B.sub(k+1, mt-1, 0, nt-1),
                    layout, priority_0, queue_0);

                // todo: target? needs batch trmm
                internal::trmm<target>(
                    Side::Left,
                    alpha, A.sub(k, k),
                           B.sub(k, k, 0, nt-1),
                    priority_0, queue_1);
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
