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
/// Triangular solve matrix (multiple right-hand sides).
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
/// @ingroup trsm_internal
///
template <Target target, typename scalar_t>
void trsm(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                               Matrix<scalar_t> B,
          uint8_t* row, Options const& opts)
{
    using blas::conj;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    const scalar_t one = 1.0;

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

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

    int64_t mt = B.mt();
    int64_t nt = B.nt();

    const int priority_one  = 1;
    const int priority_zero = 0;

    Options opts2 = opts;

    // Requires 2+lookahead queues
    if (target == Target::Devices) {
        assert(B.numComputeQueues() >= 2+lookahead);

        // Use only TileReleaseStrategy::Slate for trsm.
        // Internal routines (trsm and gemm) called here
        // won't release any tiles. Trsm will
        // clean up tiles.
        opts2[ Option::TileReleaseStrategy ] = TileReleaseStrategy::Slate;
    }

    const int64_t queue_0 = 0;
    const int64_t queue_1 = 1;

    if (A.uplo() == Uplo::Lower) {
        // ----------------------------------------
        // Lower/NoTrans or Upper/Trans, Left case
        // Forward sweep
        for (int64_t k = 0; k < mt; ++k) {
            scalar_t alph = k == 0 ? alpha : one;

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
                    priority_one, layout, queue_1, opts2);

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
                    internal::gemm<target>(
                        -one, A.sub(i, i, k, k),
                              B.sub(k, k, 0, nt-1),
                        alph, B.sub(i, i, 0, nt-1),
                        layout, priority_one, i-k+1, opts2);
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
                        -one, A.sub(k+1+lookahead, mt-1, k, k),
                              B.sub(k, k, 0, nt-1),
                        alph, B.sub(k+1+lookahead, mt-1, 0, nt-1),
                        layout, priority_zero, queue_0, opts2);
                }
            }

            // Erase remote or workspace tiles.
            #pragma omp task depend(inout:row[k])
            {
                auto A_panel = A.sub(k, mt-1, k, k);
                A_panel.eraseRemoteWorkspace();
                A_panel.eraseLocalWorkspace();

                auto B_panel = B.sub(k, k, 0, nt-1);
                B_panel.eraseRemoteWorkspace();

                // Copy back modifications to tiles in the B panel
                // before they are erased.
                B_panel.tileUpdateAllOrigin();
                B_panel.eraseLocalWorkspace();
            }
        }
    }
    else {
        // ----------------------------------------
        // Upper/NoTrans or Lower/Trans, Left case
        // Backward sweep
        for (int64_t k = mt-1; k >= 0; --k) {
            scalar_t alph = k == (mt-1) ? alpha : one;

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
                    priority_one, layout, queue_1, opts2);

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
                    internal::gemm<target>(
                        -one, A.sub(i, i, k, k),
                              B.sub(k, k, 0, nt-1),
                        alph, B.sub(i, i, 0, nt-1),
                        layout, priority_one, i-k+lookahead+2, opts2);
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
                        -one, A.sub(0, k-1-lookahead, k, k),
                              B.sub(k, k, 0, nt-1),
                        alph, B.sub(0, k-1-lookahead, 0, nt-1),
                        layout, priority_zero, queue_0, opts2);
                }
            }

            // Erase remote or workspace tiles.
            #pragma omp task depend(inout:row[k])
            {
                auto A_panel = A.sub(0, k, k, k);
                A_panel.eraseRemoteWorkspace();
                A_panel.eraseLocalWorkspace();

                auto B_panel = B.sub(k, k, 0, nt-1);
                B_panel.eraseRemoteWorkspace();

                // Copy back modifications to tiles in the B panel
                // before they are erased.
                B_panel.tileUpdateAllOrigin();
                B_panel.eraseLocalWorkspace();
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
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsm<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsm<Target::HostTask, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostNest, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostBatch, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::Devices, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsm<Target::HostTask, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostNest, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::HostBatch, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsm<Target::Devices, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

} // namespace work
} // namespace slate
