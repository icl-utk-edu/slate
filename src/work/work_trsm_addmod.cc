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
/// AddMod factors solve matrix (multiple right-hand sides).
///
/// @tparam target
///         One of HostTask, HostNest, HostBatch, Devices.
///
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether W appears on the left or on the right of X:
///         - Side::Left:  solve $W X = \alpha B$
///         - Side::Right: solve $X W = \alpha B$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] W
///         - If side = left,  the m-by-m AddMod factor matrix W;
///         - if side = right, the n-by-n AddMod factor matrix W.
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
/// @ingroup trsm_addmod_internal
///
template <Target target, typename scalar_t>
void trsm_addmod(Side side, Uplo uplo,
                 scalar_t alpha, AddModFactors<scalar_t> W,
                                        Matrix<scalar_t> B,
                 uint8_t* row, Options const& opts)
{
    using blas::conj;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    const scalar_t one = 1.0;

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    int64_t ib = W.block_size;
    auto& A = W.A;
    auto& U = W.U_factors;
    auto& VT = W.VT_factors;
    auto& S = W.singular_values;
    auto blockFactorType = W.factorType;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Because of the asymmetric between upper and lower, we can't transpose both sides

    // Check sizes
    if (side == Side::Left) {
        assert(A.mt() == B.mt());
        assert(A.nt() == B.mt());
    }
    else {
        assert(A.mt() == B.nt());
        assert(A.nt() == B.nt());
    }

    int64_t mt = B.mt();
    int64_t nt = B.nt();

    const int priority_one  = 1;
    const int priority_zero = 0;

    Options opts2 = opts;

    // Requires 2+lookahead queues
    if (target == Target::Devices) {
        assert(B.numComputeQueues() >= 2+lookahead);

        // Use only TileReleaseStrategy::Slate for trsm_addmod.
        // Internal routines (trsm_addmod and gemm) called here
        // won't release any tiles. Trsm will
        // clean up tiles.
        opts2[ Option::TileReleaseStrategy ] = TileReleaseStrategy::Slate;
    }

    const int64_t queue_0 = 0;
    const int64_t queue_1 = 1;

    if (side == Side::Left) {
        if (uplo == Uplo::Lower) {
            // ----------------------------------------
            // Lower, Left case
            // Forward sweep
            for (int64_t k = 0; k < mt; ++k) {
                scalar_t alph = k == 0 ? alpha : one;

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // send A(k, k) to ranks owning block row B(k, :)
                    A .template tileBcast(k, k, B.sub(k, k, 0, nt-1), layout);
                    U .template tileBcast(k, k, B.sub(k, k, 0, nt-1), layout);

                    // solve A(k, k) B(k, :) = alpha B(k, :)
                    internal::trsm_addmod<target>(
                        Side::Left, Uplo::Lower,
                        alph, A.sub(k, k, k, k),
                              U.sub(k, k, k, k),
                             VT.sub(k, k, k, k),
                              std::move(S[k]),
                              B.sub(k, k, 0, nt-1),
                        blockFactorType, ib, priority_one, layout, queue_1, opts2);

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
                    A_panel.releaseRemoteWorkspace();
                    A_panel.releaseLocalWorkspace();

                    auto B_panel = B.sub(k, k, 0, nt-1);
                    B_panel.releaseRemoteWorkspace();

                    // Copy back modifications to tiles in the B panel
                    // before they are erased.
                    B_panel.tileUpdateAllOrigin();
                    B_panel.releaseLocalWorkspace();
                }
            }
        }
        else {
            // ----------------------------------------
            // Upper, Left case
            // Backward sweep
            for (int64_t k = mt-1; k >= 0; --k) {
                scalar_t alph = k == (mt-1) ? alpha : one;

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // send A(k, k) to ranks owning block row B(k, :)
                    A .template tileBcast(k, k, B.sub(k, k, 0, nt-1), layout);
                    VT.template tileBcast(k, k, B.sub(k, k, 0, nt-1), layout);

                    // solve A(k, k) B(k, :) = alpha B(k, :)
                    internal::trsm_addmod<target>(
                        Side::Left, Uplo::Upper,
                        alph, A.sub(k, k, k, k),
                              U.sub(k, k, k, k),
                             VT.sub(k, k, k, k),
                              std::move(S[k]),
                              B.sub(k, k, 0, nt-1),
                        blockFactorType, ib, priority_one, layout, queue_1, opts2);

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
                    A_panel.releaseRemoteWorkspace();
                    A_panel.releaseLocalWorkspace();

                    auto B_panel = B.sub(k, k, 0, nt-1);
                    B_panel.releaseRemoteWorkspace();

                    // Copy back modifications to tiles in the B panel
                    // before they are erased.
                    B_panel.tileUpdateAllOrigin();
                    B_panel.releaseLocalWorkspace();
                }
            }
        }
    }
    else {
        if (uplo == Uplo::Lower) {
            // ----------------------------------------
            // Lower, Right case
            // Backward sweep
            for (int64_t k = nt-1; k >= 0; --k) {
                scalar_t alph = k == (nt-1) ? alpha : one;

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // send A(k, k) to ranks owning block column B(:, k)
                    A .template tileBcast(k, k, B.sub(0, mt-1, k, k), layout);
                    U .template tileBcast(k, k, B.sub(0, mt-1, k, k), layout);

                    // solve B(k, :) A(k, k) = alpha B(k, :)
                    internal::trsm_addmod<target>(
                        Side::Right, Uplo::Lower,
                        alph, A.sub(k, k, k, k),
                              U.sub(k, k, k, k),
                             VT.sub(k, k, k, k),
                              std::move(S[k]),
                              B.sub(0, mt-1, k, k),
                        blockFactorType, ib, priority_one, layout, queue_1, opts2);

                    // send A(k, j=0:k-1) to ranks owning block column B(:, j)
                    BcastList bcast_list_A;
                    for (int64_t j = 0; j < k; ++j)
                        bcast_list_A.push_back({k, j, {B.sub(0, mt-1, j, j)}});
                    A.template listBcast<target>(bcast_list_A, layout);

                    // send B(i=0:nt-1, k) to ranks owning block col B(i, 0:k-1)
                    BcastList bcast_list_B;
                    for (int64_t i = 0; i < mt; ++i)
                        bcast_list_B.push_back({i, k, {B.sub(i, i, 0, k-1)}});
                    B.template listBcast<target>(bcast_list_B, layout);
                }

                // lookahead update, B(:, k-la:k-1) -= B(:, k) A(k, k-la:k-1)
                for (int64_t j = k-1; j > k-1-lookahead && j >= 0; --j) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[j]) priority(1)
                    {
                        internal::gemm<target>(
                            -one, B.sub(0, mt-1, k, k),
                                  A.sub(k, k, j, j),
                            alph, B.sub(0, mt-1, j, j),
                            layout, priority_one, j-k+lookahead+2, opts2);
                    }
                }

                // trailing update, B(:, 0:k-1-la) -= B(:, k) A(k, 0:k-1-la)
                // Updates columns 0 to k-1-la, but two depends are sufficient:
                // depend on k-1-la is all that is needed in next iteration;
                // depend on 0 daisy chains all the trailing updates.
                if (k-1-lookahead >= 0) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[k-1-lookahead]) \
                                     depend(inout:row[0])
                    {
                        internal::gemm<target>(
                            -one, B.sub(0, mt-1, k, k),
                                  A.sub(k, k, 0, k-1-lookahead),
                            alph, B.sub(0, mt-1, 0, k-1-lookahead),
                            layout, priority_zero, queue_0, opts2);
                    }
                }

                // Erase remote or workspace tiles.
                #pragma omp task depend(inout:row[k])
                {
                    auto A_panel = A.sub(k, k, 0, k);
                    A_panel.releaseRemoteWorkspace();
                    A_panel.releaseLocalWorkspace();

                    auto B_panel = B.sub(0, mt-1, k, k);
                    B_panel.releaseRemoteWorkspace();

                    // Copy back modifications to tiles in the B panel
                    // before they are erased.
                    B_panel.tileUpdateAllOrigin();
                    B_panel.releaseLocalWorkspace();
                }
            }
        }
        else {
            // ----------------------------------------
            // Upper, Right case
            // Forward sweep
            for (int64_t k = 0; k < nt; ++k) {
                scalar_t alph = k == 0 ? alpha : one;

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // send A(k, k) to ranks owning block column B(:, k)
                    A .template tileBcast(k, k, B.sub(0, mt-1, k, k), layout);
                    VT.template tileBcast(k, k, B.sub(0, mt-1, k, k), layout);

                    // solve B(:, k) A(k, k) = alpha B(:, k)
                    internal::trsm_addmod<target>(
                        Side::Right, Uplo::Upper,
                        alph, A.sub(k, k, k, k),
                              U.sub(k, k, k, k),
                              VT.sub(k, k, k, k),
                              std::move(S[k]),
                              B.sub(0, mt-1, k, k),
                        blockFactorType, ib, priority_one, layout, queue_1, opts2);

                    // send A(k, j=k+1:nt-1) to ranks owning block column B(:, j)
                    BcastList bcast_list_A;
                    for (int64_t j = k+1; j < nt; ++j)
                        bcast_list_A.push_back({k, j, {B.sub(0, mt-1, j, j)}});
                    A.template listBcast<target>(bcast_list_A, layout);

                    // send B(i=0:mt-1, k) to ranks owning
                    // block row B(i, k+1:nt-1)
                    BcastList bcast_list_B;
                    for (int64_t i = 0; i < mt; ++i) {
                        bcast_list_B.push_back(
                            {i, k, {B.sub(i, i, k+1, nt-1)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }

                // lookahead update, B(:, k+1:k+la) -= B(:, k) A(k, k+1:k+la)
                for (int64_t j = k+1; j < k+1+lookahead && j < nt; ++j) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[j]) priority(1)
                    {
                        internal::gemm<target>(
                            -one, B.sub(0, mt-1, k, k),
                                  A.sub(k, k, j, j),
                            alph, B.sub(0, mt-1, j, j),
                            layout, priority_one, j-k+1, opts2);
                    }
                }

                // trailing update,
                // B(:, k+1+la:nt-1) -= B(:, k) A(k, k+1+la:nt-1)
                // Updates rows k+1+la to mt-1, but two depends are sufficient:
                // depend on k+1+la is all that is needed in next iteration;
                // depend on mt-1 daisy chains all the trailing updates.
                if (k+1+lookahead < nt) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[k+1+lookahead]) \
                                     depend(inout:row[nt-1])
                    {
                        internal::gemm<target>(
                            -one, B.sub(0, mt-1, k, k),
                                  A.sub(k, k, k+1+lookahead, nt-1),
                            alph, B.sub(0, mt-1, k+1+lookahead, nt-1),
                            layout, priority_zero, queue_0, opts2);
                    }
                }

                // Erase remote or workspace tiles.
                #pragma omp task depend(inout:row[k])
                {
                    auto A_panel = A.sub(k, k, k, nt-1);
                    A_panel.releaseRemoteWorkspace();
                    A_panel.releaseLocalWorkspace();

                    auto B_panel = B.sub(0, mt-1, k, k);
                    B_panel.releaseRemoteWorkspace();

                    // Copy back modifications to tiles in the B panel
                    // before they are erased.
                    B_panel.tileUpdateAllOrigin();
                    B_panel.releaseLocalWorkspace();
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
void trsm_addmod<Target::HostTask, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> A,
                        Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostNest, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> A,
                        Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostBatch, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> A,
                        Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::Devices, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> A,
                        Matrix<float> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsm_addmod<Target::HostTask, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> A,
                         Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostNest, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> A,
                         Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostBatch, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> A,
                         Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::Devices, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> A,
                         Matrix<double> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsm_addmod<Target::HostTask, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> A,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostNest, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> A,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostBatch, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> A,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::Devices, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> A,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsm_addmod<Target::HostTask, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> A,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostNest, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> A,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::HostBatch, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> A,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsm_addmod<Target::Devices, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> A,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

} // namespace work
} // namespace slate
