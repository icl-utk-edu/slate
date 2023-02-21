// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "work/work.hh"

namespace slate {
namespace work {

// TODO update the description of the function
//------------------------------------------------------------------------------
/// AddMod factor solve matrix (multiple right-hand sides).
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
///         - If side = left,  the m-by-m triangular matrix W;
///         - if side = right, the n-by-n triangular matrix W.
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
void trsmA_addmod(Side side, Uplo uplo,
                  scalar_t alpha, AddModFactors<scalar_t> W,
                                         Matrix<scalar_t> B,
                  uint8_t* row, int64_t lookahead)
{
    using blas::conj;
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using std::real;
    using std::imag;

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
        assert(W.A.mt() == B.mt());
        assert(W.A.nt() == B.mt());
    }
    else {
        assert(W.A.mt() == B.nt());
        assert(W.A.nt() == B.nt());
    }

    int64_t mt = B.mt();
    int64_t nt = B.nt();

    const int priority_one  = 1;
    const int priority_zero = 0;

    // Requires 2 queues
    if (target == Target::Devices)
        assert(B.numComputeQueues() >= 2);
    //const int64_t queue_0 = 0;
    const int64_t queue_1 = 1;

    const scalar_t one = 1.0;

    if (side == Side::Left) {
        if (uplo == Uplo::Lower) {
            // ----------------------------------------
            // Lower, Left case
            // Forward sweep
            for (int64_t k = 0; k < mt; ++k) {

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // Scale the RHS in order to be consistent with the upper case
                    if (k == 0 && alpha != one) {
                        for (int64_t i = 0; i < mt; ++i) {
                            for (int64_t j = 0; j < nt; ++j) {
                                if (B.tileIsLocal(i, j)) {
                                    tile::scale( alpha, B(i, j) );
                                }
                            }
                        }
                    }

                    // Create the local B tiles where A(k,k) is located
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t j = 0; j < nt; ++j) {
                            if (! B.tileIsLocal(k, j) && ! B.tileExists(k, j)) {
                                B.tileInsert(k, j);
                                B.at(k, j).set(0, 0);
                            }
                        }
                    }

                    // Gather B(k,:) to rank owning diagonal block A(k,k)
                    using ReduceList = typename Matrix<scalar_t>::ReduceList;
                    ReduceList reduce_list_B;
                    for (int64_t j = 0; j < nt; ++j) {
                        reduce_list_B.push_back({k, j,
                                                  A.sub(k, k, k, k),
                                                  { A.sub(k, k, 0, k),
                                                    B.sub(k, k, j, j )
                                                  }
                                                });
                    }
                    B.template listReduce<target>(reduce_list_B, layout);

                    if (A.tileIsLocal(k, k)) {
                        // solve A(k, k) B(k, :) = alpha B(k, :)
                        internal::trsmA_addmod<target>(
                            Side::Left, Uplo::Lower,
                            one, A.sub(k, k, k, k),
                                 U.sub(k, k, k, k),
                                VT.sub(k, k, k, k),
                                 std::move(S[k]),
                                 B.sub(k, k, 0, nt-1),
                            blockFactorType,
                            ib, priority_one, layout, queue_1);
                    }

                    // Send the solution back to where it belongs
                    // TODO : could be part of the bcast of the solution,
                    // but not working now
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t j = 0; j < nt; ++j) {
                            int dest = B.tileRank(k, j);
                            B.tileSend(k, j, dest);
                        }
                    }
                    else {
                        const int root = A.tileRank(k, k);

                        for (int64_t j = 0; j < nt; ++j) {
                            if (B.tileIsLocal(k, j)) {
                                B.tileRecv(k, j, root, layout);
                            }
                        }
                    }

                    for (int64_t j = 0; j < nt; ++j)
                        if (B.tileExists(k, j) && ! B.tileIsLocal(k, j))
                            B.tileErase(k, j);

                    // Bcast the result of the solve, B(k,:) to
                    // ranks owning block row A(k + 1 : mt, k)
                    BcastList bcast_list_upd_B;
                    for (int64_t j = 0; j < nt; ++j) {
                        bcast_list_upd_B.push_back(
                            {k, j, { A.sub(k + 1, mt - 1, k, k), }});
                    }
                    B.template listBcast<target>(bcast_list_upd_B, layout);
                }

                // lookahead update, B(k+1:k+la, :) -= A(k+1:k+la, k) B(k, :)
                for (int64_t i = k+1; i < k+1+lookahead && i < mt; ++i) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[i]) priority(1)
                    {
                        if (A.tileIsLocal(i, k)) {
                            for (int64_t j = 0; j < nt; ++j) {
                                if (! B.tileIsLocal(i, j)
                                    && ! B.tileExists(i, j))
                                {
                                    B.tileInsert(i, j);
                                    B.at(i, j).set(0, 0);
                                }
                            }
                        }
                        // TODO: execute lookahead on devices
                        internal::gemmA<Target::HostTask>(
                            -one, A.sub(i, i, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(i, i, 0, nt-1),
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
                        for (int64_t i = k+1+lookahead; i < mt; ++i) {
                            if (A.tileIsLocal(i, k)) {
                                for (int64_t j = 0; j < nt; ++j) {
                                    if (! B.tileIsLocal(i, j)
                                        && ! B.tileExists(i, j))
                                    {
                                        B.tileInsert(i, j);
                                        B.at(i, j).set(0, 0);
                                    }
                                }
                            }
                        }

                        //internal::gemmA<target>(
                        internal::gemmA<Target::HostTask>(
                            -one, A.sub(k+1+lookahead, mt-1, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(k+1+lookahead, mt-1, 0, nt-1),
                            layout, priority_zero); //, queue_0);
                    }
                }
            }
        }
        else {
            // ----------------------------------------
            // Upper, Left case
            // Backward sweep
            for (int64_t k = mt-1; k >= 0; --k) {

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // Scale the RHS to handle the alpha issue since B is moved
                    // around instead of the A as in trsm
                    if (k == mt - 1 && alpha != one) {
                        for (int64_t i = 0; i < mt; ++i) {
                            for (int64_t j = 0; j < nt; ++j) {
                                if (B.tileIsLocal(i, j)) {
                                    tile::scale( alpha, B(i, j) );
                                }
                            }
                        }
                    }

                    // Create the local B tiles where A(k,k) is located
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t j = 0; j < nt; ++j) {
                            if (! B.tileIsLocal(k, j) && ! B.tileExists(k, j)) {
                                B.tileInsert(k, j);
                                B.at(k, j).set(0, 0); // Might not needed if alph is set correctly
                            }
                        }
                    }

                    // Gather B(k,:) to rank owning diagonal block A(k,k)
                    using ReduceList = typename Matrix<scalar_t>::ReduceList;
                    ReduceList reduce_list_B;
                    for (int64_t j = 0; j < nt; ++j) {
                        reduce_list_B.push_back({k, j,
                                                  A.sub(k, k, k, k),
                                                  { A.sub(k, k, k, mt - 1),
                                                    B.sub(k, k, j, j )
                                                  }
                                                });
                    }
                    B.template listReduce<target>(reduce_list_B, layout);

                    if (A.tileIsLocal(k, k)) {
                        // solve A(k, k) B(k, :) = alpha B(k, :)
                        internal::trsmA_addmod<target>(
                            Side::Left, Uplo::Upper,
                            one, A.sub(k, k, k, k),
                                 U.sub(k, k, k, k),
                                VT.sub(k, k, k, k),
                                 std::move(S[k]),
                                 B.sub(k, k, 0, nt-1),
                            blockFactorType,
                            ib, priority_one, layout, queue_1);
                    }

                    // Send the solution back to where it belongs
                    // TODO : could be part of the bcast of the solution,
                    // but not working now
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t j = 0; j < nt; ++j) {
                            int dest = B.tileRank(k, j);
                            B.tileSend(k, j, dest);
                        }
                    }
                    else {
                        const int root = A.tileRank(k, k);

                        for (int64_t j = 0; j < nt; ++j) {
                            if (B.tileIsLocal(k, j)) {
                                B.tileRecv(k, j, root, layout);
                            }
                        }
                    }

                    for (int64_t j = 0; j < nt; ++j)
                        if (B.tileExists(k, j) && ! B.tileIsLocal(k, j))
                            B.tileErase(k, j);

                    // Bcast the result of the solve, B(k,:) to
                    // ranks owning block row A(k + 1 : mt, k)
                    BcastList bcast_list_upd_B;
                    for (int64_t j = 0; j < nt; ++j) {
                        bcast_list_upd_B.push_back(
                            {k, j, { A.sub(0, k - 1, k, k), }});
                    }
                    B.template listBcast<target>(bcast_list_upd_B, layout);
                }

                // lookahead update, B(k-la:k-1, :) -= A(k-la:k-1, k) B(k, :)
                for (int64_t i = k-1; i > k-1-lookahead && i >= 0; --i) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[i]) priority(1)
                    {
                        if (A.tileIsLocal(i, k)) {
                            for (int64_t j = 0; j < nt; ++j) {
                                if (! B.tileIsLocal(i, j)
                                    && ! B.tileExists(i, j))
                                {
                                    B.tileInsert(i, j);
                                    B.at(i, j).set(0, 0);
                                }
                            }
                        }
                        // TODO: execute lookahead on devices
                        internal::gemmA<Target::HostTask>(
                            -one, A.sub(i, i, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(i, i, 0, nt-1),
                            layout, priority_one);
                    }
                }

                // trailing update,
                // B(0:k-1-la, :) -= A(0:k-1-la, k) B(k, :)
                // Updates rows 0 to k-1-la, but two depends are sufficient:
                // depend on k-1-la is all that is needed in next iteration;
                // depend on 0 daisy chains all the trailing updates.
                if (k-1-lookahead >= 0) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[k-1-lookahead]) \
                                     depend(inout:row[0])
                    {
                        for (int64_t i = 0; i < k - lookahead; ++i) {
                            if (A.tileIsLocal(i, k)) {
                                for (int64_t j = 0; j < nt; ++j) {
                                    if (! B.tileIsLocal(i, j)
                                        && ! B.tileExists(i, j))
                                    {
                                        B.tileInsert(i, j);
                                        B.at(i, j).set(0, 0);
                                    }
                                }
                            }
                        }

                        //internal::gemm<target>(
                        internal::gemmA<Target::HostTask>(
                            -one, A.sub(0, k-1-lookahead, k, k),
                                  B.sub(k, k, 0, nt-1),
                            one,  B.sub(0, k-1-lookahead, 0, nt-1),
                            layout, priority_zero); //, queue_0);
                    }
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

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // Scale the RHS to handle the alpha issue since B is moved
                    // around instead of the A as in trsm
                    if (k == nt - 1 && alpha != one) {
                        for (int64_t i = 0; i < mt; ++i) {
                            for (int64_t j = 0; j < nt; ++j) {
                                if (B.tileIsLocal(i, j)) {
                                    tile::scale( alpha, B(i, j) );
                                }
                            }
                        }
                    }

                    // Create the local B tiles where A(k,k) is located
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t i = 0; i < mt; ++i) {
                            if (! B.tileIsLocal(i, k) && ! B.tileExists(i, k)) {
                                B.tileInsert(i, k);
                                B.at(i, k).set(0, 0); // Might not needed if alph is set correctly
                            }
                        }
                    }

                    // Gather B(:,k) to rank owning diagonal block A(k,k)
                    using ReduceList = typename Matrix<scalar_t>::ReduceList;
                    ReduceList reduce_list_B;
                    for (int64_t i = 0; i < mt; ++i) {
                        reduce_list_B.push_back({i, k,
                                                  A.sub(k, k, k, k),
                                                  { A.sub(k, nt - 1, k, k),
                                                    B.sub(i, i, k, k)
                                                  }
                                                });
                    }
                    B.template listReduce<target>(reduce_list_B, layout);

                    if (A.tileIsLocal(k, k)) {
                        // solve A(k, k) B(k, :) = alpha B(k, :)
                        internal::trsmA_addmod<target>(
                            Side::Right, Uplo::Lower,
                            one, A.sub(k, k, k, k),
                                 U.sub(k, k, k, k),
                                VT.sub(k, k, k, k),
                                 std::move(S[k]),
                                 B.sub(0, mt-1, k, k),
                            blockFactorType,
                            ib, priority_one, layout, queue_1);
                    }

                    // Send the solution back to where it belongs
                    // TODO : could be part of the bcast of the solution,
                    // but not working now
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t i = 0; i < mt; ++i) {
                            int dest = B.tileRank(i, k);
                            B.tileSend(i, k, dest);
                        }
                    }
                    else {
                        const int root = A.tileRank(k, k);

                        for (int64_t i = 0; i < mt; ++i) {
                            if (B.tileIsLocal(i, k)) {
                                B.tileRecv(i, k, root, layout);
                            }
                        }
                    }

                    for (int64_t i = 0; i < mt; ++i)
                        if (B.tileExists(i, k) && ! B.tileIsLocal(i, k))
                            B.tileErase(i, k);

                    // Bcast the result of the solve, B(:,k) to
                    // ranks owning block column A(k, k + 1 : nt)
                    BcastList bcast_list_upd_B;
                    for (int64_t i = 0; i < mt; ++i) {
                        bcast_list_upd_B.push_back(
                            {i, k, { A.sub(k, k, 0, k - 1), }});
                    }
                    B.template listBcast<target>(bcast_list_upd_B, layout);
                }

                // lookahead update, B(:, k-la:k-1) -= B(:, k) A(k, k-la:k-1)
                for (int64_t j = k-1; j > k-1-lookahead && j >= 0; --j) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[j]) priority(1)
                    {
                        if (A.tileIsLocal(k, j)) {
                            for (int64_t i = 0; i < mt; ++i) {
                                if (! B.tileIsLocal(i, j)
                                    && ! B.tileExists(i, j))
                                {
                                    B.tileInsert(i, j);
                                    B.at(i, j).set(0, 0);
                                }
                            }
                        }
                        // TODO: execute lookahead on devices
                        //internal::gemmB<Target::HostTask>(
                        //    -one, B.sub(0, mt-1, k, k),
                        //          A.sub(k, k, j, j),
                        //    one,  B.sub(0, mt-1, j, j),
                        //    layout, priority_one);
                        internal::gemmA<Target::HostTask>(
                            -one, conjTranspose(A.sub(k, k, j, j)),
                                  conjTranspose(B.sub(0, mt-1, k, k)),
                            one,  conjTranspose(B.sub(0, mt-1, j, j)),
                            layout, priority_one);
                    }
                }

                // trailing update,
                // B(:, 0:k-1-la) -= B(:, k) A(k, 0:k-1-la)
                // Updates columns 0 to k-1-la, but two depends are sufficient:
                // depend on k-1-la is all that is needed in next iteration;
                // depend on 0 daisy chains all the trailing updates.
                if (k-1-lookahead >= 0) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[k-1-lookahead]) \
                                     depend(inout:row[0])
                    {
                        for (int64_t j = 0; j < k - lookahead; ++j) {
                            if (A.tileIsLocal(k, j)) {
                                for (int64_t i = 0; i < mt; ++i) {
                                    if (! B.tileIsLocal(i, j)
                                        && ! B.tileExists(i, j))
                                    {
                                        B.tileInsert(i, j);
                                        B.at(i, j).set(0, 0);
                                    }
                                }
                            }
                        }

                        //internal::gemmB<Target::HostTask>(
                        //    -one, B.sub(0, mt-1, k, k),
                        //          A.sub(k, k, 0, k-1-lookahead),
                        //    one,  B.sub(0, mt-1, 0, k-1-lookahead),
                        //    layout, priority_zero); //, queue_0);
                        internal::gemmA<Target::HostTask>(
                            -one, conjTranspose(A.sub(k, k, 0, k-1-lookahead)),
                                  conjTranspose(B.sub(0, mt-1, k, k)),
                            one,  conjTranspose(B.sub(0, mt-1, 0, k-1-lookahead)),
                            layout, priority_zero); //, queue_0);
                    }
                }
            }
        }
        else {
            // ----------------------------------------
            // Upper, Right case
            // Forward sweep
            for (int64_t k = 0; k < nt; ++k) {

                // panel (Akk tile)
                #pragma omp task depend(inout:row[k]) priority(1)
                {
                    // Scale the RHS in order to be consistent with the upper case
                    if (k == 0 && alpha != one) {
                        for (int64_t i = 0; i < mt; ++i) {
                            for (int64_t j = 0; j < nt; ++j) {
                                if (B.tileIsLocal(i, j)) {
                                    tile::scale( alpha, B(i, j) );
                                }
                            }
                        }
                    }

                    // Create the local B tiles where A(k,k) is located
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t i = 0; i < mt; ++i) {
                            if (! B.tileIsLocal(i, k) && ! B.tileExists(i, k)) {
                                B.tileInsert(i, k);
                                B.at(i, k).set(0, 0);
                            }
                        }
                    }

                    // Gather B(:,k) to rank owning diagonal block A(k,k)
                    using ReduceList = typename Matrix<scalar_t>::ReduceList;
                    ReduceList reduce_list_B;
                    for (int64_t i = 0; i < mt; ++i) {
                        reduce_list_B.push_back({i, k,
                                                  A.sub(k, k, k, k),
                                                  { A.sub(0, k, k, k),
                                                    B.sub(i, i, k, k )
                                                  }
                                                });
                    }
                    B.template listReduce<target>(reduce_list_B, layout);

                    if (A.tileIsLocal(k, k)) {
                        // solve B(:, k) A(k, k) = alpha B(:, k)
                        internal::trsmA_addmod<target>(
                            Side::Right, Uplo::Upper,
                            one, A.sub(k, k, k, k),
                                 U.sub(k, k, k, k),
                                VT.sub(k, k, k, k),
                                 std::move(S[k]),
                                 B.sub(0, mt-1, k, k),
                            blockFactorType,
                            ib, priority_one, layout, queue_1);
                    }

                    // Send the solution back to where it belongs
                    // TODO : could be part of the bcast of the solution,
                    // but not working now
                    if (A.tileIsLocal(k, k)) {
                        for (int64_t i = 0; i < mt; ++i) {
                            int dest = B.tileRank(i, k);
                            B.tileSend(i, k, dest);
                        }
                    }
                    else {
                        const int root = A.tileRank(k, k);

                        for (int64_t i = 0; i < mt; ++i) {
                            if (B.tileIsLocal(i, k)) {
                                B.tileRecv(i, k, root, layout);
                            }
                        }
                    }

                    for (int64_t i = 0; i < mt; ++i)
                        if (B.tileExists(i, k) && ! B.tileIsLocal(i, k))
                            B.tileErase(i, k);

                    // Bcast the result of the solve, B(:,k) to
                    // ranks owning block column A(k, k + 1 : nt)
                    BcastList bcast_list_upd_B;
                    for (int64_t i = 0; i < mt; ++i) {
                        bcast_list_upd_B.push_back(
                            {i, k, { A.sub(k, k, k + 1, nt - 1), }});
                    }
                    B.template listBcast<target>(bcast_list_upd_B, layout);
                }

                // lookahead update, B(:, k+1:k+la) -= B(:, k) A(k, k+1:k+la)
                for (int64_t j = k+1; j < k+1+lookahead && j < nt; ++j) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[j]) priority(1)
                    {
                        if (A.tileIsLocal(k, j)) {
                            for (int64_t i = 0; i < mt; ++i) {
                                if (! B.tileIsLocal(i, j)
                                    && ! B.tileExists(i, j))
                                {
                                    B.tileInsert(i, j);
                                    B.at(i, j).set(0, 0);
                                }
                            }
                        }
                        // TODO: execute lookahead on devices
                        //internal::gemmB<Target::HostTask>(
                        //    -one, B.sub(0, mt-1, k, k),
                        //          A.sub(k, k, j, j),
                        //    one,  B.sub(0, mt-1, j, j),
                        //    layout, priority_one);
                        internal::gemmA<Target::HostTask>(
                            -one, conjTranspose(A.sub(k, k, j, j)),
                                  conjTranspose(B.sub(0, mt-1, k, k)),
                            one,  conjTranspose(B.sub(0, mt-1, j, j)),
                            layout, priority_one);
                    }
                }

                // trailing update,
                // B(:, k+1+la:nt-1) -= B(:, k) A(k, k+1+la:nt-1)
                // Updates columns k+1+la to nt-1, but two depends are sufficient:
                // depend on k+1+la is all that is needed in next iteration;
                // depend on nt-1 daisy chains all the trailing updates.
                if (k+1+lookahead < nt) {
                    #pragma omp task depend(in:row[k]) \
                                     depend(inout:row[k+1+lookahead]) \
                                     depend(inout:row[nt-1])
                    {
                        for (int64_t j = k+1+lookahead; j < nt; ++j) {
                            if (A.tileIsLocal(k, j)) {
                                for (int64_t i = 0; i < mt; ++i) {
                                    if (! B.tileIsLocal(i, j)
                                        && ! B.tileExists(i, j))
                                    {
                                        B.tileInsert(i, j);
                                        B.at(i, j).set(0, 0);
                                    }
                                }
                            }
                        }

                        //internal::gemmB<Target::HostTask>(
                        //    -one, B.sub(0, mt-1, k, k),
                        //          A.sub(k, k, k+1+lookahead, nt-1),
                        //    one,  B.sub(0, mt-1, k+1+lookahead, nt-1),
                        //    layout, priority_zero); //, queue_0);
                        internal::gemmA<Target::HostTask>(
                            -one, conjTranspose(A.sub(k, k, k+1+lookahead, nt-1)),
                                  conjTranspose(B.sub(0, mt-1, k, k)),
                            one,  conjTranspose(B.sub(0, mt-1, k+1+lookahead, nt-1)),
                            layout, priority_zero); //, queue_0);
                    }
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
void trsmA_addmod<Target::HostTask, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> A,
                        Matrix<float> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostNest, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> W,
                        Matrix<float> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostBatch, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> W,
                        Matrix<float> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::Devices, float>(
    Side side, Uplo uplo,
    float alpha, AddModFactors<float> W,
                        Matrix<float> B,
    uint8_t* row, int64_t lookahead);

// ----------------------------------------
template
void trsmA_addmod<Target::HostTask, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> W,
                         Matrix<double> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostNest, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> W,
                         Matrix<double> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostBatch, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> W,
                         Matrix<double> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::Devices, double>(
    Side side, Uplo uplo,
    double alpha, AddModFactors<double> W,
                         Matrix<double> B,
    uint8_t* row, int64_t lookahead);

// ----------------------------------------
template
void trsmA_addmod<Target::HostTask, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> W,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostNest, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> W,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostBatch, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> W,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::Devices, std::complex<float>>(
    Side side, Uplo uplo,
    std::complex<float> alpha, AddModFactors<std::complex<float>> W,
                                      Matrix<std::complex<float>> B,
    uint8_t* row, int64_t lookahead);

// ----------------------------------------
template
void trsmA_addmod<Target::HostTask, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> W,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostNest, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> W,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::HostBatch, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> W,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

template
void trsmA_addmod<Target::Devices, std::complex<double>>(
    Side side, Uplo uplo,
    std::complex<double> alpha, AddModFactors<std::complex<double>> W,
                                       Matrix<std::complex<double>> B,
    uint8_t* row, int64_t lookahead);

} // namespace work
} // namespace slate
