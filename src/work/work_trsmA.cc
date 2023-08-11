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
void trsmA(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                                Matrix<scalar_t> B,
           uint8_t* row, Options const& opts)
{
    using blas::conj;
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using std::real;
    using std::imag;
    using std::swap;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int queue_0 = 0;
    const int queue_1 = 1;

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    auto tileStrategy = get_option<TileReleaseStrategy>( opts, Option::TileReleaseStrategy, TileReleaseStrategy::Slate );

    Options local_opts = opts;
    local_opts[ Option::Lookahead ] = lookahead;

    // XXX This should be removed later, based on Kadir's comment.
    local_opts[ Option::TileReleaseStrategy ] = tileStrategy;

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

    // Requires at least 2 queues
    if (target == Target::Devices)
        assert(A.numComputeQueues() >= 2);

    // Scale the RHS to handle the alpha issue since B is moved
    // around instead of the A as in trsm
    // TODO Call scale( alpha, one, B, local_opts ) when
    // transpose will be handled.
    if (alpha != one) {
        if (target == Target::Devices) {
            for (int64_t i = 0; i < mt; ++i) {
                for (int64_t j = 0; j < nt; ++j) {
                    scalar_t alpha_ = alpha;
                    if (B.tileIsLocal( i, j )) {
                        int device = B.tileDevice( i, j );

                        B.tileGetForWriting( i, j, device,
                                LayoutConvert( layout ) );

                        blas::Queue* queue = A.compute_queue( device, queue_0 );
                        assert( queue != nullptr );
                        auto T = B( i, j, device );
                        int64_t T_mb = T.mb();
                        int64_t T_nb = T.nb();
                        if (T.op() != Op::NoTrans) {
                            swap( T_mb, T_nb );
                            if (T.op() == Op::ConjTrans)
                                alpha_ = conj( alpha );
                        }

                        device::gescale( T_mb, T_nb, alpha_, one,
                                T.data(), T.stride(), *queue );
                        queue->sync();
                    }
                }
            }
        }
        else {
            for (int64_t i = 0; i < mt; ++i) {
                for (int64_t j = 0; j < nt; ++j) {
                    if (B.tileIsLocal( i, j )) {
                        B.tileGetForWriting( i, j, LayoutConvert( layout ) );
                        tile::scale( alpha, B(i, j) );
                    }
                }
            }
        }
    }

    if (A.uplo() == Uplo::Lower) {
        // ----------------------------------------
        // Lower/NoTrans or Upper/Trans, Left case
        // Forward sweep
        for (int64_t k = 0; k < mt; ++k) {
            // panel (Akk tile)
            #pragma omp task depend(inout:row[k]) priority(1)
            {
                // Create the local B tiles where A(k,k) is located
                if (A.tileIsLocal(k, k)) {
                    // XXX insert only what is needed for this iteration, otherwise,
                    // all missing tiles are inserted.
                    for (int64_t j = 0; j < nt; ++j) {
                        if (! B.tileIsLocal(k, j) ) {
                            if (target == Target::Devices) {
                                int device = A.tileDevice( k, k );
                                if (! B.tileExists( k, j, device )) {
                                    B.tileInsertWorkspace( k, j, device, layout );
                                    B.tileModified( k, j, device );

                                    // XXX maybe reset the memory in case it got created from a previous call.
                                    blas::Queue* queue = A.compute_queue( device, queue_0 );
                                    assert( queue != nullptr );
                                    auto T = B( k, j, device );
                                    int64_t T_mb = T.mb();
                                    int64_t T_nb = T.nb();
                                    if (T.op() != Op::NoTrans) {
                                        swap( T_mb, T_nb );
                                    }

                                    device::geset( T_mb, T_nb, zero, zero,
                                            T.data(), T.stride(), *queue );
                                    queue->sync();
                                }
                            }
                            else {
                                if (! B.tileExists( k, j )) {
                                    B.tileInsertWorkspace( k, j, HostNum, layout );
                                    B.tileModified( k, j, HostNum );

                                    B( k, j ).set( 0, 0 );
                                }
                            }
                        }
                    }
                    // XXX We can sync here instead of at each iteration.
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
                B.template listReduce<target>(reduce_list_B, layout, k);

                if (A.tileIsLocal(k, k)) {
                    // solve A(k, k) B(k, :) = alpha B(k, :)
                    internal::trsmA<target>(
                        Side::Left,
                        one, A.sub(k, k),
                             B.sub(k, k, 0, nt-1),
                        priority_1, layout, queue_1, local_opts );
                }

                // Send the solution back to where it belongs
                // TODO : could be part of the bcast of the solution,
                // but not working now because of listBcast constraint.
                if (A.tileIsLocal(k, k)) {
                    const int root = A.tileRank(k, k);
                    for (int64_t j = 0; j < nt; ++j) {
                        int dest = B.tileRank(k, j);
                        if (dest == root) continue;

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

                // Bcast the result of the solve, B(k,:) to
                // ranks owning block row A(k + 1 : mt, k)
                // TODO does it work for the last iteration?
                BcastList bcast_list_upd_B;
                for (int64_t j = 0; j < nt; ++j) {
                    // FIXME add B( k, j ) as dest
                    bcast_list_upd_B.push_back(
                        {k, j, { A.sub(k + 1, mt - 1, k, k), }});
                }

                auto B_row_k = B.sub( k, k, 0, nt-1 );
                // XXX Should it be just 1 at the last iteration?
                // XXX Should we ignore this since the life will be removed
                B.template listBcast<target>( bcast_list_upd_B, layout, k, lookahead + 1 );

            }

            // lookahead update, B(k+1:k+la, :) -= A(k+1:k+la, k) B(k, :)
            for (int64_t i = k+1; i < k+1+lookahead && i < mt; ++i) {
                #pragma omp task depend(in:row[k]) \
                                 depend(inout:row[i]) priority(1)
                {

                    int queue_ik1 = i - k + 1;
                    for (int j = 0; j < nt; ++j) {
                        internal::gemmA<target>(
                            -one, A.sub(i, i, k, k),
                                  B.sub(k, k, j, j),
                            one,  B.sub(i, i, j, j),
                            layout, priority_1, queue_ik1, local_opts );
                    }
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
                    for (int64_t j = 0; j < nt; ++j) {
                        internal::gemmA<target>(
                            -one, A.sub(k+1+lookahead, mt-1, k, k),
                                  B.sub(k, k, j, j),
                            one,  B.sub(k+1+lookahead, mt-1, j, j),
                            layout, priority_0, queue_0, local_opts );
                    }
                }
            }

            // Erase remote or workspace tiles.
            #pragma omp task depend(inout:row[k])
            {
                auto A_col_k = A.sub( k, mt-1, k, k );
                A_col_k.releaseRemoteWorkspace();
                A_col_k.releaseLocalWorkspace();

                auto B_row_k = B.sub( k, k, 0, nt-1 );

                B_row_k.releaseRemoteWorkspace();

                // Copy back modifications to tiles in the B panel
                // before they are erased.
                B_row_k.tileUpdateAllOrigin();
                B_row_k.releaseLocalWorkspace();
            }
        }
    }
    else {
        // ----------------------------------------
        // Upper/NoTrans or Lower/Trans, Left case
        // Backward sweep
        for (int64_t k = mt-1; k >= 0; --k) {

            // panel (Akk tile)
            #pragma omp task depend(inout:row[k]) priority(1)
            {
                // Create the local B tiles where A(k,k) is located
                if (A.tileIsLocal(k, k)) {
                    for (int64_t j = 0; j < nt; ++j) {
                        if (! B.tileIsLocal(k, j) ) {
                            if (target == Target::Devices) {
                                int device = A.tileDevice( k, k );
                                if (! B.tileExists( k, j, device )) {
                                    B.tileInsertWorkspace( k, j, device, layout );
                                    B.tileModified( k, j, device );

                                    blas::Queue* queue = A.compute_queue( device, queue_0 );
                                    assert( queue != nullptr );
                                    auto T = B( k, j, device );
                                    int64_t T_mb = T.mb();
                                    int64_t T_nb = T.nb();
                                    if (T.op() != Op::NoTrans) {
                                        swap( T_mb, T_nb );
                                    }

                                    device::geset( T_mb, T_nb, zero, zero,
                                            T.data(), T.stride(), *queue );
                                    queue->sync();
                                }
                            }
                            else {
                                if (! B.tileExists( k, j )) {
                                    B.tileInsertWorkspace( k, j, HostNum, layout );
                                    B.tileModified( k, j, HostNum );

                                    B( k, j ).set( 0, 0 );
                                }
                            }
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
                B.template listReduce<target>(reduce_list_B, layout, k);

                if (A.tileIsLocal(k, k)) {
                    // solve A(k, k) B(k, :) = alpha B(k, :)
                    internal::trsmA<target>(
                        Side::Left,
                        one, A.sub(k, k),
                             B.sub(k, k, 0, nt-1),
                        priority_1, layout, queue_1, local_opts );
                }

                // Send the solution back to where it belongs
                if (A.tileIsLocal(k, k)) {
                    const int root = A.tileRank(k, k);
                    for (int64_t j = 0; j < nt; ++j) {
                        int dest = B.tileRank(k, j);
                        if (dest == root) continue;

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

                // Bcast the result of the solve, B(k,:) to
                // ranks owning block row A(k + 1 : mt, k)
                BcastList bcast_list_upd_B;
                for (int64_t j = 0; j < nt; ++j) {
                    bcast_list_upd_B.push_back(
                        {k, j, { A.sub(0, k - 1, k, k), }});
                }
                B.template listBcast<target>(bcast_list_upd_B, layout, k, lookahead + 1 );
            }

            // lookahead update, B(k-la:k-1, :) -= A(k-la:k-1, k) B(k, :)
            for (int64_t i = k-1; i > k-1-lookahead && i >= 0; --i) {
                #pragma omp task depend(in:row[k]) \
                                 depend(inout:row[i]) priority(1)
                {
                    int queue_k1lai = k - 1 + lookahead - i;
                    for (int j = 0; j < nt; ++j) {
                        internal::gemmA<target>(
                            -one, A.sub(i, i, k, k),
                                  B.sub(k, k, j, j),
                            one,  B.sub(i, i, j, j),
                            layout, priority_1, queue_k1lai, local_opts );
                    }
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
                    for (int64_t j = 0; j < nt; ++j) {
                        internal::gemmA<target>(
                            -one, A.sub(0, k-1-lookahead, k, k),
                                  B.sub(k, k, j, j),
                            one,  B.sub(0, k-1-lookahead, j, j),
                            layout, priority_0, queue_0, local_opts );
                    }
                }
            }

            // Erase remote or workspace tiles.
            #pragma omp task depend(inout:row[k])
            {
                auto A_col_k = A.sub( 0, k, k, k );
                A_col_k.releaseRemoteWorkspace();
                A_col_k.releaseLocalWorkspace();

                auto B_row_k = B.sub( k, k, 0, nt-1 );
                B_row_k.releaseRemoteWorkspace();
                // Copy back modifications to tiles in the B panel
                // before they are erased.
                B_row_k.tileUpdateAllOrigin();
                B_row_k.releaseLocalWorkspace();
            }
        }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void trsmA<Target::HostTask, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostNest, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostBatch, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::Devices, float>(
    Side side,
    float alpha, TriangularMatrix<float> A,
                           Matrix<float> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsmA<Target::HostTask, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostNest, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostBatch, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::Devices, double>(
    Side side,
    double alpha, TriangularMatrix<double> A,
                            Matrix<double> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsmA<Target::HostTask, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostNest, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostBatch, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::Devices, std::complex<float>>(
    Side side,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>> A,
                                         Matrix<std::complex<float>> B,
    uint8_t* row, Options const& opts);

// ----------------------------------------
template
void trsmA<Target::HostTask, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostNest, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::HostBatch, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

template
void trsmA<Target::Devices, std::complex<double>>(
    Side side,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>> A,
                                          Matrix<std::complex<double>> B,
    uint8_t* row, Options const& opts);

} // namespace work
} // namespace slate
