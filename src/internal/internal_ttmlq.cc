// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "internal/Tile_tpmlqt.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Distributed multiply matrix by Q from LQ triangle-triangle factorization of
/// row of tiles.
/// Dispatches to target implementations.
/// todo: This assumes A and T have already been communicated as needed.
/// However, it necesarily handles communication for C.
/// Tag is used in gelqf to differentiate communication for look-ahead panel
/// from rest of trailing matrix.
/// @ingroup gelqf_internal
///
template <Target target, typename scalar_t>
void ttmlq(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           int tag )
{
    ttmlq(internal::TargetType<target>(),
          side, op, A, T, C, tag );
}

//------------------------------------------------------------------------------
/// Distributed multiply matrix by Q from LQ triangle-triangle factorization of
/// row of tiles, host implementation.
/// @ingroup gelqf_internal
///
template <typename scalar_t>
void ttmlq(internal::TargetType<Target::HostTask>,
           Side side, Op op,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& T,
           Matrix<scalar_t>& C,
           int tag )
{
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t A_nt = A.nt();
    assert(A.mt() == 1);
    if (side == Side::Left)
        assert(A_nt == C.mt());
    else
        assert(A_nt == C.nt());

    // Find ranks in this row of A.
    std::set<int> ranks_set;
    A.getRanks(&ranks_set);

    // Find each rank's first (left-most) col in this row of A,
    // which is the triangular tile resulting from local gelqf panel.
    std::vector< std::pair<int, int64_t> > rank_indices;
    rank_indices.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t j = 0; j < A_nt; ++j) {
            if (A.tileRank(0, j) == r) {
                rank_indices.push_back({r, j});
                break;
            }
        }
    }
    // Sort rank_indices by index.
    std::sort(rank_indices.begin(), rank_indices.end(),
              compareSecond<int, int64_t>);

    int nranks = rank_indices.size();
    int nlevels = int( ceil( log2( nranks ) ) );

    // Apply reduction tree.
    // If Left, Trans or Right, NoTrans, apply descending from root to leaves,
    // i.e., in reverse order of how they were created.
    // If Left, NoTrans or Right, Trans, apply ascending from leaves to root,
    // i.e., in same order as they were created.
    // Example for A.mt == 8.
    // Leaves:
    //     ttqrt( a0, a1 )
    //     ttqrt( a2, a3 )
    //     ttqrt( a4, a5 )
    //     ttqrt( a6, a7 )
    // Next level:
    //     ttqrt( a0, a2 )
    //     ttqrt( a4, a6 )
    // Root:
    //     ttqrt( a0, a4 )
    bool descend = (side == Side::Left) != (op == Op::NoTrans);
    int step;
    if (descend)
        step = pow(2, nlevels - 1);
    else
        step = 1;

    int64_t k_end;
    int64_t i, j, i1, j1, i_dst, j_dst;

    if (side == Side::Left) {
        k_end = C.nt();
    }
    else {
        k_end = C.mt();
    }

    std::vector<MPI_Request> requests;
    for (int level = 0; level < nlevels; ++level) {
        for (int index = 0; index < nranks; index += step) {
            int64_t rank_ind = rank_indices[ index ].second;
            requests.clear();

            if (index % (2*step) == 0) {
                if (index + step >= nranks) {
                    break;
                }
                int64_t k_dst = rank_indices[ index + step ].second;

                size_t message_count = 0;
                // if (side == left), scan rows of C for local tiles;
                // if (side == right), scan cols of C for local tiles
                // Three for-loops: 1) send, receive 2) update 3) receive, send
                for (int64_t k = 0; k < k_end; ++k) {
                    if (side == Side::Left) {
                        i = rank_ind;
                        j = k;
                    }
                    else {
                        i = k;
                        j = rank_ind;
                    }
                    if (C.tileIsLocal(i, j)) {
                        // Send tile to dst.
                        if (side == Side::Left) {
                            i_dst = k_dst;
                            j_dst = k;
                        }
                        else {
                            i_dst = k;
                            j_dst = k_dst;
                        }
                        int dst = C.tileRank(i_dst, j_dst);
                        // Don't need to wait since the tile isn't modified
                        // until receiving it back
                        MPI_Request req;
                        C.tileIsend( i, j, dst, tag+k, &req );
                        MPI_Request_free( &req );
                        message_count++;
                    }
                }


                if (message_count == 0) {
                    continue;
                }

                // Avoid incrementally reallocating
                requests.reserve(message_count);

                for (int64_t k = 0; k < k_end; ++k) {
                    if (side == Side::Left) {
                        i = rank_ind;
                        j = k;
                    }
                    else {
                        i = k;
                        j = rank_ind;
                    }
                    if (C.tileIsLocal(i, j)) {
                        // Receive updated tile back.
                        if (side == Side::Left) {
                            i_dst = k_dst;
                            j_dst = k;
                        }
                        else {
                            i_dst = k;
                            j_dst = k_dst;
                        }
                        int dst = C.tileRank(i_dst, j_dst);
                        MPI_Request req;
                        C.tileIrecv( i, j, dst, layout, tag+k, &req );
                        requests.push_back( req );
                    }
                }
                slate_mpi_call(
                    MPI_Waitall( requests.size(), requests.data(), MPI_STATUSES_IGNORE ) );
            }
            else {
                int64_t k_src = rank_indices[ index - step ].second;
                size_t message_count = 0;
                // if (side == left), scan rows of C for local tiles;
                // if (side == right), scan cols of C for local tiles
                // Three for-loops: 1) send, receive 2) update 3) receive, send
                for (int64_t k = 0; k < k_end; ++k) {
                    if (side == Side::Left) {
                        i = rank_ind;
                        j = k;
                    }
                    else {
                        i = k;
                        j = rank_ind;
                    }
                    if (C.tileIsLocal(i, j)) {
                        // Receive tile from src.
                        if (side == Side::Left) {
                            i1 = k_src;
                            j1 = k;
                        }
                        else {
                            i1 = k;
                            j1 = k_src;
                        }

                        int src = C.tileRank( i1, j1 );
                        MPI_Request req;
                        C.tileIrecv( i1, j1, src, layout, tag+k, &req );
                        requests.push_back( req );
                        message_count++;
                    }
                }

                if (message_count == 0) {
                    continue;
                }

                // The above and below loops iterate in the same order, so incrementing
                // this counter gives the right request object
                int64_t recv_index = 0;
                for (int64_t k = 0; k < k_end; ++k) {
                    if (side == Side::Left) {
                        i = rank_ind;
                        j = k;
                    }
                    else {
                        i = k;
                        j = rank_ind;
                    }
                    if (C.tileIsLocal(i, j)) {
                        if (side == Side::Left) {
                            i1 = k_src;
                            j1 = k;
                        }
                        else {
                            i1 = k;
                            j1 = k_src;
                        }

                        #pragma omp task slate_omp_default_none \
                            shared( A, T, C, requests ) \
                            firstprivate( i, j, k, rank_ind, layout, i1, j1 ) \
                            firstprivate( side, op, tag, recv_index )
                        {
                            // Don't start compute until the tile's been recieved
                            MPI_Wait( &requests[ recv_index ], MPI_STATUS_IGNORE );

                            A.tileGetForReading(0, rank_ind, LayoutConvert(layout));
                            T.tileGetForReading(0, rank_ind, LayoutConvert(layout));
                            C.tileGetForWriting(i, j, LayoutConvert(layout));

                            // Apply Q.
                            tpmlqt(side, op, std::min(A.tileMb(0), A.tileNb(rank_ind)),
                                   A(0, rank_ind), T(0, rank_ind),
                                   C(i1, j1), C(i, j));

                            int src = C.tileRank( i1, j1 );
                            // Send updated tile back.
                            C.tileIsend( i1, j1, src, tag+k, &requests[ recv_index ] );
                        }
                        recv_index++;
                    }
                }
                #pragma omp taskwait
                slate_mpi_call(
                    MPI_Waitall( requests.size(), requests.data(), MPI_STATUSES_IGNORE ) );
            }
            break;
        }
        if (descend)
            step /= 2;
        else
            step *= 2;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void ttmlq<Target::HostTask, float>(
    Side side, Op op,
    Matrix<float>&& A,
    Matrix<float>&& T,
    Matrix<float>&& C,
    int tag );

// ----------------------------------------
template
void ttmlq<Target::HostTask, double>(
    Side side, Op op,
    Matrix<double>&& A,
    Matrix<double>&& T,
    Matrix<double>&& C,
    int tag );

// ----------------------------------------
template
void ttmlq< Target::HostTask, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& T,
    Matrix< std::complex<float> >&& C,
    int tag );

// ----------------------------------------
template
void ttmlq< Target::HostTask, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& T,
    Matrix< std::complex<double> >&& C,
    int tag );

} // namespace internal
} // namespace slate
