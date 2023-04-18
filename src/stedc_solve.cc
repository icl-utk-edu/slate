// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
/// @return Next higher power of 2 >= x, i.e., 2^ceil( log2( x ) ).
/// See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float
/// @ingroup util
///
template <typename T>
T next_power2( T x )
{
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    if constexpr (sizeof(x) > 1) {  // 16-bit
        x |= x >> 8;
    }
    if constexpr (sizeof(x) > 2) {  // 32-bit
        x |= x >> 16;
    }
    if constexpr (sizeof(x) > 4) {  // 64-bit
        x |= x >> 32;
    }
    x += 1;
    return x;
}

}  // namespace internal

//------------------------------------------------------------------------------
/// Computes all eigenvalues and eigenvectors of a symmetric tridiagonal
/// matrix in parallel, using the divide and conquer algorithm.
///
/// Corresponds to ScaLAPACK pdlaed0.
//------------------------------------------------------------------------------
/// @tparam real_t
///     One of float, double.
//------------------------------------------------------------------------------
/// @param[in,out] D
///     On entry, the diagonal elements of the tridiagonal matrix.
///     On exit, the eigenvalues, not sorted.
///
/// @param[in,out] E
///     On entry, the subdiagonal elements of the tridiagonal matrix.
///     On exit, E has been destroyed.
///
/// @param[in,out] Q
///     On entry, Q is the Identity.
///     On exit, Q contains the orthonormal eigenvectors of the
///     symmetric tridiagonal matrix.
///
/// @param[out] W
///     W is a workspace, the same size as Q.
///
/// @param[out] U
///     U is a workspace, the same size as Q.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup heev_computational
///
template <typename real_t>
void stedc_solve(
    std::vector<real_t>& D, std::vector<real_t>& E,
    Matrix<real_t>& Q,
    Matrix<real_t>& W,
    Matrix<real_t>& U,
    Options const& opts )
{
    int mpi_rank = Q.mpiRank();

    using internal::next_power2;

    const int tag = 0;
    const int root = 0;

    int64_t n = D.size();
    int64_t nb = Q.tileNb( 0 );
    // todo: assumes uniform nb. Should be fixable to non-uniform blocks.

    // Divide the matrix into nsubpbs submatrices of size at most nb
    // using rank-1 modifications (cuts).
    for (int64_t i = nb; i < n; i += nb) {
        real_t rho = std::abs( E[ i-1 ] );
        D[ i-1 ] -= rho;
        D[ i   ] -= rho;
    }

    // Solve each submatrix eigenproblem at the bottom of the divide and
    // conquer tree. D is the same on each process.
    // ii is row index, i is block-row index.
    int64_t ii = 0;
    int64_t ib;
    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t i = 0; i < Q.nt(); ++i) {
            ib = Q.tileNb( i );
            assert( ib == std::min( nb, n - ii ) );
            if (Q.tileIsLocal( i, i )) {
                #pragma omp task
                {
                    auto Qii = Q( i, i );
                    assert( Qii.mb() == ib );
                    assert( Qii.nb() == ib );
                    #if 0  // todo: get from opts.
                        lapack::steqr( lapack::Job::Vec, ib, &D[ ii ], &E[ ii ],
                                       Qii.data(), Qii.stride() );
                    #else
                        lapack::stedc( lapack::Job::Vec, ib, &D[ ii ], &E[ ii ],
                                       Qii.data(), Qii.stride() );
                    #endif
                }
            }
            ii += ib;
        }
    }

    // Gather all pieces of D to node 0, then broadcast to everyone.
    // todo: worthwhile to use Isend, Irecv? Profile.
    // Like MPI_Allgatherv, but each node has multiple, non-contiguous blocks
    // (block cyclic).
    ii = 0;
    for (int64_t i = 0; i < Q.nt(); ++i) {
        int src = Q.tileRank( i, i );
        ib = Q.tileNb( i );
        if (src != root) {
            if (mpi_rank == src) {
                slate_mpi_call(
                    MPI_Send( &D[ ii ], ib, mpi_type<real_t>::value,
                              root, tag, Q.mpiComm() ) );
            }
            else if (mpi_rank == root) {
                slate_mpi_call(
                    MPI_Recv( &D[ ii ], ib, mpi_type<real_t>::value,
                              src, tag, Q.mpiComm(), MPI_STATUS_IGNORE ) );
            }
        }
        ii += ib;
    }
    slate_mpi_call(
        MPI_Bcast( &D[ 0 ], n, mpi_type<real_t>::value, root, Q.mpiComm() ) );

    // Determine the size and placement of the submatrices, and save in subs.
    // For i = 0 .. end-1, subs[ i ] is what block subproblem i starts at,
    // from 0 .. nt-1, where end = 2^ceil( log2( nsubpbs ) ),
    // i.e., round nsubpbs up to power of 2.
    int64_t nsubpbs = ceildiv( n, nb );  // (tsubpbs)
    std::vector<int64_t> subs;
    subs.reserve( next_power2( nsubpbs ) );

    // Initially, set subs to size of each subproblem for i = 0 .. subpbs-1.
    // While last and largest subproblem is > 1, add another level to tree
    // and split problems in half, with the larger half second.
    // Example: nsubpbs = 13. (a b) are a pair that divided.
    // end =  1; subs = [ 13 ]
    // end =  2; subs = [ (6 7) ]
    // end =  4; subs = [ (3 3) (3 4) ]
    // end =  8; subs = [ (1 2) (1 2) (1 2) (2 2) ]
    // end = 16; subs = [ (0 1) (1 1) (0 1) (1 1) (0 1) (1 1) (1  1) (1  1) ]
    // running sum:
    //           subs = [ 0  1  2  3  3  4  5  6  6  7  8  9  10  11  12  13  ]
    int64_t end = 1;  // C++ convention: index past-the-end. (subpbs)
    subs.resize( end );
    subs.at( end-1 ) = nsubpbs;
    while (subs.at( end-1 ) > 1) {
        subs.resize( 2*end );
        for (int64_t i = end-1; i >= 0; --i) {
            int64_t nmerge = subs.at( i );
            subs.at( 2*i + 1 ) = (nmerge + 1) / 2;  //  ceil( nmerge/2 )
            subs.at( 2*i     ) =  nmerge / 2;       // floor( nmerge/2 )
        }
        end *= 2;
    }

    // Running sum to find starting block.
    // If subs[ i ] == subs[ i-1 ], there is no subproblem i.
    // If subs[ 0 ] == 0, there is no subproblem 0.
    for (int64_t i = 1; i < end; ++i) {
        subs.at( i ) += subs.at( i-1 );
    }

    // Successively merge eigensystems of adjacent submatrices
    // into eigensystem for the corresponding larger matrix.
    // nblock  is # blocks in merged problem
    // nblock1 is # blocks in 1st subproblem; if 0, nothing to do.
    // nmerge  is # rows   in merged problem (matsiz)
    // nmerge1 is # rows   in 1st subproblem (n1)
    // Example: nsubpbs = 13.
    // (a, b) are a pair to merge;
    // {a, b} are a pair to merge where nblock1 = 0, so there's nothing to do.
    // end = 16; subs = [ {0 1} (2 3) {3 4} (5 6) {6 7} (8 9) (10 11) (12 13) ]
    // end =  8; subs = [ (  1     3) (  4     6) (  7     9) (   11      13) ]
    // end =  4; subs = [ (        3           6) (        9              13) ]
    // end =  2; subs = [ (                    6                          13) ]
    // end =  1; done
    int64_t nblock, nblock1, nmerge1, nmerge, j, j2, jj;
    while (end > 1) {
        for (int64_t i = 0; i <= end-2; i += 2) {
            if (i == 0) {
                nblock  = subs.at( 1 );
                nblock1 = subs.at( 0 );
                j = 0;
                jj = 0;
            }
            else {
                nblock  = subs.at( i+1 ) - subs.at( i-1 );
                nblock1 = nblock / 2;
                j = subs.at( i-1 );
                jj = j * nb;
            }
            nmerge  = std::min( nblock * nb, n - jj );
            nmerge1 = nblock1 * nb;
            if (nblock1 > 0) {
                real_t rho = E[ jj + nmerge1 - 1 ];
                j2 = j + nblock - 1;
                auto Qsub = Q.sub( j, j2, j, j2 );
                auto Wsub = W.sub( j, j2, j, j2 );
                auto Usub = U.sub( j, j2, j, j2 );
                assert( Qsub.n() == nmerge );

                stedc_merge( nmerge, nmerge1, rho, &D[ jj ], Qsub, Wsub, Usub,
                             opts );
            }

            // Shift: subs[ 0, 1, 2, .., (end-2)/2 ] = subs[ 1, 3, 5, .., end-1 ]
            subs.at( i/2 ) = subs.at( i + 1 );
        }
        end /= 2;
        subs.resize( end );
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Only real, not complex.
template
void stedc_solve<float>(
    std::vector<float>& D, std::vector<float>& E,
    Matrix<float>& Q,
    Matrix<float>& W,
    Matrix<float>& U,
    Options const& opts );

template
void stedc_solve<double>(
    std::vector<double>& D, std::vector<double>& E,
    Matrix<double>& Q,
    Matrix<double>& W,
    Matrix<double>& U,
    Options const& opts );

} // namespace slate
