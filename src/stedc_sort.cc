// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal_copy_col.hh"

#include <numeric>

namespace slate {

//------------------------------------------------------------------------------
/// Sorts eigenvalues in D and apply same permutation to eigenvectors in Q.
/// Corresponds to ScaLAPACK pdlasrt.
///
/// @param[in,out] D
///     On entry, unsorted eigenvalues.
///     On exit, eigenvalues in ascending order.
///
/// @param[in] Q
///     Eigenvectors corresponding to D.
///
/// @param[out] Qout
///     On exit, the eigenvectors ordered according to sorted D.
///
/// @param[in] opts
///     Currently unused.
///
template <typename real_t>
void stedc_sort(
    std::vector<real_t>& D,
    Matrix<real_t>& Q,
    Matrix<real_t>& Qout,
    Options const& opts )
{
    // Constants.
    const int tag_0 = 0;

    // Get parameters.
    int64_t n = D.size();
    assert( n == Q.n() );
    int64_t m = Q.m();
    int64_t mb = Q.tileMb( 0 );  // assume fixed
    int64_t nb = Q.tileNb( 0 );  // assume fixed

    // Assumes matrix is 2D block cyclic.
    GridOrder grid_order;
    int nprow, npcol, myrow, mycol;
    Q.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    slate_assert( nprow > 0 );  // require 2D block-cyclic
    slate_assert( grid_order == GridOrder::Col );
    int64_t mlocal = num_local_rows_cols( m, mb, myrow, 0, nprow );

    // Quick return.
    if (mlocal == 0)
        return;

    std::vector<real_t> work( std::max( n, mlocal * nb ) );

    // Determine permutation isort to sort eigenvalues in D.
    std::vector<int64_t> isort( n ), isort_inv( n );
    std::iota( isort.begin(), isort.end(), 0 );
    std::sort( isort.begin(), isort.end(),
               [&D](int64_t const& i_, int64_t const& j_) {
                   return D[i_] < D[j_];
               } );

    // Apply permutation to D and determine inverse permutation.
    blas::copy( n, &D[0], 1, &work[0], 1 );
    for (int64_t j = 0; j < n; ++j) {
        D[ j ] = work[ isort[ j ] ];
        isort_inv[ isort[ j ] ] = j;
    }

    std::vector<int>     pcols( nb );
    std::vector<int64_t> imine( nb ),
                         pcnt( npcol ),
                         poffset( npcol );

    // Apply permutation Qout = P Q.
    // j,  k  is tile index in Q, Qout respectively.
    // jj, kk is offset within tile
    // jg, kg is global index. jg is index of first column in tile j.
    // jb     is tile size
    // pj, pk is process column for j, k resp.
    int pj, pk;
    int64_t jb, k, kk, kg, cnt;
    int64_t jg = 0;
    int64_t nt = Q.nt();
    for (int64_t j = 0; j < nt; ++j) {
        jb = Q.tileNb( j );
        pj = (jg / nb) % npcol;  // indxg2p

        // Get destination process col for each column, and
        // count columns in each destination process col.
        std::fill( &pcnt[ 0 ], &pcnt[ npcol ], 0 );
        imine.clear();
        for (int64_t jj = 0; jj < jb; ++jj) {
            kg = isort_inv[ jg + jj ];
            pk = (kg / nb) % npcol;  // indxg2p
            pcols[ jj ] = pk;
            pcnt[ pk ] += 1;
            if (pk == mycol) {
                imine.push_back( kg );
            }
        }

        if (pj == mycol) {
            // Running sum of column counts.
            poffset[ 0 ] = 0;
            std::partial_sum( &pcnt[ 0 ], &pcnt[ npcol-1 ], &poffset[ 1 ] );
            assert( int64_t( imine.size() ) == pcnt[ mycol ] );

            // Copy columns to workspace, grouped by destination process col (pk).
            // Copy my columns with permutation directly to destination Qout.
            for (int64_t jj = 0; jj < jb; ++jj) {
                pk = pcols[ jj ];
                if (pk == mycol) {
                    kg = isort_inv[ jg + jj ];
                    k  = kg / nb;
                    kk = kg % nb;
                    internal::copy_col( Q, j, jj, Qout, k, kk );
                }
                else {
                    kk = poffset[ pk ]++;
                    internal::copy_col( Q, j, jj, &work[ kk*mlocal ] );
                }
            }

            // Reset running sum of column counts (same as above).
            poffset[ 0 ] = 0;
            std::partial_sum( &pcnt[ 0 ], &pcnt[ npcol-1 ], &poffset[ 1 ] );

            // Send each process's part of workspace to that process.
            // todo: MPI_Isend
            for (int p = 0; p < npcol; ++p) {
                cnt = pcnt[ p ];
                if (p != mycol && cnt > 0 && myrow < Q.mt()) {
                    int dst = Q.tileRank( myrow, p );
                    kk = poffset[ p ];
                    slate_mpi_call(
                        MPI_Send( &work[ kk*mlocal ], mlocal * cnt,
                                  mpi_type<real_t>::value,
                                  dst, tag_0, Q.mpiComm() ) );
                }
            }
        }
        else if (imine.size() > 0 && myrow < Q.mt()) {
            // Recv workspace, then copy with permutation to destination Qout.
            cnt = imine.size();
            assert( cnt == pcnt[ mycol ] );
            int src = Q.tileRank( myrow, pj );
            slate_mpi_call(
                MPI_Recv( &work[ 0 ], mlocal * cnt,
                          mpi_type<real_t>::value,
                          src, tag_0, Q.mpiComm(), MPI_STATUS_IGNORE ) );
            for (int64_t jj = 0; jj < cnt; ++jj) {
                kg = imine[ jj ];
                k  = kg / nb;
                kk = kg % nb;
                internal::copy_col( &work[ jj*mlocal ], Qout, k, kk );
            }
        }

        jg += jb;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Only real, not complex.
template
void stedc_sort<float>(
    std::vector<float>& D,
    Matrix<float>& Q,
    Matrix<float>& Qout,
    Options const& opts );

template
void stedc_sort<double>(
    std::vector<double>& D,
    Matrix<double>& Q,
    Matrix<double>& Qout,
    Options const& opts );

} // namespace slate
