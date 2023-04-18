// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

#include <numeric>

namespace slate {

//------------------------------------------------------------------------------
/// Finds the nsecular roots of the secular equation, as defined by the values
/// in rho, D, z, and computes the corresponding eigenvectors in U.
///
/// Corresponds to ScaLAPACK pdlaed3.
//------------------------------------------------------------------------------
/// @tparam real_t
///     One of float, double.
//------------------------------------------------------------------------------
/// @param[in] nsecular
///     The number of non-deflated eigenvalues, and the order of the
///     related secular equation. 0 <= nsecular <= n.
///
/// @param[in] n
///     The number of rows and columns in the U matrix.
///     n >= nsecular (deflation may result in n > nsecular).
///
/// @param[in] rho
///     The off-diagonal element associated with the rank-1 cut that
///     originally split the two submatrices to be merged,
///     as modified by stedc_deflate to be positive and make $z$ unit norm.
///
/// @param[in] D
///     The nsecular non-deflated eigenvalues of the two sub-problems,
///     as output by stedc_deflate in Dsecular, which will be used by
///     laed4 to form the secular equation.
///
/// @param[in] z
///     The nsecular entries of the deflation-adjusted z vector,
///     as output by stedc_deflate in zsecular.
///
/// @param[in,out] Lambda
///     On entry, Lambda contains the (n - nsecular) deflated eigenvalues,
///     as output by stedc_deflate in D.
///     On exit, Lambda contains all n eigenvalues, permuted by itype.
///
/// @param[out] U
///     On exit, U contains the orthonormal eigenvectors from the secular
///     equation, permuted by itype.
///
/// @param[in] itype
///     Permutation that arranged Qtype locally into 4 groups by column type,
///         Qtype = [ Q11  Q12       Q14 ],
///                 [      Q22  Q23  Q24 ]
///     as output by stedc_deflate.
///     Used to permute Lambda and U to match Qtype.
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
void stedc_secular(
    int64_t nsecular, int64_t n,
    real_t rho,
    real_t* D,
    real_t* z,
    real_t* Lambda,
    Matrix<real_t>& U,
    int64_t* itype,
    Options const& opts )
{
    const MPI_Datatype mpi_real_t = mpi_type<real_t>::value;

    int mpi_rank = U.mpiRank();
    int mpi_size;
    slate_mpi_call(
        MPI_Comm_size( U.mpiComm(), &mpi_size ) );

    int64_t info = 0, iinfo;

    // Assumes matrix is 2D block cyclic.
    GridOrder grid_order;
    int nprow, npcol, myrow, mycol;
    U.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    slate_assert( nprow > 0 );  // require 2D block-cyclic
    slate_assert( grid_order == GridOrder::Col );

    //----------
    // Phase 1: compute Lambda and ztilde.

    // This process computes mycnt eigenvalues in range [ begin, end ).
    // Distribute evenly, with first rem processes having 1 extra.
    int64_t min_cnt = nsecular / mpi_size;
    int64_t rem     = nsecular % mpi_size;
    int64_t mycnt   = min_cnt + (mpi_rank < rem ? 1 : 0);
    int64_t begin   = mpi_rank * min_cnt + blas::min( mpi_rank, rem );
    int64_t end     = begin + mycnt;

    // ztilde_partial = 1.
    std::vector<real_t> ztilde( nsecular, 1.0 ),
                        deltaJ( nsecular ),
                        Lambda_local( nsecular );
    for (int64_t j = begin; j < end; ++j) {
        iinfo = lapack::laed4( nsecular, j, &D[ 0 ], &z[ 0 ], &deltaJ[ 0 ],
                               rho, &Lambda_local[ j ] );
        if (iinfo != 0)
            info = j;

        // Update local partial product ztilde_partial
        // ztilde_partial *= deltaJ / (d_i - d_j)
        for (int64_t i = 0; i < j; ++i) {
            ztilde[ i ] *= deltaJ[ i ] / (D[ i ] - D[ j ]);
        }
        // for i = j, exclude (d_i - d_j) term in denominator.
        ztilde[ j ] *= deltaJ[ j ];
        for (int64_t i = j+1; i < nsecular; ++i) {
            ztilde[ i ] *= deltaJ[ i ] / (D[ i ] - D[ j ]);
        }
    }

    // ztilde = +- sqrt( prod_{all ranks} ztilde_partial )
    slate_mpi_call(
        MPI_Allreduce( MPI_IN_PLACE, &ztilde[ 0 ], nsecular, mpi_real_t,
                       MPI_PROD, U.mpiComm() ) );
    // Compute final ztilde, with sign from original z (redundantly).
    for (int64_t i = 0; i < nsecular; ++i) {
        ztilde[ i ] = copysign( sqrt( -ztilde[ i ] ), z[ i ] );
    }

    // recv_cnts = [ min_cnt+1, .., min_cnt+1, min_cnt, .., min_cnt ]
    // recv_offsets[ j ] = sum_{i=0, .., j-1} recv_cnts[ i ]
    std::vector<int> recv_cnts( mpi_size ),
                     recv_offsets( mpi_size+1 );
    std::fill( &recv_cnts[ 0 ], &recv_cnts[ rem ], min_cnt + 1 );
    std::fill( &recv_cnts[ rem ], &recv_cnts[ mpi_size ], min_cnt );
    std::partial_sum( &recv_cnts[ 0 ], &recv_cnts[ mpi_size ],
                      &recv_offsets[ 1 ] );
    slate_mpi_call(
        MPI_Allgatherv( MPI_IN_PLACE, mycnt, mpi_real_t,
                        &Lambda_local[ 0 ], &recv_cnts[ 0 ], &recv_offsets[ 0 ],
                        mpi_real_t, U.mpiComm() ) );

    if (info != 0)
        slate_error( "info " + std::to_string( info ) );

    //----------
    // Phase 2: using computed Lambda and ztilde, compute U.

    // todo: compute pcols once & pass to stedc_deflate, stedc_secular?
    // Set prows( j ) = process row of D(j),
    // and pcols( j ) = process col of D(j).
    // pcols == prows if nprow == npcol (square grid) and drow == dcol.
    int64_t nb = U.tileNb( 0 );
    std::vector<int> prows( n );
    std::vector<int> pcols( n );

    {
        // j is row, col index; jj is block-row, block-col index.
        int r0 = U.tileRank( 0, 0 );
        int prow = r0 % nprow;  // todo: assumes col-major grid
        int pcol = r0 / nprow;  // todo: assumes col-major grid
        int64_t j = 0;
        int64_t nt = U.nt();
        for (int64_t jj = 0; jj < nt; ++jj) {
            int64_t jb = U.tileNb( jj );  // assumes square tiles
            std::fill( &prows[ j ], &prows[ j + jb ], prow );
            std::fill( &pcols[ j ], &pcols[ j + jb ], pcol );
            j += jb;
            prow = (prow + 1) % nprow;
            pcol = (pcol + 1) % npcol;
        }
    }

    // Permute Lambda by itype.
    // Build icol and irow to find local cols and rows of U.
    std::vector<int64_t> icol, irow;
    for (int64_t j = 0; j < nsecular; ++j) {
        int64_t jq = itype[ j ];
        assert( 0 <= jq && jq < n );
        Lambda[ jq ] = Lambda_local[ j ];
        if (pcols[ jq ] == mycol) {
            icol.push_back( j );
        }
        if (prows[ jq ] == myrow) {
            irow.push_back( j );
        }
    }
    int64_t col_cnt = icol.size();
    int64_t row_cnt = irow.size();

    // Compute u vectors.
    // Each rank in processor column computes redundantly in order to get norm.
    // todo: cache delta_jj terms and compute other deltas from D[i] - Lambda[j],
    // rather than redundantly calling laed4.
    for (int64_t jj = 0; jj < col_cnt; ++jj) {
        int64_t j  = icol[ jj ];
        int64_t jq = itype[ j ];
        int64_t jq_tile   = jq / nb;
        int64_t jq_offset = jq % nb;

        assert( 0 <= j  && j  < n );
        assert( 0 <= jq && jq < n );
        assert( 0 <= jq_tile   && jq_tile < U.nt() );
        assert( 0 <= jq_offset && jq_offset < nb );

        real_t dummy;
        iinfo = lapack::laed4( nsecular, j, &D[ 0 ], &z[ 0 ], &deltaJ[ 0 ],
                               rho, &dummy );

        real_t nrm;
        if (nsecular <= 2) {
            nrm = 1.0;
        }
        else {
            for (int64_t i = 0; i < nsecular; ++i) {
                deltaJ[ i ] = ztilde[ i ] / deltaJ[ i ];
            }
            nrm = blas::nrm2( nsecular, &deltaJ[ 0 ], 1 );
        }
        for (int64_t ii = 0; ii < row_cnt; ++ii) {
            int64_t i  = irow[ ii ];
            int64_t iq = itype[ i ];
            int64_t iq_tile   = iq / nb;
            int64_t iq_offset = iq % nb;

            assert( 0 <= i  && i  < n );
            assert( 0 <= iq && iq < n );
            assert( 0 <= iq_tile   && iq_tile < U.mt() );
            assert( 0 <= iq_offset && iq_offset < nb );

            auto Uij = U( iq_tile, jq_tile );
            Uij.at( iq_offset, jq_offset ) = deltaJ[ i ] / nrm;
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Only real, not complex.
template
void stedc_secular<float>(
    int64_t nsecular, int64_t n,
    float rho,
    float* D,
    float* z,
    float* Lambda,
    Matrix<float>& U,
    int64_t* itype,
    Options const& opts );

template
void stedc_secular<double>(
    int64_t nsecular, int64_t n,
    double rho,
    double* D,
    double* z,
    double* Lambda,
    Matrix<double>& U,
    int64_t* itype,
    Options const& opts );

} // namespace slate
