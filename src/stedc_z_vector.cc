// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Communicates the z vector to all ranks.
///
/// Corresponds to ScaLAPACK pdlaedz.
//------------------------------------------------------------------------------
/// @tparam real_t
///     One of float, double.
//------------------------------------------------------------------------------
///
/// @param[in] Q
///     On entry, matrix of eigenvectors for 2 sub-problems being merged,
///         Q = [ Q1  0  ].
///             [ 0   Q2 ]
///     If Q is nt-by-nt tiles, then:
///     Q1 is nt1-by-nt1 tiles with nt1 = floor( nt / 2 ),
///     Q2 is nt2-by-nt2 tiles with nt2 =  ceil( nt / 2 ).
///
/// @param[out] z
///     On exit, z vector in divide-and-conquer algorithm, consisting of
///     the last row of Q1 and the first row of Q2:
///         z = Q^T [ e_{n_1} ] = [ Q1^T e_{n_1} ].
///                 [ e_1     ]   [ Q2^T e_1     ]
///     z is duplicated on all MPI ranks.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Currently none
///
/// @ingroup heev_computational
///
template <typename real_t>
void stedc_z_vector(
    Matrix<real_t>& Q,
    std::vector<real_t>& z,
    Options const& opts )
{
    int mpi_rank = Q.mpiRank();

    const int tag = 0;
    const int root = 0;

    assert( Q.mt() == Q.nt() );
    int64_t nt = Q.nt();
    int64_t nt1 = nt/2;
    //int64_t nt2 = ceildiv( nt, 2 );

    // Gather z1 = last row of Q1.
    // todo: We could pack all the data on each rank into one workspace to
    // have only one send, followed by unpack, as in ScaLAPACK.
    int64_t jj = 0;  // position in z vector == # elements received so far.

    // Start gathering z1 = last row of Q1, from block row nt1-1.
    int64_t i  = nt1 - 1;
    int64_t ii = Q.tileMb( i ) - 1;  // Last row of block.
    for (int64_t j = 0; j < nt; ++j) {
        if (j == nt1) {
            // Switch to gathering z2 = 1st row of Q2, from block row nt1.
            i  = nt1;
            ii = 0;  // First row.
        }
        int64_t nb = Q.tileNb( j );
        if (Q.tileIsLocal( i, j )) {
            Q.tileGetForReading( i, j, LayoutConvert::None );
            auto Qij = Q( i, j );
            // Pack it into contiguous space. On root, this is its final place.
            blas::copy( nb, &Qij.at( ii, 0 ), Qij.stride(), &z[ jj ], 1 );
            if (mpi_rank != root) {
                // todo: isend?
                slate_mpi_call(
                    MPI_Send( &z[ jj ], nb, mpi_type<real_t>::value,
                              root, tag, Q.mpiComm() ) );
            }
        }
        else if (mpi_rank == root) {
            // todo: irecv?
            slate_mpi_call(
                MPI_Recv( &z[ jj ], nb, mpi_type<real_t>::value,
                          Q.tileRank( i, j ), tag, Q.mpiComm(),
                          MPI_STATUS_IGNORE ) );
        }
        jj += nb;
    }

    // Broadcast complete z to all ranks.
    MPI_Bcast( &z[ 0 ], jj, mpi_type<real_t>::value, root, Q.mpiComm() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Only real, not complex.
template
void stedc_z_vector<float>(
    Matrix<float>& Q,
    std::vector<float>& z,
    Options const& opts );

template
void stedc_z_vector<double>(
    Matrix<double>& Q,
    std::vector<double>& z,
    Options const& opts );

} // namespace slate
