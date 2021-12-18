// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Copy local rows of column from matrix A, tile j, column jj,
/// to matrix B, tile k, column kk.
/// A and B must have the same distribution, number of rows, and tile mb;
/// they may differ in the number of columns.
///
/// todo: duplicated from stedc_sort.cc
///
template <typename real_t>
void copy_col(
    Matrix<real_t>& A, int64_t j, int64_t jj,
    Matrix<real_t>& B, int64_t k, int64_t kk )
{
    assert( A.mt() == B.mt() );

    int64_t mt = A.mt();
    int64_t ii = 0;
    for (int64_t i = 0; i < mt; ++i) {
        if (A.tileIsLocal( i, j )) {
            assert( B.tileIsLocal( i, j ) );
            auto Aij = A( i, j );
            auto Bik = B( i, k );
            int64_t mb = Aij.mb();
            assert( mb == Bik.mb() );
            blas::copy( mb, &Aij.at( 0, jj ), 1, &Bik.at( 0, kk ), 1 );
            ii += mb;
        }
    }
}

//------------------------------------------------------------------------------
/// Computes the updated eigensystem of a diagonal matrix after
/// modification by a rank-one symmetric matrix, in parallel.
/// \[
///     T = Q_{in} ( D_{in} + \rho Z Z^H ) Q_{in}^H = Q_{out} D_{out} Q_{out}^H
/// \]
/// where $z = Q^H v$ and $v = [ e_{n1} e_1 ]$ is a vector of length $n$
/// with ones in the $n1$ and $n1 + 1$ elements and zeros elsewhere.
///
/// The eigenvectors of the original matrix are stored in Q, and the
/// eigenvalues are in D. The algorithm consists of three stages:
///
/// The first stage consists of deflating the size of the problem
/// when there are multiple eigenvalues or if there is a zero in
/// the $z$ vector. For each such occurence, the dimension of the
/// secular equation problem is reduced by one. This stage is
/// performed by the routine stedc_deflate.
///
/// The second stage consists of calculating the updated
/// eigenvalues. This is done by finding the roots of the secular
/// equation via the LAPACK routine laed4, called by stedc_secular.
/// This routine also calculates the eigenvectors of the current
/// problem.
///
/// The final stage consists of computing the updated eigenvectors
/// directly using the updated eigenvalues. The eigenvectors for
/// the current problem are multiplied with the eigenvectors from
/// the overall problem.
///
/// Corresponds to ScaLAPACK pdlaed1.
//------------------------------------------------------------------------------
/// @tparam real_t
///     One of float, double.
//------------------------------------------------------------------------------
/// @param[in,out] D
///     On entry, the eigenvalues of two subproblems.
///     On exit, the eigenvalues of the merged problem. (Not sorted?)
///
/// @param[in,out] Q
///     On entry, Q contains eigenvectors of subproblems.
///     On exit, Q contains the orthonormal eigenvectors of the
///     merged problem.
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
void stedc_merge(
    int64_t n, int64_t n1,
    real_t rho,
    real_t* D,
    Matrix<real_t>& Q,
    Matrix<real_t>& Qbar,
    Matrix<real_t>& U,
    Options const& opts )
{
    // Constants
    const real_t zero = 0.;
    const real_t one  = 1.;

    int nprow, npcol, myrow, mycol;
    Q.gridinfo( &nprow, &npcol, &myrow, &mycol );

    int64_t nb  = Q.tileNb( 0 );
    int64_t nt  = Q.nt();
    int64_t nt1 = nt / 2;  // smaller half first.
    assert( n1 == nt1 * nb );

    std::vector<real_t>  Dsecular( n ), z( n ), zsecular( n );
    std::vector<int64_t> ibar( n );

    int64_t nsecular  = 0;
    int64_t nU123     = -1;
    int64_t Q12_begin = -1, Q12_end = -1;
    int64_t Q23_begin = -1, Q23_end = -1;

    stedc_z_vector( Q, z );

    stedc_deflate( n, n1, rho,
                   &D[0], &Dsecular[0],
                   &z[0], &zsecular[0],
                   Q, Qbar, &ibar[0],
                   nsecular, nU123,
                   Q12_begin, Q12_end,
                   Q23_begin, Q23_end,
                   opts );

    if (nsecular > 0) {
        // todo: Is there reason to set U? It gets overwritten, right?
        set( zero, one, U );  // U = Identity

        stedc_secular( nsecular, n, rho,
                       &Dsecular[0], &zsecular[0], &D[0], U,
                       &ibar[0], opts );

        // Compute the updated eigenvectors.
        // Q = [ Q11  Q12   0   Q13 ]  N1 (size of Q1 before deflation)
        //     [  0   Q21  Q22  Q23 ]  N2 (size of Q2 before deflation)
        //        n1   r    n2   k'    r = n - n1 - n2 - k', k' = min( k1, k2 )
        //
        // Q*U = N1 [ Q11  Q12   0   Q13 ] [ U1    ] n1
        //       N2 [  0   Q21  Q22  Q23 ] [ U2    ] r = n - n1 - n2 - k'
        //                                 [ U3    ] n2
        //                                 [     I ] k'
        //
        // Q*U = { [ Q11  Q12 ] [ U1 ] }
        //       {              [ U2 ] }
        //       {                     }
        //       { [ Q21  Q22 ] [ U2 ] }
        //       {              [ U3 ] }
        //
        int64_t U_begin = std::min( Q12_begin, Q23_begin );
        int64_t U_end   = U_begin + nU123;

        // Convert to tile indices. todo: assumes fixed size tiles.
        U_begin /= nb;
        U_end = (U_end - 1) / nb;
        if (Q12_begin < Q12_end) {
            // Convert to tile indices. todo: assumes fixed size tiles.
            // todo: could slice matrices, but gemm<Devices> doesn't
            // support arbitrary slices.
            Q12_begin /= nb;
            Q12_end = (Q12_end - 1) / nb;
            auto Qbar12 = Qbar.sub( 0, nt1 - 1, Q12_begin, Q12_end );
            auto U12 = U.sub( Q12_begin, Q12_end, U_begin, U_end );
            auto Q12 = Q.sub( 0, nt1 - 1, U_begin, U_end );
            gemm( one, Qbar12, U12, zero, Q12 );
        }

        if (Q23_begin < Q23_end) {
            // Convert to tile indices. todo: see above.
            Q23_begin /= nb;
            Q23_end = (Q23_end - 1) / nb;
            auto Qbar23 = Qbar.sub( nt1, nt - 1, Q23_begin, Q23_end );
            auto U23 = U.sub( Q23_begin, Q23_end, U_begin, U_end );
            auto Q23 = Q.sub( nt1, nt - 1, U_begin, U_end );
            gemm( one, Qbar23, U23, zero, Q23 );
        }

        int r0 = Q.tileRank( 0, 0 );
        int dcol = r0 / nprow;  // todo: assumes col-major grid

        // Copy deflated eigenvectors from Qbar to Q (local operation).
        // Why not just put them into Q in the first place??? Ah... permuting.
        for (int64_t j = nsecular; j < n; ++j) {
            int64_t kg = ibar[ j ]; // global index
            int64_t k  = kg / nb;   // block index
            int64_t kk = kg % nb;   // offset within block
            int64_t pk = (k + dcol) % npcol; // process column
            if (pk == mycol) {
                copy_col( Qbar, k, kk, Q, k, kk );
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Only real, not complex.
template
void stedc_merge<float>(
    int64_t n, int64_t n1,
    float rho,
    float* D,
    Matrix<float>& Q,
    Matrix<float>& Qbar,
    Matrix<float>& U,
    Options const& opts );

template
void stedc_merge<double>(
    int64_t n, int64_t n1,
    double rho,
    double* D,
    Matrix<double>& Q,
    Matrix<double>& Qbar,
    Matrix<double>& U,
    Options const& opts );

} // namespace slate
