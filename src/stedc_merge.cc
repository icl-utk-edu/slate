// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal_copy_col.hh"

namespace slate {

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
/// @param[in] n
///     Size of merged system. n = n1 + n2.
///
/// @param[in] n1
///     Size of first subproblem, D1.
///     Size of second subproblem, D2, is n2 = n - n1.
///
/// @param[in,out] rho
///     On entry, the off-diagonal element associated with the rank-1
///     cut that originally split the two submatrices to be merged.
///     On exit, updated by deflation process.
///
/// @param[in,out] D
///     On entry, the eigenvalues of two subproblems.
///     On exit, the eigenvalues of the merged problem, not sorted.
///
/// @param[in,out] Q
///     On entry, Q contains eigenvectors of subproblems.
///     On exit, Q contains the orthonormal eigenvectors of the
///     merged problem.
///
/// @param[out] Qtype
///     Qtype is a workspace, the same size as Q.
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
void stedc_merge(
    int64_t n, int64_t n1,
    real_t rho,
    real_t* D,
    Matrix<real_t>& Q,
    Matrix<real_t>& Qtype,
    Matrix<real_t>& U,
    Options const& opts )
{
    // Constants
    const real_t zero = 0.;
    const real_t one  = 1.;

    // Assumes matrix is 2D block cyclic.
    GridOrder grid_order;
    int nprow, npcol, myrow, mycol;
    Q.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    slate_assert( nprow > 0 );  // require 2D block-cyclic
    slate_assert( grid_order == GridOrder::Col );

    int64_t nb  = Q.tileNb( 0 );
    int64_t nt  = Q.nt();
    int64_t nt1 = nt / 2;  // smaller half first.
    assert( n1 == nt1 * nb );

    std::vector<real_t>  Dsecular( n ), z( n ), zsecular( n );
    std::vector<int64_t> itype( n );

    int64_t nsecular   = 0;
    int64_t Qt12_begin = -1, Qt12_end = -1;
    int64_t Qt23_begin = -1, Qt23_end = -1;

    stedc_z_vector( Q, z );

    stedc_deflate( n, n1, rho,
                   &D[0], &Dsecular[0],
                   &z[0], &zsecular[0],
                   Q, Qtype, &itype[0],
                   nsecular,
                   Qt12_begin, Qt12_end,
                   Qt23_begin, Qt23_end,
                   opts );

    if (nsecular > 0) {
        // Set U = Identity for deflated eigenvectors (lower right of U).
        set( zero, one, U );

        stedc_secular( nsecular, n, rho,
                       &Dsecular[0], &zsecular[0], &D[0], U,
                       &itype[0], opts );

        // Compute the updated eigenvectors.
        // Qt is Qtype, locally permuted into col types 1, 2, 3, and 4:
        //     Qt = [ Qt_{1,1}  Qt_{1,2}  0         |  Qt_{1,4} ]  N1 rows
        //          [ 0         Qt_{2,2}  Qt_{2,3}  |  Qt_{2,4} ]  N2 rows
        // col type:  1         2         3         |  4
        // N1, N2 are the size of the Q1, Q2 subproblems before deflation.
        //
        // Due to Pt being a local permutation, in the parallel algorithm
        // the global structure is more complicated than this. Qt12 and
        // Qt23 can include columns of other column types. See SWAN 13.
        //
        // Qt*U = [ Qt_{1,1}  Qt_{1,2}  0         |  Qt_{1,4} ]*[ U_1    ]
        //        [ 0         Qt_{2,2}  Qt_{2,3}  |  Qt_{2,4} ] [ U_2    ]
        //                                                      [ U_3    ]
        //                                                      [      I ]
        //
        //      = [ Qt_{1,1:2} * U_{1:2}  |  Qt_{1,4} ]
        //        [ Qt_{2,2:3} * U_{2:3}  |  Qt_{2,4} ]
        //
        //      = [ Qt12 * U12  |  Qt_{1,4} ]
        //        [ Qt23 * U23  |  Qt_{2,4} ]
        //
        // Variable Qt12 is Qt_{1,1:2}, Qt23 is Qt_{2,2:3},
        // U12 is U_{1:2}, U23 is U_{2:3}, U123 is U_{1:3}.

        // U123 begin to end are cols in U1, U2, and U3.
        int64_t U123_begin = std::min( Qt12_begin, Qt23_begin );
        int64_t U123_end   = std::max( Qt12_end, Qt23_end );
        // Convert to tile indices. todo: assumes fixed size tiles.
        U123_begin /= nb;
        U123_end = (U123_end - 1) / nb;

        // Qt12_begin to end includes all cols of types 1 and 2, forming Qt12.
        // Due to local permutation, it can have some cols of types 3 and 4.
        if (Qt12_begin < Qt12_end) {
            // Convert to tile indices. todo: assumes fixed size tiles.
            // todo: could slice matrices, but gemm<Devices> doesn't
            // support arbitrary slices.
            Qt12_begin /= nb;
            Qt12_end = (Qt12_end - 1) / nb;
            auto Qt12 = Qtype.sub( 0, nt1 - 1, Qt12_begin, Qt12_end );
            auto U12 = U.sub( Qt12_begin, Qt12_end, U123_begin, U123_end );
            auto Q12 = Q.sub( 0, nt1 - 1, U123_begin, U123_end );
            gemm( one, Qt12, U12, zero, Q12, opts );
        }

        // Qt23_begin to end includes all cols of types 2 and 3, forming Qt23.
        // Due to local permutation, it can have some cols of types 3 and 4.
        if (Qt23_begin < Qt23_end) {
            // Convert to tile indices. todo: see above.
            Qt23_begin /= nb;
            Qt23_end = (Qt23_end - 1) / nb;
            auto Qt23 = Qtype.sub( nt1, nt - 1, Qt23_begin, Qt23_end );
            auto U23 = U.sub( Qt23_begin, Qt23_end, U123_begin, U123_end );
            auto Q23 = Q.sub( nt1, nt - 1, U123_begin, U123_end );
            gemm( one, Qt23, U23, zero, Q23, opts );
        }

        int r0 = Q.tileRank( 0, 0 );
        int dcol = r0 / nprow;  // todo: assumes col-major grid

        // Copy deflated eigenvectors from Qtype to Q (local operation).
        for (int64_t j = nsecular; j < n; ++j) {
            int64_t kg = itype[ j ]; // global index
            int64_t k  = kg / nb;   // block index
            int64_t kk = kg % nb;   // offset within block
            int64_t pk = (k + dcol) % npcol; // process column
            if (pk == mycol) {
                internal::copy_col( Qtype, k, kk, Q, k, kk );
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
    Matrix<float>& Qtype,
    Matrix<float>& U,
    Options const& opts );

template
void stedc_merge<double>(
    int64_t n, int64_t n1,
    double rho,
    double* D,
    Matrix<double>& Q,
    Matrix<double>& Qtype,
    Matrix<double>& U,
    Options const& opts );

} // namespace slate
