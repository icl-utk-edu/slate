// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/Array2D.hh"

#include <numeric>

namespace slate {

//------------------------------------------------------------------------------
/// Sorts the two sets of eigenvalues together into a single sorted set,
/// then deflates eigenvalues, which both resolves stability issues (division
/// by zero) and reduces the size of the problem. There are two ways in
/// which deflation can occur:
///   1) There is a zero or tiny entry in the z vector, indicating an
///      eigenvalue of the subproblems is already converged to an
///      eigenvalue of the merged problem, so it can be deflated.
///   2) Two eigenvalues are (nearly) identical. In this case, a Givens
///      rotation is applied to zero an entry of z, and the corresponding
///      eigenvalue can be deflated.
/// For each such occurrence the order of the related secular equation
/// problem is reduced by one.
///
/// Corresponds to ScaLAPACK pdlaed2. (Was: indicates ScaLAPACK names.)
//------------------------------------------------------------------------------
/// @tparam real_t
///     One of float, double.
//------------------------------------------------------------------------------
/// @param[in] n
///     Size of merged system. n = n1 + n2.
///
/// @param[in] n1
///     Size of first subproblem, D1.
///     Note that n2, the size of second subproblem, D2, is not passed
///     and is computed as n2 = n - n1.
///
/// @param[in,out] rho
///     On entry, the off-diagonal element associated with the rank-1
///     cut that originally split the two submatrices to be merged.
///     On exit, rho has been scaled to be positive and make $z$ unit norm,
///     as required by stedc_secular.
///
/// @param[in,out] D
///     Real vector of dimension n.
///     On entry,
///     D1 = D[ 0 : n1-1 ] contains eigenvalues of the first subproblem,
///     D2 = D[ n1 : n-1 ] contains eigenvalues of the second subproblem.
///     On exit, deflated eigenvalues are in decreasing order at the
///     *local* end within a process column (pcol) per the 2D block
///     cyclic distribution of Q.
///     Example on 2 ranks with nb = 4. To easily track values,
///     let D1 be multiples of 3, and D2 be even numbers.
///     Deflate arbitrary values 6, 9, 21, 4, 12 (see ^ marks). Value 6
///     is type 2 deflation since it is repeated; the rest are type 1
///     deflation. Note 9, 6, 4 stay on pcol 0, and 21, 12 stay on pcol 1.
///         pcol     = [ 0  0  0  0 |  1  1  1  1 |  0  0  0  0 |  1  1  1  1 ]
///         D_in     = [ 3  6  9 12 | 15 18 21 24 |  2  4  6  8 | 10 12 14 16 ]
///                    [    ^  ^    |        ^    |     ^       |     ^       ]
///         D_out    = [ #  #  #  # |  #  #  #  # |  #  9  6  4 |  #  # 21 12 ]
///                    [            |             |     ^  ^  ^ |        ^  ^ ]
///         Dsecular = [ 2  3  6  8 | 10 12 14 15 | 16 18 24  # |  #  #  #  # ]
///     where | separate blocks, and # values are not set.
///
/// @param[out] Dsecular
///     Real vector of dimension n.
///     On exit, Dsecular[ 0 : nsecular-1 ] contains non-deflated eigenvalues,
///     sorted in increasing order. (Was: dlamda)
///
/// @param[in,out] z
///     Real vector of dimension n.
///     On entry, z is the updating vector,
///         z = Q^T v = [ Q1^T  0    ] [ e_n1 ],
///                     [ 0     Q2^T ] [ e_1  ]
///     which is the last row of Q1 and the first row of Q2.
///     On exit, the contents of z are destroyed.
///
/// @param[out] zsecular
///     Real vector of dimension n.
///     On exit, zsecular[ 0 : nsecular-1 ] has non-deflated entries of z,
///     as updated by normalizing and applying Givens rotations in deflation,
///     for use by stedc_secular. (Was: w)
///
/// @param[in,out] Q
///     Real n-by-n matrix.
///     On entry, the eigenvectors of the two subproblems,
///         Q = [ Q1  0  ].
///             [ 0   Q2 ]
///     On exit, eigenvectors associated with type 2 deflation have been
///     modified.
///
/// @param[in,out] Qtype
///     Real n-by-n matrix.
///     A copy of all the eigenvectors locally ordered by column type such that
///         Qtype(:, itype) = Q(:, ideflate).
///     Non-deflated eigenvectors will be used by stedc_merge in a matrix
///     multiply (gemm) to solve for the new eigenvectors; deflated
///     eigenvectors will be copied back to Q in the correct place.
///
/// @param[out] ct_count
///     Integer npcol-by-5 array.
///     On exit, ct_count( pcol, ctype ) is the number of columns of
///     column type ctype on process column pcol.
///     ctype = 1:4, column ctype = 0 is unused.
///
/// @param[out] itype
///     Integer vector of dimension n.
///     On exit, permutation to arrange columns of Qtype locally into 4
///     column types based on block structure:
///     1: non-zero in upper half only (rows 0 : n1-1).
///     2: non-zero in all rows; non-deflated eigenvector from type 2 deflation
///         when one vector is from Q1 and the other vector is from Q2.
///     3: non-zero in lower half only (rows n1 : n-1).
///     4: may be non-zero in all rows; deflated eigenvectors.
///     (Was: indx)
///
/// @param[out] nsecular
///     On exit, number of non-deflated eigenvalues. (Was: k)
///
/// @param[out] Qtype12_begin
///     On exit, index of first column in Qtype sub-matrix spanning
///     column types 1 and 2.
///     Because of local permutation, this may include columns of
///     types 3 and 4. (Was: ib1)
///
/// @param[out] Qtype12_end
///     On exit, index of last column + 1 in Qtype sub-matrix spanning
///     column types 1 and 2. (Was: ib1 + nn1)
///
/// @param[out] Qtype23_begin
///     On exit, index of first column in Qtype sub-matrix spanning
///     column types 2 and 3.
///     Because of local permutation, this may include columns of
///     types 1 and 4. (Was: ib2)
///
/// @param[out] Qtype23_end
///     On exit, index of last column + 1 in Qtype sub-matrix spanning
///     column types 2 and 3. (Was: ib2 + nn2)
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
void stedc_deflate(
    int64_t n,
    int64_t n1,
    real_t& rho,
    real_t* D, real_t* Dsecular,
    real_t* z, real_t* zsecular,
    Matrix<real_t>& Q,
    Matrix<real_t>& Qtype,
    int64_t* itype,
    int64_t& nsecular,
    int64_t& Qtype12_begin, int64_t& Qtype12_end,
    int64_t& Qtype23_begin, int64_t& Qtype23_end,
    Options const& opts )
{
    int mpi_rank = Q.mpiRank();
    MPI_Comm comm = Q.mpiComm();

    // Constants.
    const int tag_0 = 0;
    const MPI_Datatype mpi_real_t = mpi_type<real_t>::value;

    // Check arguments.
    int64_t n2 = n - n1;
    assert( n1 > 0 );
    assert( n2 > 0 );
    int64_t nt = Q.nt();
    assert( nt == Q.mt() );  // square
    assert( n == Q.n() );
    assert( n == Q.m() );
    assert( Q.mt() == Qtype.mt() );
    assert( Q.nt() == Qtype.nt() );
    assert( Q.m() == Qtype.m() );
    assert( Q.n() == Qtype.n() );

    // Assumes matrix is 2D block cyclic.
    GridOrder grid_order;
    int nprow, npcol, myrow, mycol;
    Q.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    slate_assert( nprow > 0 );  // require 2D block-cyclic
    slate_assert( grid_order == GridOrder::Col );

    // Initial values, if all eigenvalues were deflated.
    Qtype12_begin = 0;
    Qtype12_end   = 0;
    Qtype23_begin = 0;
    Qtype23_end   = 0;

    // nsecular is number of non-deflated eig;
    // idx_deflate is index past-the-last of deflated eig, counting backwards.
    // In the ideflate permutation:
    //     ideflate( 0           : nsecular-1    ) are non-deflated.
    //     ideflate( nsecular    : idx_deflate-1 ) are not yet set.
    //     ideflate( idx_deflate : n-1           ) are deflated.
    nsecular = 0;
    int64_t idx_deflate = n;

    // Normalize so || z ||_2 = 1.
    // z1 and z2 are already normalized, since they are rows from
    // unitary matrices Q1 and Q2, so just divide by sqrt(2).
    // Secular equation solver (laed4) requires rho > 0, so move sign to z2.
    // rho = abs( || z_orig ||^2 * rho ) = abs( 2 * rho )
    const real_t r_sqrt2 = 1/sqrt(2);
    blas::scal( n1, r_sqrt2, z, 1 );  // scale z1
    blas::scal( n2, r_sqrt2 * sign( rho ), &z[ n1 ], 1 );  // scale z2
    rho = std::abs( 2 * rho );

    // Calculate the allowable deflation tolerance
    real_t zmax = std::abs( z[ blas::iamax( n, z, 1 ) ] );
    real_t dmax = std::abs( D[ blas::iamax( n, D, 1 ) ] );
    // assert_warn( 1/sqrt( n ) <= zmax && zmax <= 1 );

    // Deflation tolerance, per (Sca)LAPACK; doesn't quite match paper.
    // || z ||_2 = 1, so zmax isn't too large or small: 1/sqrt(n) <= zmax <= 1.
    // Note lamch('e') == unit_roundoff == 0.5 * epsilon.
    real_t unit_roundoff = 0.5 * std::numeric_limits<real_t>::epsilon();
    real_t tol = 8 * unit_roundoff * std::max( dmax, zmax );

    // If rank-1 modifier, rho, is small, nothing to do (nsecular = 0).
    if (rho * zmax <= tol) {
        return;
    }

    // If there are multiple eigenvalues then the problem deflates. Here
    // the number of equal eigenvalues are found. As each equal
    // eigenvalue is found, an elementary reflector is computed to rotate
    // the corresponding eigensubspace so that the corresponding
    // components of z are zero in this new basis.

    //-----
    // Local variables & arrays, after quick return.
    // D( isort ) are ascending.
    // After deflation:
    //     D( ideflate( 0 : nsecular-1 ) ) are non-deflated, ascending;
    //     D( ideflate( nsecular : n-1 ) ) are deflated, descending.
    std::vector<real_t> buf_vector( n );  // todo: only need nb?
    real_t* buf = buf_vector.data();
    std::vector<int64_t> isort( n );        // (Was: indx, as workspace)
    std::vector<int64_t> ideflate( n );     // (Was: indxp)
    std::vector<int64_t> iglobal( n );      // (Was: indxc)

    // Determine permutation isort so eigenvalues D[ isort ] are ascending.
    // iota fills isort = [ 0, 1, ..., n-1 ]
    std::iota( isort.begin(), isort.end(), 0 );
    std::sort( isort.begin(), isort.end(),
               [&D](int64_t const& i_, int64_t const& j_) {
                   return D[i_] < D[j_];
               } );

    // coltype[ 0 : n1-1 ] = 1: values from D1 that are not (yet) deflated.
    // coltype[ n1 : n-1 ] = 3: values from D2 that are not (yet) deflated.
    std::vector<int> coltype( n );
    std::fill( &coltype[ 0  ], &coltype[ n1 ], 1 );
    std::fill( &coltype[ n1 ], &coltype[ n  ], 3 );

    // Set pcols( j ) = process column of D(j).
    int64_t nb = Q.tileNb( 0 );
    std::vector<int> pcols( n );

    int r0 = Q.tileRank( 0, 0 );
    int dcol = r0 / nprow;  // todo: assumes col-major grid

    {
        // j is col index, jj is block-col index.
        int pcol = dcol;
        int64_t j = 0;
        for (int64_t jj = 0; jj < nt; ++jj) {
            int64_t jb = Q.tileNb( jj );
            std::fill( &pcols[ j ], &pcols[ j + jb ], pcol );
            j += jb;
            pcol = (pcol + 1) % npcol;
        }
    }

    // Search for eigenvalues to deflate.
    // Any negligble z( js2 ) are deflated.
    // z( js1 ) is most recently found non-negligible candidate;
    // -1 indicates none found yet.
    // Finds next non-negligible value, z( js2 ), and checks if we can rotate
    // [ z(js1), z(js2) ] to eliminate z(js1) while keeping the off-diagonal
    // element negligible; if so, deflate js1.
    // Otherwise stores js1 as next eigenvalue.
    // js1 = js2 as next non-negligible candidate to consider.
    // After loop, store last js1 as last eigenvalue.
    int64_t js1 = -1;                   // (Was: pj, prev j)
    for (int64_t j = 0; j < n; ++j) {
        int64_t js2 = isort[ j ];       // (Was: nj, next j)

        if (std::abs( rho * z[ js2 ] ) <= tol) {
            // Deflate due to small z component.
            idx_deflate -= 1;
            coltype[ js2 ] = 4;
            ideflate[ idx_deflate ] = js2;
        }
        else if (js1 >= 0) {
            // After we've found the first non-negligible eigenvalue,
            // check if eigenvalues are close enough to allow deflation.
            // z[ js1 ], z[ js2 ] are both non-negligible.
            // Define Givens rotation to eliminate z[ js1 ],
            //     G = [  c  s ].
            //         [ -s  c ]
            real_t s = -z[ js1 ];
            real_t c =  z[ js2 ];
            real_t tau = lapack::lapy2( c, s );  // == sqrt( c*c + s*s )
            s /= tau;
            c /= tau;
            // Check off-diagonal element after applying G on both sides:
            //     [ D(js1)   offdiag ] = G * [ D(js1)  0      ] * G^T.
            //     [ offdiag  D(js2)  ]       [ 0       D(js2) ]
            real_t offdiag = c * s * (D[ js2 ] - D[ js1 ]);
            if (std::abs( offdiag ) <= tol) {
                // Deflation is possible.
                // Apply Givens rotation to zero out z[ js1 ].
                z[ js1 ] = 0;
                z[ js2 ] = tau;

                // If one of js1, js2 is in Q1 (col type 1)
                //      and the other is in Q2 (col type 3),
                // or js1 is already col type 2 from earlier deflation,
                // then js2 moves to col type 2 due to Givens rotation.
                // js1 moves to col type 4 (deflated).
                if (coltype[ js1 ] != coltype[ js2 ]) {
                    coltype[ js2 ] = 2;
                }
                coltype[ js1 ] = 4;

                // Map js[12] col to jj[12] block col & offset.
                int64_t jj1        = js1 / nb;
                int64_t jj1_offset = js1 % nb;
                int64_t jj2        = js2 / nb;
                int64_t jj2_offset = js2 % nb;

                // Apply Givens rotation on right to columns js1, js2 of Q
                // Q( :, [js1, js2] ) = Q( :, [js1, js2] ) * G';
                for (int64_t ii = 0; ii < nt; ++ii) {
                    int64_t mb = Q.tileMb( ii );
                    int rank1 = Q.tileRank( ii, jj1 );
                    int rank2 = Q.tileRank( ii, jj2 );
                    if (rank1 == mpi_rank && rank2 == mpi_rank) {
                        // Both tiles are local; apply rot.
                        auto T1 = Q( ii, jj1 );
                        auto T2 = Q( ii, jj2 );
                        real_t* x1 = &T1.at( 0, jj1_offset );
                        real_t* x2 = &T2.at( 0, jj2_offset );
                        blas::rot( mb, x1, 1, x2, 1, c, s );
                    }
                    else if (rank1 == mpi_rank) {
                        // js1 is local; send js1, recv js2, apply rot.
                        // rot is applied redundantly on both rank1 and rank2;
                        // buf contents are discarded.
                        auto T1 = Q( ii, jj1 );
                        real_t* x1 = &T1.at( 0, jj1_offset );
                        slate_mpi_call(
                            MPI_Sendrecv(
                                x1,  mb, mpi_real_t, rank2, tag_0,
                                buf, mb, mpi_real_t, rank2, tag_0,
                                comm, MPI_STATUS_IGNORE ));
                        blas::rot( mb, x1, 1, buf, 1, c, s );
                    }
                    else if (rank2 == mpi_rank) {
                        // js2 is local; recv js1, send js2, apply rot as above.
                        auto T2 = Q( ii, jj2 );
                        real_t* x2 = &T2.at( 0, jj2_offset );
                        slate_mpi_call(
                            MPI_Sendrecv(
                                x2,  mb, mpi_real_t, rank1, tag_0,
                                buf, mb, mpi_real_t, rank1, tag_0,
                                comm, MPI_STATUS_IGNORE ));
                        blas::rot( mb, buf, 1, x2, 1, c, s );
                    }
                }

                // Apply Givens rotation on both sides of D (see offdiag above).
                // Off-diagonal elements are negligible, per if condition.
                real_t D_js1 = D[ js1 ];
                D[ js1 ] = D_js1*(c*c) + D[ js2 ]*(s*s);
                D[ js2 ] = D_js1*(s*s) + D[ js2 ]*(c*c);

                // Shift deflated eigenvalues down until we find where to
                // insert D( js1 ) in descending order.
                int64_t i = idx_deflate;
                idx_deflate -= 1;
                while (i < n) {
                    if (D[ js1 ] >= D[ ideflate[ i ] ])
                        break;
                    ideflate[ i-1 ] = ideflate[ i ];
                    i += 1;
                }
                // Insert D[ js1 ].
                ideflate[ i-1 ] = js1;
            }
            else {
                // No deflation, record js1 as eigenvalue.
                Dsecular[ nsecular ] = D[ js1 ];
                zsecular[ nsecular ] = z[ js1 ];
                ideflate[ nsecular ] = js1;
                nsecular += 1;
            }

            // Move to next candidate.
            js1 = js2;
        }
        else {
            // First non-negligible eigenvalue candidate.
            js1 = js2;
        }
    }
    // Record last js1 as eigenvalue (same code as above).
    // Guaranteed to be at least one non-deflated eigenvalue here,
    // due to early exit above if all eigenvalues are negligible.
    assert( js1 != -1 );
    Dsecular[ nsecular ] = D[ js1 ];
    zsecular[ nsecular ] = z[ js1 ];
    ideflate[ nsecular ] = js1;
    nsecular += 1;

    assert( nsecular == idx_deflate );

    //----------------------------------------
    // Locally permute to sort col types:
    //     Qtype_local = [ Q11  Q12  0    Q14 ]
    //                   [ 0    Q22  Q23  Q24 ]
    // col type:             1    2    3    4
    // Globally Qtype does not have this structure. See SWAN 13 for examples.
    // Pg = iglobal is another permutation such that (Pg Pt^T Pd coltype)
    // is ordered globally. It is used here only to compute
    // Qtype{12,23}_{begin,end}, the first and last+1 columns
    // of Qtype of column types (1,2) and (2,3), respectively.

    // Note in all these arrays, column ctype == 0 is ignored for
    // compatibility with Fortran numbering.

    // Count number of columns of each type in each process column.
    // ct_count( pcol, ctype ) is number of matrix columns of type
    // ctype that process column pcol owns.
    internal::Array2D<int64_t> ct_count( npcol, 5 );
    for (int64_t j = 0; j < n; ++j) {
        int ctype = coltype[ j ];
        int pcol  = pcols[ j ];
        ct_count( pcol, ctype ) += 1;
    }

    // ct_idx_local is local start of each col type on each process col.
    // (Was: PSM = Position in SubMatrix)
    internal::Array2D<int64_t> ct_idx_local( npcol, 5 );
    for (int pcol = 0; pcol < npcol; ++pcol) {
        ct_idx_local( pcol, 1 ) = 0;
        ct_idx_local( pcol, 2 ) = ct_idx_local( pcol, 1 ) + ct_count( pcol, 1 );
        ct_idx_local( pcol, 3 ) = ct_idx_local( pcol, 2 ) + ct_count( pcol, 2 );
        ct_idx_local( pcol, 4 ) = ct_idx_local( pcol, 3 ) + ct_count( pcol, 3 );
    }

    // ct_idx_global is global start of each col type across all processes.
    // (Was: PTT)
    std::vector<int64_t> ct_idx_global( 5 );
    ct_idx_global[ 1 ] = 0;
    for (int ctype = 2; ctype <= 4; ++ctype) {
        int64_t sum = 0;
        for (int pcol = 0; pcol < npcol; ++pcol) {
            sum += ct_count( pcol, ctype-1 );
        }
        ct_idx_global[ ctype ] = ct_idx_global[ ctype-1 ] + sum;
    }

    // Fill in itype to locally order all type 1 columns first, then type 2,
    // then type 3, then type 4.
    //
    // This merges 3 loops from ScaLAPACK.
    // For permuting D, uses z as workspace, since z was copied to zsecular.
    std::copy( &D[ 0 ], &D[ n ], z );
    for (int64_t j = 0; j < n; ++j) {
        int64_t jd, jt, jt_local, jg;
        int pcol, ctype;
        // jd is index after deflation.
        // jt is for Qtype,   locally  permuted to order coltype.
        // jg is for Qglobal, globally permuted to order coltype.
        jd = ideflate[ j ];
        pcol = pcols[ jd ];
        ctype = coltype[ jd ];
        jt_local = ct_idx_local( pcol, ctype )++;
        jt = local2global( jt_local, nb, pcol, dcol, npcol );
        itype[ j ] = jt;  // was isort (INDX)
        jg = ct_idx_global[ ctype ]++;
        iglobal[ jg ] = jt;

        // Copy & permute Q(:, ideflate(j)) => Qtype(:, itype(j)),
        // if this process owns them.
        if (pcol == mycol) {
            int64_t jjd        = jd / nb;
            int64_t jjd_offset = jd % nb;
            int64_t jjt        = jt / nb;
            int64_t jjt_offset = jt % nb;
            for (int64_t ii = 0; ii < nt; ++ii) {
                int64_t mb = Q.tileMb( ii );
                if (Q.tileIsLocal( ii, jjd )) {
                    assert( Qtype.tileIsLocal( ii, jjt ) );
                    auto T1 =     Q( ii, jjd );
                    auto T2 = Qtype( ii, jjt );
                    real_t* x1 = &T1.at( 0, jjd_offset );
                    real_t* x2 = &T2.at( 0, jjt_offset );
                    blas::copy( mb, x1, 1, x2, 1 );
                }
            }
        }

        // The deflated eigenvalues and their corresponding vectors go
        // into the last (n_local - nsecular_local) local slots of D and Qtype,
        // respectively. Qtype is handled above.
        if (j >= nsecular) {
            D[ jt ] = z[ jd ];  // i.e., Dorig[ jd ] stored in z.
        }
        //else {
        //    // For debugging, set non-deflated entries to NaN for easy id.
        //    D[ jt ] = nan("");
        //}
    }

    // Restore ct_idx_global (same code as above).
    ct_idx_global[ 1 ] = 0;
    for (int ctype = 2; ctype <= 4; ++ctype) {
        int64_t sum = 0;
        for (int pcol = 0; pcol < npcol; ++pcol) {
            sum += ct_count( pcol, ctype-1 );
        }
        ct_idx_global[ ctype ] = ct_idx_global[ ctype-1 ] + sum;
    }

    // Find begin and end of Q12 (ctype 1 and 2).
    // As usual in C++, `end` points past-the-last.
    // This implementation starts with empty array, then grows it,
    // unlike ScaLAPACK which starts with a 1-element array and can't
    // make an empty array.
    Qtype12_begin = iglobal[ ct_idx_global[ 1 ] ];
    Qtype12_end   = Qtype12_begin;
    for (int64_t j = ct_idx_global[ 1 ]; j < ct_idx_global[ 3 ]; ++j) {
        Qtype12_begin = blas::min( Qtype12_begin, iglobal[ j ] );
        Qtype12_end   = blas::max( Qtype12_end,   iglobal[ j ] + 1 );
    }

    // Find begin and end of Q23 (ctype 2 and 3).
    Qtype23_begin = iglobal[ ct_idx_global[ 2 ] ];
    Qtype23_end   = Qtype23_begin;
    for (int64_t j = ct_idx_global[ 2 ]; j < ct_idx_global[ 4 ]; ++j) {
        Qtype23_begin = blas::min( Qtype23_begin, iglobal[ j ] );
        Qtype23_end   = blas::max( Qtype23_end,   iglobal[ j ] + 1 );
    }

}

//------------------------------------------------------------------------------
// Explicit instantiations.
// Only real, not complex.
template
void stedc_deflate<float>(
    int64_t n,
    int64_t n1,
    float& rho,
    float* D, float* Dsecular,
    float* z, float* zsecular,
    Matrix<float>& Q,
    Matrix<float>& Qtype,
    int64_t* itype,
    int64_t& nsecular,
    int64_t& Qtype12_begin, int64_t& Qtype12_end,
    int64_t& Qtype23_begin, int64_t& Qtype23_end,
    Options const& opts );

template
void stedc_deflate<double>(
    int64_t n,
    int64_t n1,
    double& rho,
    double* D, double* Dsecular,
    double* z, double* zsecular,
    Matrix<double>& Q,
    Matrix<double>& Qtype,
    int64_t* itype,
    int64_t& nsecular,
    int64_t& Qtype12_begin, int64_t& Qtype12_end,
    int64_t& Qtype23_begin, int64_t& Qtype23_end,
    Options const& opts );

} // namespace slate
