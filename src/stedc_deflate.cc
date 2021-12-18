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
/// then tries to deflate the size of the problem. There are two ways in
/// which deflation can occur:
///   1) there is a zero or tiny entry in the z vector, or
///   2) two eigenvalues are identical or close together.
/// For each such occurrence the order of the related secular equation
/// problem is reduced by one.
///
/// Corresponds to ScaLAPACK pdlaed2.
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
///     On exit, D[ nsecular : n-1 ] contains deflated eigenvalues,
///     sorted in increasing order.
///     todo: is that right? Or their distributed by some permutation?
///
/// @param[out] Dsecular
///     Real vector of dimension n.
///     On exit, Dsecular[ 0 : nsecular-1 ] contains non-deflated eigenvalues,
///     sorted in increasing order.
///
/// @param[in] z
///     Real vector of dimension n.
///     On entry, z is the updating vector,
///         z = Q^T v = [ Q1^T  0    ] [ e_n1 ],
///                     [ 0     Q2^T ] [ e_1  ]
///     that is the last row of Q1 and the first row of Q2.
///     On exit, the contents of z are destroyed.
///
/// @param[in] zsecular
///     Real vector of dimension n.
///     On exit, zsecular[ 0 : nsecular-1 ] has the deflation-altered z vector,
///     for use by laed3_secular.
///
/// @param[in,out] Q
///     Real n-by-n matrix.
///     On entry, the eigenvectors of the two subproblems,
///         Q = [ Q1  0  ].
///             [ 0   Q2 ]
///     On exit, the eigenvectors associated with deflated eigenvalues
///     are in the last (n - nsecular) columns.
///     todo: is that right? Or they are distributed by some permutation?
///
/// @param[in,out] Qbar
///     Real n-by-n matrix.
///     todo
///
/// @param[out] ct_count
///     Integer npcol-by-5 array.
///     On exit, ct_count( pcol, ctype ) is the number of columns of
///     column type ctype on process column pcol.
///     ctype = 1:4, column ctype = 0 is unused.
///
/// @param[out] ibar
///     Integer vector of dimension n.
///     On exit, permutation to arrange columns of Qbar locally into 4
///     column types:
///     1: non-zero in upper half only (rows 0 : n1-1).
///     2: dense; non-deflated eigenvector from type 2 deflation.
///     3: non-zero in lower half only (rows n1 : n-1).
///     4: dense; deflated eigenvectors.
///
/// @param[out] nsecular
///     On exit, number of non-deflated eigenvalues.
///
/// @param[out] nU123
///     Number of columns in U sub-matrix spanning column types 1, 2, 3.
///     Because of local permutation, this may include columns of
///     type 4, hence nsecular <= nU123 <= n.
///
/// @param[out] Qbar12_begin
///     On exit, index of first column in Qbar sub-matrix spanning
///     column types 1 and 2.
///     Because of local permutation, this may include columns of
///     types 3 and 4.
///
/// @param[out] Qbar12_end
///     On exit, index of last column + 1 in Qbar sub-matrix spanning
///     column types 1 and 2.
///
/// @param[out] Qbar23_begin
///     On exit, index of first column in Qbar sub-matrix spanning
///     column types 2 and 3.
///     Because of local permutation, this may include columns of
///     types 1 and 4.
///
/// @param[out] Qbar23_end
///     On exit, index of last column + 1 in Qbar sub-matrix spanning
///     column types 2 and 3.
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
    Matrix<real_t>& Qbar,
    int64_t* ibar,
    int64_t& nsecular,
    int64_t& nU123,
    int64_t& Qbar12_begin, int64_t& Qbar12_end,
    int64_t& Qbar23_begin, int64_t& Qbar23_end,
    Options const& opts )
{
    int mpi_rank = Q.mpiRank();
    MPI_Comm comm = Q.mpiComm();

    // Constants.
    const real_t one = 1.0;
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
    assert( Q.mt() == Qbar.mt() );
    assert( Q.nt() == Qbar.nt() );
    assert( Q.m() == Qbar.m() );
    assert( Q.n() == Qbar.n() );

    // Initial values, if all eigenvalues were deflated.
    nU123        = 0;
    Qbar12_begin = 0;
    Qbar12_end   = 0;
    Qbar23_begin = 0;
    Qbar23_end   = 0;

    // nsecular is number of non-deflated eig;
    // idx_deflate is index past-the-last of deflated eig, counting backwards:
    //     D( idx( 0           : nsecular-1    ) ) are non-deflated.
    //     D( idx( nsecular    : idx_deflate-1 ) ) are not yet processed.
    //     D( idx( idx_deflate : n-1           ) ) are deflated.
    nsecular = 0;
    int64_t idx_deflate = n;

    // Apply theta = sign( beta ) to z2 = z[ n1 : n-1 ].
    // todo: can merge these 2 scal for single pass:
    //     scal( n1, 1/sqrt(2), z )
    //     scal( n2, sign(rho) * 1/sqrt(2), z[n1] )
    if (rho < 0) {
        blas::scal( n2, -one, &z[ n1 ], 1 );
    }

    // rho = theta beta = |beta|.
    // Normalize so || z ||_2 = 1.
    // z1 and z2 are already normalized, so just divide by sqrt(2).
    // rho = abs( || z_orig ||^2 * rho ) = abs( 2 * rho )
    blas::scal( n, 1/sqrt(2), z, 1 );
    rho = std::abs( 2 * rho );

    // Calculate the allowable deflation tolerance
    real_t zmax = std::abs( z[ blas::iamax( n, z, 1 ) ] );
    real_t dmax = std::abs( D[ blas::iamax( n, D, 1 ) ] );
    // assert_warn( 1/sqrt( n ) <= zmax && zmax <= 1 );

    // Deflation tolerance, per (Sca)LAPACK; doesn't quite match paper.
    // || z ||_2 = 1, so zmax isn't too large or small: 1/sqrt(n) <= zmax <= 1.
    // Note lamch('e') == unit_roundoff.
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
    std::vector<real_t> buf_vector( n );  // todo: only need nb?
    real_t* buf = buf_vector.data();
    std::vector<int64_t> isort( n );
    std::vector<int64_t> ideflate( n );
    std::vector<int64_t> iglobal( n );

    // Determine permutation isort that sorts eigenvalues in D ascending.
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

    // Assumes matrix is 2D block cyclic.
    int nprow, npcol, myrow, mycol;
    Q.gridinfo( &nprow, &npcol, &myrow, &mycol );
    if (nprow <= 0)
        throw Exception( "requires 2D block cyclic distribution" );

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
    // [ z(js1), z(js2) ] to eliminate z(js1) while , deflating it.
    // Otherwise stores js1 as next eigenvalue.
    // js1 = js2 as next non-negligible candidate to consider.
    // After loop, store last js1 as last eigenvalue.
    int64_t js1 = -1;
    for (int64_t j = 0; j < n; ++j) {
        int64_t js2 = isort[ j ];

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
            //     [ D(js1)   offdiag ] = G * [ D(js1)        ] * G^T.
            //     [ offdiag  D(js2)  ]       [ 0      D(js2) ]
            real_t offdiag = c * s * (D[ js2 ] - D[ js1 ]);
            if (std::abs( offdiag ) <= tol) {
                // Deflation is possible.
                // Apply Givens rotation to zero out z[ js1 ].
                z[ js1 ] = 0;
                z[ js2 ] = tau;

                // If one of js1, js2 is in Q1 (col type 1)
                //      and the other is in Q2 (col type 3),
                // or js1 is already col type 2 (dense) from earlier deflation,
                // then js2 moves to col type 2 (dense) due to Givens rotation.
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
                    int r1 = Q.tileRank( ii, jj1 );
                    int r2 = Q.tileRank( ii, jj2 );
                    if (r1 == mpi_rank && r2 == mpi_rank) {
                        // Both tiles are local; apply rot.
                        auto T1 = Q( ii, jj1 );
                        auto T2 = Q( ii, jj2 );
                        real_t* x1 = &T1.at( 0, jj1_offset );  //T1.data() + jj1_offset*T1.stride();
                        real_t* x2 = &T2.at( 0, jj2_offset );  //T2.data() + jj2_offset*T2.stride();
                        blas::rot( mb, x1, 1, x2, 1, c, s );
                    }
                    else if (r1 == mpi_rank) {
                        // js1 is local; send js1, recv js2, apply rot.
                        // rot is applied redundantly on both r1 and r2;
                        // buf contents are discarded.
                        auto T1 = Q( ii, jj1 );
                        real_t* x1 = &T1.at( 0, jj1_offset );  //T1.data() + jj1_offset*T1.stride();
                        slate_mpi_call(
                            MPI_Sendrecv(
                                x1,  mb, mpi_real_t, r2, tag_0,
                                buf, mb, mpi_real_t, r2, tag_0,
                                comm, MPI_STATUS_IGNORE ));
                        blas::rot( mb, x1, 1, buf, 1, c, s );
                    }
                    else if (r2 == mpi_rank) {
                        // js2 is local; recv js1, send js2, apply rot as above.
                        auto T2 = Q( ii, jj2 );
                        real_t* x2 = &T2.at( 0, jj2_offset );  //T2.data() + jj2_offset*T2.stride();
                        slate_mpi_call(
                            MPI_Sendrecv(
                                x2,  mb, mpi_real_t, r1, tag_0,
                                buf, mb, mpi_real_t, r1, tag_0,
                                comm, MPI_STATUS_IGNORE ));
                        blas::rot( mb, buf, 1, x2, 1, c, s );
                    }
                }

                // Apply Givens rotation on both sides of D (see offdiag above).
                // Off-diagonal elements are negligible, per if condition.
                real_t D_js1 = D[ js1 ];
                D[ js1 ] = D_js1*(c*c) + D[ js2 ]*(s*s);
                D[ js2 ] = D_js1*(s*s) + D[ js2 ]*(c*c);

                idx_deflate -= 1;

                // Shift deflated eigenvalues down until we find where to
                // insert D( js1 ) in descending order.
                int64_t i = idx_deflate + 1;
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
    //     Qbar = [ Q11  Q12  0    Q14 ]
    //            [ 0    Q22  Q23  Q24 ]
    // col type:      1    2    3    4
    // Globally Qbar does not have this structure. The iglobal permutation ...

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
    // (was PSM = Position in SubMatrix)
    internal::Array2D<int64_t> ct_idx_local( npcol, 5 );
    for (int pcol = 0; pcol < npcol; ++pcol) {
        ct_idx_local( pcol, 1 ) = 0;
        ct_idx_local( pcol, 2 ) = ct_idx_local( pcol, 1 ) + ct_count( pcol, 1 );
        ct_idx_local( pcol, 3 ) = ct_idx_local( pcol, 2 ) + ct_count( pcol, 2 );
        ct_idx_local( pcol, 4 ) = ct_idx_local( pcol, 3 ) + ct_count( pcol, 3 );
    }

    // ct_idx_global is global start of each col type across all processes.
    // (was PTT)
    std::vector<int64_t> ct_idx_global( 5 );
    ct_idx_global[ 1 ] = 0;
    for (int ctype = 2; ctype <= 4; ++ctype) {
        int64_t sum = 0;
        for (int pcol = 0; pcol < npcol; ++pcol) {
            sum += ct_count( pcol, ctype-1 );
        }
        ct_idx_global[ ctype ] = ct_idx_global[ ctype-1 ] + sum;
    }

    // Fill in iglobal to order all type 1 columns first, then type 2,
    // then type 3, then type 4.
    // This merges 3 loops from ScaLAPACK.
    // For permuting D, uses z as workspace, since z was copied to zsecular.
    std::copy( &D[ 0 ], &D[ n ], z );
    //std::vector<real_t> Dorig( &D[ 0 ], &D[ n ] );
    for (int64_t j = 0; j < n; ++j) {
        int64_t jd, jbar, jbar_local, jg;
        int pcol, ctype;
        // jd   is index after deflation.
        // jbar is for Q_bar,    locally  permuted to order coltype.
        // jg   is for Q_global, globally permuted to order coltype.
        jd = ideflate[ j ];
        pcol = pcols[ jd ];
        ctype = coltype[ jd ];
        jbar_local = ct_idx_local( pcol, ctype )++;
        jbar = local2global( jbar_local, nb, pcol, dcol, npcol );
        ibar[ j ] = jbar;  // was isort (INDX)
        jg = ct_idx_global[ ctype ]++;
        iglobal[ jg ] = jbar;

        // Copy & permute Q(:, ideflate(j)) => Qbar(:, ibar(j)), if local.
        if (pcol == mycol) {
            // todo: assert( jbar_local == global2local( jbar, nb, npcol ) );
            // todo: int64_t jd_local = 0;  // todo: global2local( jd, nb, npcol );
            // todo: Q2( :, jbar_local ) = Q( :, jd_local );
            int64_t jjd          = jd / nb;
            int64_t jjd_offset   = jd % nb;
            int64_t jjbar        = jbar / nb;
            int64_t jjbar_offset = jbar % nb;
            for (int64_t ii = 0; ii < nt; ++ii) {
                int64_t mb = Q.tileMb( ii );
                if (Q.tileIsLocal( ii, jjd )) {
                    assert( Qbar.tileIsLocal( ii, jjbar ) );
                    auto T1 =    Q( ii, jjd );
                    auto T2 = Qbar( ii, jjbar );
                    real_t* x1 = &T1.at( 0, jjd_offset );  //data() + jjd_offset*T1.stride();
                    real_t* x2 = &T2.at( 0, jjbar_offset );  //T2.data() + jjbar_offset*T2.stride();
                    blas::copy( mb, x1, 1, x2, 1 );
                }
            }
        }

        // The deflated eigenvalues and their corresponding vectors go
        // into the last (n - nsecular_local) local slots of D and Qbar,
        // respectively. Qbar is handled above.
        if (j >= nsecular) {
            D[ jbar ] = z[ jd ];  // i.e., Dorig[ jd ]
            //z[ jbar ] = z[ jd ];
        }
        //else {
        //    D[ jbar ] = nan("");
        //    //z[ jbar ] = nan("");
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
    Qbar12_begin = iglobal[ ct_idx_global[ 1 ] ];
    Qbar12_end   = Qbar12_begin;
    for (int64_t j = ct_idx_global[ 1 ]; j < ct_idx_global[ 3 ]; ++j) {
        Qbar12_begin = blas::min( Qbar12_begin, iglobal[ j ] );
        Qbar12_end   = blas::max( Qbar12_end,   iglobal[ j ] + 1 );
    }

    // Find begin and end of Q23 (ctype 2 and 3).
    Qbar23_begin = iglobal[ ct_idx_global[ 2 ] ];
    Qbar23_end   = Qbar23_end;
    for (int64_t j = ct_idx_global[ 2 ]; j < ct_idx_global[ 4 ]; ++j) {
        Qbar23_begin = blas::min( Qbar23_begin, iglobal[ j ] );
        Qbar23_end   = blas::max( Qbar23_end,   iglobal[ j ] + 1 );
    }

    // Size of U123 sub-matrix.
    nU123 = std::max( Qbar12_end, Qbar23_end )
            - std::min( Qbar12_begin, Qbar23_begin );
    assert( nU123 >= nsecular );
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
    Matrix<float>& Qbar,
    int64_t* ibar,
    int64_t& nsecular,
    int64_t& nU123,
    int64_t& Qbar12_begin, int64_t& Qbar12_end,
    int64_t& Qbar23_begin, int64_t& Qbar23_end,
    Options const& opts );

template
void stedc_deflate<double>(
    int64_t n,
    int64_t n1,
    double& rho,
    double* D, double* Dsecular,
    double* z, double* zsecular,
    Matrix<double>& Q,
    Matrix<double>& Qbar,
    int64_t* ibar,
    int64_t& nsecular,
    int64_t& nU123,
    int64_t& Qbar12_begin, int64_t& Qbar12_end,
    int64_t& Qbar23_begin, int64_t& Qbar23_end,
    Options const& opts );

} // namespace slate
