// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <blas.hh>
#include <lapack.hh>
#include <slate/internal/util.hh>
#include <slate/types.hh>

#include <assert.h>
#include <complex>
#include <math.h>
#include <stdint.h>

#define LOOKAHEAD 1

namespace slate {

//------------------------------------------------------------------------------
/// Computes all eigenvalues and, optionally, eigenvectors of a
/// symmetric tridiagonal matrix using the implicit QL or QR method.
///
/// SLATE steqr is a modified, C++ version of LAPACK routines
/// {s,d,c,z}steqr and ScaLAPACK routines {s,d,c,z}steqr2.
/// It is modified from LAPACK steqr to allow each process
/// to perform updates on a 1D distributed matrix Z. Compare LAPACK bdsqr.
///
/// This is the low-level implementation, with similar semantics to LAPACK.
/// An overloaded higher-level wrapper is available (steqr.cc).
//------------------------------------------------------------------------------
/// @param[in] n
///     The order of the matrix. n >= 0.
///
/// @param[in,out] D
///     real array, dimension (n)
///     On entry, the diagonal elements of the tridiagonal matrix.
///     On exit, if info = 0, the eigenvalues in ascending order.
///
/// @param[in,out] E
///     real array, dimension (n-1)
///     On entry, the (n-1) subdiagonal elements of the tridiagonal
///     matrix.
///     On exit, E has been destroyed.
///
/// @param[in,out] Z
///     scalar array, dimension (ldz, n).
///     On entry, the local nrows-by-n matrix Z. Typically the global
///     matrix Z is n-by-n, distributed in 1D block row [cyclic] fashion.
///     To compute eigenvectors of the tridiagonal matrix, the global Z
///     should contain the identity matrix.
///     To compute eigenvectors of the original Hermitian matrix, the
///     global Z should contain the unitary matrix used in the reduction
///     to tridiagonal form.
///     On exit, Z is updated by the transformations reducing the
///     tridiagonal matrix to diagonal.
///     Not referenced if nrows = 0.
///
/// @param[in] ldz
///     The leading dimension of the array Z. ldz >= max( 1, nrows ).
///
/// @param[in] nrows
///     The local number of rows of the matrix Z to be updated. nrows >= 0.
///     For distributed use, this is typically
///     nrows = num_local_rows_cols( n, nb, myrow, 0, mpi_size );
///
/// @param[out] work
///     workspace, real array, dimension (max( 1, lwork ))
///     Not referenced if nrows = 0.
///
/// @param[in] lwork
///     The dimension of the array work.
///     If nrows > 0, lwork >= 2*n - 2.
///
///     If lwork = -1, then a workspace query is assumed; the routine
///     only calculates the optimal size of the work array, returns
///     this value as the first entry of the work array, and no error
///     message related to lwork is issued.
///
/// @param[out] info
///     = 0:  successful exit
///     < 0:  if info = -i, the i-th argument had an illegal value
///     > 0:  the algorithm has failed to find all the eigenvalues in
///           a total of 30*n iterations; if info = i, then i
///           elements of E have not converged to zero; on exit, D
///           and E contain the elements of a symmetric tridiagonal
///           matrix which is unitarily similar to the original
///           matrix.
///
template <typename scalar_t>
int64_t steqr(
    int64_t n,
    blas::real_type<scalar_t>* D,
    blas::real_type<scalar_t>* E,
    scalar_t* Z, int64_t ldz,
    int64_t nrows,
    blas::real_type<scalar_t>* work, int64_t lwork )
{
    using real_t = blas::real_type<scalar_t>;
    using std::abs, std::sqrt;
    using lapack::MatrixType, lapack::Norm, lapack::Side, lapack::Pivot,
          lapack::Direction;

    // Constants
    // Determine the unit roundoff and over/underflow thresholds.
    // LAPACK "eps" is unit roundoff, which is 0.5 epsilon from C++.
    const real_t eps = 0.5 * std::numeric_limits<real_t>::epsilon();
    const real_t eps2 = sqr( eps );
    const real_t safe_min = std::numeric_limits<real_t>::min();
    const real_t safe_max = 1.0 / safe_min;
    const real_t sqrt_safe_max = sqrt( safe_max ) / 3.0;
    const real_t sqrt_safe_min = sqrt( safe_min ) / eps2;

    // Local variables
    real_t b=0, c, f, g, p, r, s, rt1, rt2, tst;
    int64_t k;

    int64_t ilast = 0;
    int64_t info = 0;

    #if LOOKAHEAD
        real_t gp, rp, old_ej, old_gp, old_rp;
        int64_t nlook;
    #else
        SLATE_UNUSED( ilast );  // set but not used.
    #endif

    // Test the input parameters.
    bool lquery = (lwork == -1);
    if (n < 0) {
        info = -1;
    }
    else if (ldz < blas::max( 1, nrows )) {
        info = -5;
    }
    else if (nrows < 0) {
        info = -6;
    }
    else if (! lquery && nrows > 0 && lwork < blas::max( 1, 2*n - 2 )) {
        info = -8;
    }

    if (info != 0) {
        return info;
    }
    else if (lquery) {
        work[ 0 ] = nrows == 0 ? 1 : blas::max( 1, 2*n - 2 );
        return info;
    }

    // Quick return if possible
    if (n <= 1) {
        return info;
    }

    // If eigenvectors are not desired, sterf is faster.
    if (nrows == 0) {
        return lapack::sterf( n, D, E );
    }

    // Compute the eigenvalues and eigenvectors of the tridiagonal matrix.
    int64_t n_max_iter = n * 30;
    int64_t iters = 0;

    // Determine where the matrix splits and choose QL or QR iteration
    // for each block, according to whether the top or bottom diagonal
    // element is smaller.
    int64_t j1 = 0;
    while (j1 < n) {
        if (j1 > 0) {
            E[ j1 - 1 ] = 0.0;
        }
        for (k = j1; k < n - 1; ++k) {
            tst = abs( E[ k ] );
            if (tst <= sqrt( abs( D[ k ] ) ) * sqrt( abs( D[ k+1 ] ) ) * eps) {
                E[ k ] = 0.0;
                break;
            }
        }

        // Block is [ j, jend ], inclusive.
        // This is less confusing than the usual C++ half-exclusive
        // range [ j, jend ) because QR swaps j and jend.
        int64_t j = j1;
        int64_t j_save = j;
        int64_t jend = k;
        int64_t jend_save = jend;
        j1 = k + 1;

        // If block j:jend is 1x1, then found eigenvalue already.
        if (jend == j) {
            continue;
        }

        // Scale submatrix in rows and columns j to jend.
        real_t Anorm = lapack::lanst( Norm::Max, jend - j + 1, &D[ j ], &E[ j ] );
        if (Anorm == 0.0) {
            continue;
        }
        int iscale = 0;
        if (Anorm > sqrt_safe_max) {
            // Scale by sqrt_safe_max / Anorm.
            iscale = 1;
            info = lapack::lascl(
                MatrixType::General, 0, 0, Anorm, sqrt_safe_max,
                jend - j + 1, 1, &D[ j ], n );
            assert( info == 0 );
            info = lapack::lascl(
                MatrixType::General, 0, 0, Anorm, sqrt_safe_max,
                jend - j, 1, &E[ j ], n );
            assert( info == 0 );
        }
        else if (Anorm < sqrt_safe_min) {
            // Scale by sqrt_safe_min / Anorm.
            iscale = 2;
            info = lapack::lascl(
                MatrixType::General, 0, 0, Anorm, sqrt_safe_min,
                jend - j + 1, 1, &D[ j ], n );
            assert( info == 0 );
            info = lapack::lascl(
                MatrixType::General, 0, 0, Anorm, sqrt_safe_min,
                jend - j, 1, &E[ j ], n );
            assert( info == 0 );
        }

        // Choose between QL and QR iteration.
        // QL iterates k from top to bottom.
        // QR swaps j, jend and iterates k from bottom to top.
        if (abs( D[ jend ] ) < abs( D[ j ] )) {
            jend = j_save;
            j = jend_save;
        }

        if (j < jend) {
            // QL Iteration
            while (j <= jend) {
                // Look for small subdiagonal element.
                for (k = j; k < jend; ++k) {
                    tst = sqr( abs( E[ k ] ) );
                    if (tst <= eps2 * abs( D[ k ] ) * abs( D[ k+1 ] ) + safe_min) {
                        E[ k ] = 0.0;
                        break;
                    }
                }

                // If block j:k is 1-by-1, D[ j ] is an eigenvalue.
                if (k == j) {
                    j += 1;
                    continue;
                }

                // If block j:k is 2-by-2, use lae2 or laev2
                // to compute its eigensystem.
                // work[ 0:n-2       ] inclusive stores c values,
                // work[ 0:n-2 + n-1 ] inclusive stores s values.
                if (k == j + 1) {
                    if (nrows > 0) {
                        lapack::laev2( D[ j ], E[ j ], D[ j+1 ],
                                       &rt1, &rt2, &c, &s );
                        work[ j ] = c;
                        work[ j + n - 1 ] = s;
                        lapack::lasr(
                            Side::Right, Pivot::Variable, Direction::Backward,
                            nrows, 2, &work[ j ], &work[ j + n - 1 ],
                            &Z[ 0 + j*ldz ], ldz );
                    }
                    else {
                        lapack::lae2( D[ j ], E[ j ], D[ j+1 ], &rt1, &rt2 );
                    }
                    D[ j   ] = rt1;
                    D[ j+1 ] = rt2;
                    E[ j   ] = 0.0;
                    j += 2;
                    continue;
                }

                if (iters == n_max_iter) {
                    break;
                }
                ++iters;

                // Form Wilkinson shift based on j:j+1 block.
                p = D[ j ];
                g = (D[ j+1 ] - p) / (2.0 * E[ j ]);
                r = std::hypot( g, 1.0 );
                g = D[ k ] - p + E[ j ] / (g + copysign( r, g ));

#if LOOKAHEAD
                if (nrows > 0) {
                    // Do lookahead
                    old_ej = abs( E[ j ] );
                    gp = g;
                    rp = r;
                    tst = sqr( abs( E[ j ] ) );
                    tst /= eps2 * abs( D[ j ] ) * abs( D[ j+1 ] ) + safe_min;

                    nlook = 1;
                    while (tst > 1.0 && nlook <= 15) {
                        // This is the lookahead loop, going until we have
                        // convergence or too many steps have been taken.
                        s = 1.0;
                        c = 1.0;
                        p = 0.0;

                        // Inner loop
                        for (int64_t i = k - 1; i >= j; --i) {
                            f = s * E[ i ];
                            b = c * E[ i ];
                            lapack::lartg( gp, f, &c, &s, &rp );
                            gp = D[ i+1 ] - p;
                            rp = (D[ i ] - gp) * s + 2.0 * c * b;
                            p = s * rp;
                            if (i != j) {
                                gp = c * rp - b;
                            }
                        }
                        old_gp = gp;
                        old_rp = rp;
                        // Find GP & RP for the next iteration
                        if (abs( c * old_rp - b ) > safe_min) {
                            gp = (old_gp + p - (D[ j ] - p))
                               / (2.0*(c * old_rp - b));
                        }
                        else {
                            // Here it seems usually, c*old_rp - b == 0.0
                            // Goto put in by G. Henry to fix ALPHA problem
                            goto QL_lookahead_end;
                            //gp = (old_gp + p - (D[ j ] - p))
                            //   / (2.0*(c * old_rp - b) + safe_min);
                        }
                        rp = std::hypot( gp, 1.0 );
                        gp = D[ k ] - (D[ j ] - p) + (c * old_rp - b)
                           / (gp + copysign( rp, gp ));
                        tst = sqr( abs( c * old_rp - b ) );
                        tst /= eps2 * abs( D[ j ] - p ) * abs( old_gp + p )
                                + safe_min;
                        // Make sure that we are making progress
                        if (abs( c * old_rp - b ) > old_ej * 0.9) {
                            if (abs( c * old_rp - b ) > old_ej) {
                                gp = g;
                                rp = r;
                            }
                            tst = 0.5;
                        }
                        else {
                            old_ej = abs( c * old_rp - b );
                        }
                        ++nlook;
                    }
                    if (tst <= 1.0
                        && tst != 0.5
                        && abs( p ) < eps * abs( D[ j ] )
                        && ilast == j + 1
                        && sqr( abs( E[ j ] ) )
                           <= (eps2 * abs( D[ j ] ) * abs( D[ j+1 ] ) + safe_min) * 1e4) {
                        // Eigenvalue found.
                        // Skip the current step: the subdiagonal info is just noise.
                        k = j;
                        E[ k ] = 0.0;
                        p = D[ j ];
                        --iters;
                        ++j;
                        continue;
                    }
                    g = gp;
                    r = rp;
                }
            QL_lookahead_end:
#endif // lookahead

                s = 1.0;
                c = 1.0;
                p = 0.0;

                // Inner loop
                for (int64_t i = k - 1; i >= j; --i) {
                    f = s * E[ i ];
                    b = c * E[ i ];
                    lapack::lartg( g, f, &c, &s, &r );
                    if (i != k - 1) {
                        E[ i+1 ] = r;
                    }
                    g = D[ i+1 ] - p;
                    r = (D[ i ] - g) * s + 2.0 * c * b;
                    p = s * r;
                    D[ i+1 ] = g + p;
                    g = c * r - b;

                    // If eigenvectors are desired, then save rotations.
                    if (nrows > 0) {
                        work[ i ] = c;
                        work[ i + n - 1 ] = -s;
                    }
                }

                // If eigenvectors are desired, then apply saved rotations.
                if (nrows > 0) {
                    lapack::lasr(
                        Side::Right, Pivot::Variable, Direction::Backward,
                        nrows, k - j + 1, &work[ j ], &work[ j + n - 1 ],
                        &Z[ 0 + j*ldz ], ldz );
                }

                D[ j ] -= p;
                E[ j ] = g;
                ilast = j + 1;
            }
        }
        else {
            // QR Iteration
            while (j >= jend) {
                // Look for small superdiagonal element.
                for (k = j; k > jend; --k) {
                    tst = sqr( abs( E[ k-1 ] ) );
                    if (tst <= eps2 * abs( D[ k ] ) * abs( D[ k-1 ] ) + safe_min) {
                        E[ k-1 ] = 0.0;
                        break;
                    }
                }

                // If block j:k is 1-by-1, D[ j ] is an eigenvalue.
                if (k == j) {
                    j -= 1;
                    continue;
                }

                // If block k:j is 2-by-2, use lae2 or laev2
                // to compute its eigensystem.
                if (k == j - 1) {
                    if (nrows > 0) {
                        lapack::laev2( D[ j-1 ], E[ j-1 ], D[ j ],
                                       &rt1, &rt2, &c, &s );
                        work[ k ] = c;
                        work[ k + n - 1 ] = s;
                        lapack::lasr(
                            Side::Right, Pivot::Variable, Direction::Forward,
                            nrows, 2, &work[ k ], &work[ k + n - 1 ],
                            &Z[ 0 + (j-1)*ldz ], ldz );
                    }
                    else {
                        lapack::lae2( D[ j-1 ], E[ j-1 ], D[ j ], &rt1, &rt2 );
                    }
                    D[ j-1 ] = rt1;
                    D[ j   ] = rt2;
                    E[ j-1 ] = 0;
                    j -= 2;
                    continue;
                }

                if (iters == n_max_iter) {
                    break;
                }
                ++iters;

                // Form Wilkinson shift based on j-1:j block.
                p = D[ j ];
                g = (D[ j-1 ] - p) / (2.0*E[ j-1 ]);
                r = std::hypot( g, 1.0 );
                g = D[ k ] - p + E[ j-1 ] / (g + copysign( r, g ));

#if LOOKAHEAD
                if (nrows > 0) {
                    // Do lookahead
                    old_ej = abs( E[ j-1 ] );
                    gp = g;
                    rp = r;
                    tst = sqr( abs( E[ j-1 ] ) );
                    tst /= eps2 * abs( D[ j ] ) * abs( D[ j-1 ] ) + safe_min;

                    nlook = 1;
                    while (tst > 1.0 && nlook <= 15) {
                        // This is the lookahead loop, going until we have
                        // convergence or too many steps have been taken.
                        s = 1.0;
                        c = 1.0;
                        p = 0.0;

                        // Inner loop
                        for (int64_t i = k; i < j; ++i) {
                            f = s * E[ i ];
                            b = c * E[ i ];
                            lapack::lartg( gp, f, &c, &s, &rp );
                            gp = D[ i ] - p;
                            rp = (D[ i+1 ] - gp) * s + c * 2.0 * b;
                            p = s * rp;
                            if (i < j-1) {
                                gp = c * rp - b;
                            }
                        }
                        old_gp = gp;
                        old_rp = rp;
                        // Find GP & RP for the next iteration
                        if (abs( c * old_rp - b ) > safe_min) {
                            gp = (old_gp + p - (D[ j ] - p))
                               / (2.0*(c * old_rp - b));
                        }
                        else {
                            // Here it seems usually, c*old_rp - b == 0.0
                            // Goto put in by G. Henry to fix ALPHA problem
                            goto QR_lookahead_end;
                            //gp = (old_gp + p - (D[ j ] - p))
                            //   / (2.0*(c * old_rp - b) + safe_min);
                        }
                        rp = std::hypot( gp, 1.0 );
                        gp = D[ k ] - (D[ j ] - p) + (c * old_rp - b)
                           / (gp + copysign( rp, gp ));
                        tst = sqr( abs( c * old_rp - b ) );
                        tst /= eps2 * abs( D[ j ] - p ) * abs( old_gp + p )
                                + safe_min;
                        // Make sure that we are making progress
                        if (abs( c * old_rp - b ) > old_ej * 0.9) {
                            if (abs( c * old_rp - b ) > old_ej) {
                                gp = g;
                                rp = r;
                            }
                            tst = 0.5;
                        }
                        else {
                            old_ej = abs( c * old_rp - b );
                        }
                        ++nlook;
                    }
                    if (tst <= 1.0
                        && tst != 0.5
                        && abs( p ) < eps * abs( D[ j ] )
                        && ilast == j + 1
                        && sqr( abs( E[ j-1 ] ) )
                           <= (eps2 * abs( D[ j-1 ] ) * abs( D[ j ] ) + safe_min) * 1e4) {
                        // Eigenvalue found.
                        // Skip the current step: the subdiagonal info is just noise.
                        k = j;
                        E[ k-1 ] = 0.0;
                        p = D[ j ];
                        --iters;
                        --j;
                        continue;
                    }
                    g = gp;
                    r = rp;
                }
            QR_lookahead_end:
#endif // lookahead

                s = 1.0;
                c = 1.0;
                p = 0.0;

                // Inner loop
                for (int64_t i = k; i < j; ++i) {
                    f = s * E[ i ];
                    b = c * E[ i ];
                    lapack::lartg( g, f, &c, &s, &r );
                    if (i != k) {
                        E[ i-1 ] = r;
                    }
                    g = D[ i ] - p;
                    r = (D[ i+1 ] - g) * s + 2.0 * c * b;
                    p = s * r;
                    D[ i ] = g + p;
                    g = c * r - b;

                    // If eigenvectors are desired, then save rotations.
                    if (nrows > 0) {
                        work[ i ] = c;
                        work[ i + n - 1 ] = s;
                    }
                }

                // If eigenvectors are desired, then apply saved rotations.
                if (nrows > 0) {
                    lapack::lasr(
                        Side::Right, Pivot::Variable, Direction::Forward,
                        nrows, j - k + 1, &work[ k ], &work[ k + n - 1 ],
                        &Z[ 0 + k*ldz ], ldz );
                }

                D[ j ] -= p;
                E[ j-1 ] = g;
                ilast = j + 1;
            }
        }

        // Undo scaling if necessary.
        if (iscale == 1) {
            // Scale by Anorm / sqrt_safe_max.
            info = lapack::lascl(
                MatrixType::General, 0, 0, sqrt_safe_max, Anorm,
                jend_save - j_save + 1, 1, &D[ j_save ], n );
            assert( info == 0 );
            info = lapack::lascl(
                MatrixType::General, 0, 0, sqrt_safe_max, Anorm,
                jend_save - j_save, 1, &E[ j_save ], n );
            assert( info == 0 );
        }
        else if (iscale == 2) {
            // Scale by Anorm / sqrt_safe_min.
            info = lapack::lascl(
                MatrixType::General, 0, 0, sqrt_safe_min, Anorm,
                jend_save - j_save + 1, 1, &D[ j_save ], n );
            assert( info == 0 );
            info = lapack::lascl(
                MatrixType::General, 0, 0, sqrt_safe_min, Anorm,
                jend_save - j_save, 1, &E[ j_save ], n );
            assert( info == 0 );
        }

        // Check for no convergence to an eigenvalue after a total
        // of n*max_iter iterations.
        if (iters >= n_max_iter) {
            for (int64_t i = 0; i < n - 1; ++i) {
                if (E[ i ] != 0.0) {
                    ++info;
                }
            }
            return info;
        }
    }

    // Order eigenvalues and eigenvectors.
    if (nrows == 0) {
        std::sort( D, &D[ n ] );
    }
    else {
        // Use Selection Sort to minimize swaps of eigenvectors.
        for (int64_t i = 0; i < n - 1; ++i) {
            k = i;
            p = D[ i ];
            for (int64_t j = i + 1; j < n; ++j) {
                if (D[ j ] < p) {
                    k = j;
                    p = D[ j ];
                }
            }
            if (k != i) {
                D[ k ] = D[ i ];
                D[ i ] = p;
                blas::swap( nrows, &Z[ 0 + i*ldz ], 1, &Z[ 0 + k*ldz ], 1 );
            }
        }
    }

    return info;
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t steqr(
    int64_t n,
    float* D, float* E,
    float* Z, int64_t ldz,
    int64_t nrows,
    float* work, int64_t lwork );

template
int64_t steqr(
    int64_t n,
    double* D, double* E,
    double* Z, int64_t ldz,
    int64_t nrows,
    double* work, int64_t lwork );

template
int64_t steqr(
    int64_t n,
    float* D, float* E,
    std::complex<float>* Z, int64_t ldz,
    int64_t nrows,
    float* work, int64_t lwork );

template
int64_t steqr(
    int64_t n,
    double* D, double* E,
    std::complex<double>* Z, int64_t ldz,
    int64_t nrows,
    double* work, int64_t lwork );

} // namespace slate
