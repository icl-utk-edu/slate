// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"

#include "scalapack_support_routines.hh"
#include "band_utils.hh"
#include "grid_utils.hh"

// Internal data structure used for ct_count.
#include "../src/internal/Array2D.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <numeric>

//------------------------------------------------------------------------------
inline void warn_if_( bool cond, const char* cond_msg )
{
    if (cond) {
        printf( "warning: %s\n", cond_msg );
    }
}

#define warn_if( cond ) warn_if_( cond, #cond )

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_stedc_secular_work( Params& params, bool run )
{
    using real_t = blas::real_type<scalar_t>;
    using std::real, std::imag;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    const scalar_t nan_ = nan("");
    const real_t   eps  = std::numeric_limits<real_t>::epsilon();
    const real_t   tol  = params.tol() * eps/2;

    // get & mark input values
    int64_t nsecular = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    //bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    scalar_t rho = std::abs( params.alpha.get<scalar_t>() );

    // mark non-standard output values
    params.time();
    params.ref_time();

    params.error();
    params.error2();
    params.ortho();
    params.error.name( "Lambda err" );
    params.error2.name( "U err" );
    params.ortho.name( "U ortho" );

    if (! run)
        return;

    slate::Options const opts =  {
        // {slate::Option::Target, target}
        {slate::Option::PrintVerbose, verbose},
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    gridinfo( mpi_rank, p, q, &myrow, &mycol );

    // Matrix U: figure out local size.
    int64_t mlocU = num_local_rows_cols( n, nb, myrow, p );
    int64_t nlocU = num_local_rows_cols( n, nb, mycol, q );
    int64_t lldU  = blas::max( 1, mlocU ); // local leading dimension of U

    std::vector<scalar_t> U_data;
    slate::Matrix<scalar_t> U;
    if (origin != slate::Origin::ScaLAPACK) {
        U = slate::Matrix<scalar_t>(
                n, n, nb, p, q, MPI_COMM_WORLD);
        U.insertLocalTiles( origin2target( origin ) );
    }
    else {
        U_data.resize( lldU*nlocU );
        U = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &U_data[0], lldU, nb, p, q, MPI_COMM_WORLD );
    }
    slate::set( nan_, U, opts );

    std::vector<scalar_t> D( nsecular );
    std::vector<scalar_t> z( nsecular );
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( 3, iseed, nsecular, &D[ 0 ] );  // idist 3: normal
    lapack::larnv( 1, iseed, nsecular, &z[ 0 ] );  // idist 1: uniform [0, 1]
    std::sort( &D[ 0 ], &D[ nsecular ] );
    scalar_t z_norm = blas::nrm2( nsecular, &z[ 0 ], 1 );
    blas::scal( nsecular, 1/z_norm, &z[ 0 ], 1 );

    // Save copy in ScaLAPACK format for reference.
    std::vector<scalar_t> Uref_data, Lambda_ref;
    slate::Matrix<scalar_t> Uref;
    if (ref) {
        Lambda_ref.resize( n );
        Uref_data.resize( lldU*nlocU );
        Uref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Uref_data[0], lldU, nb, p, q, MPI_COMM_WORLD );
    }

    // Set itype to identity permutation, 0..n-1.
    std::vector<int64_t> itype( n );
    std::iota( &itype[0], &itype[n], 0 );

    std::vector<scalar_t> Lambda( n, -1 );

    // ct_count( q-by-5 ) = 0.
    int64_t izero = 0;
    slate::internal::Array2D<int64_t> ct_count( q, 5, izero );
    int64_t nt  = U.nt();
    int64_t nt1 = int64_t( nt/2 );
    //int64_t nt2 = nt - nt1;

    int pcol = 0;
    for (int j = 0; j < nt1; ++j) {
        ct_count( pcol, 1 ) += U.tileNb( j );
        pcol = (pcol + 1) % q;
    }
    for (int j = nt1; j < nt; ++j) {
        ct_count( pcol, 3 ) += U.tileNb( j );
        pcol = (pcol + 1) % q;
    }

    if (verbose >= 1 && mpi_rank == 0) {
        printf( "-------------------- SLATE input\n" );
        printf( "rho = %7.4f\n", rho );
        print_vector( "D", D, params );
        print_vector( "z", z, params );

        printf( "ct_count( q=%d, 5 ) = [\n", q );
        for (int i = 0; i < q; ++i) {  // loop over process cols.
            for (int j = 0; j < 5; ++j) {
                printf( " %lld", llong( ct_count( i, j ) ) );
            }
            printf( "\n" );
        }
        printf( "]\n" );
    }

    //std::vector<scalar_t> ztilde( n, -1 );
    //std::vector<int> prows( n ), pcols( n );

    if (trace)
        slate::trace::Trace::on();

    double time = barrier_get_wtime( MPI_COMM_WORLD );

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::stedc_secular( nsecular, n, rho,
                          &D[0], &z[0], &Lambda[0],
                          U, &itype[0], opts );

    params.time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

    if (trace)
        slate::trace::Trace::finish();

    // SLATE output is repeated in ref for easy comparison.
    if (! ref) {
        if (verbose >= 1 && mpi_rank == 0) {
            printf( "-------------------- SLATE output\n" );
            print_vector( "Lambda", Lambda, params );
        }
        print_matrix( "U", U, params );
        MPI_Barrier( MPI_COMM_WORLD );
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_;

            // initialize BLACS and ScaLAPACK
            Cblacs_get( -1, 0, &ictxt );
            Cblacs_gridinit( &ictxt, "Col", p, q );
            Cblacs_gridinfo( ictxt, &p_, &q_, &myrow_, &mycol_ );

            std::vector<int> itype_ref( n, -1 ),
                             pcols_ref( n, -1 ),
                             prows_ref( n, -1 ),
                             irow( n, -1 ),
                             icol( n, -1 ),
                             ct_count_ref( q*4, -1 );
            std::vector<scalar_t> buf( 3*n, -1 ),
                                  ztilde_ref( n, -1 );
            int info = 0;

            // Copy & convert idx from 0-based to 1-based.
            for (int64_t j = 0; j < n; ++j) {
                itype_ref[ j ] = itype[ j ] + 1;
            }

            // Copy ct_count_ref( :, 0:3 ) = ct_count( :, 1:4 ).
            int64_t ldc = q;
            for (int64_t i = 0; i < q; ++i) {
                for (int64_t j = 1; j <= 4; ++j) {
                    // In ct_count_ref, i is 0-based, j is 1-based.
                    ct_count_ref[ i  + (j-1)*ldc ] = ct_count( i, j );
                }
            }

            if (verbose >= 1 && mpi_rank == 0) {
                printf( "-------------------- ScaLAPACK input\n" );
                printf( "itype_ref = [ " );
                for (int64_t j = 0; j < n; ++j) {
                    printf( " %d", itype_ref[ j ] );
                }
                printf( " ];\n" );

                printf( "ct_count_ref = [ " );
                for (int64_t j = 0; j < n; ++j) {
                    printf( " %d", itype_ref[ j ] );
                }
                printf( " ];\n" );
            }

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime( MPI_COMM_WORLD );

            scalapack_plaed3(
                ictxt, nsecular, n, nb,
                &Lambda_ref[0], 0, 0, rho, &D[0], &z[0], &ztilde_ref[0],
                &Uref_data[0], lldU, &buf[0],
                &itype_ref[0],
                &pcols_ref[0], &prows_ref[0], &irow[0], &icol[0],
                &ct_count_ref[0], q, &info );
            assert( info == 0 );

            params.ref_time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            if (verbose >= 1 && mpi_rank == 0) {
                print_vector( "D      ", D, params );
                print_vector( "z      ", z, params );
                //print_vector( "ztilde ", ztilde,     params );
                print_vector( "ztilde*", ztilde_ref, params );
                print_vector( "Lambda ", Lambda,     params );
                print_vector( "Lambda*", Lambda_ref, params );
                //print_vector( "prows  ", prows,      params );
                //print_vector( "prows* ", prows_ref,  params );
                //print_vector( "pcols  ", pcols,      params );
                //print_vector( "pcols* ", pcols_ref,  params );

                scalar_t error = 0;
                //blas::axpy( n, -1.0, &ztilde[0], 1, &ztilde_ref[0], 1 );
                blas::axpy( n, -1.0, &Lambda[0], 1, &Lambda_ref[0], 1 );
                //error += blas::nrm2( n, &ztilde_ref[0], 1 );
                error += blas::nrm2( n, &Lambda_ref[0], 1 );
                //for (int i = 0; i < n; ++i) {
                //    //prows_ref[ i ] -= prows[ i ];
                //    //pcols_ref[ i ] -= pcols[ i ];
                //    error += prows_ref[ i ] - prows[ i ];
                //    error += pcols_ref[ i ] - pcols[ i ];
                //}
                printf( "error %.2e\n", error );
            }
            print_matrix( "U ", U,    params );
            print_matrix( "U*", Uref, params );

            // error = || Lambda - Lambda_ref ||_2 / || Lambda_ref ||_2
            blas::axpy( n, -one, &Lambda_ref[0], 1, &Lambda[0], 1 );
            params.error() = blas::nrm2( n, &Lambda[0], 1 )
                             / blas::nrm2( n, &Lambda_ref[0], 1 );

            // orthogonality error = || U^H U - I || / n
            slate::Matrix<scalar_t> R( n, n, nb, p, q, MPI_COMM_WORLD );
            R.insertLocalTiles();
            slate::set( zero, one, R, opts );  // R = Identity
            auto UH = conj_transpose( U );
            slate::multiply( one, UH, U, -one, R, opts );
            params.ortho() = slate::norm( slate::Norm::One, R, opts ) / n;

            // rel forward error = || U - U_ref ||_1 / || U_ref ||_1
            slate::add( -one, Uref, one, U, opts );
            params.error2() = slate::norm( slate::Norm::One, U,    opts )
                            / slate::norm( slate::Norm::One, Uref, opts );

            params.okay() = (params.error() <= tol)
                            && (params.error2() <= tol)
                            && (params.ortho() <= tol);
        #endif
    }
}

//------------------------------------------------------------------------------
void test_stedc_secular( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            //test_stedc_secular_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_stedc_secular_work<double> (params, run);
            break;

        default:
            throw std::exception();
            break;
    }
}
