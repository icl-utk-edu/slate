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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#define SLATE_HAVE_SCALAPACK

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
void test_stedc_deflate_work( Params& params, bool run )
{
    using real_t = blas::real_type<scalar_t>;
    using std::real, std::imag;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    const real_t   eps  = std::numeric_limits<real_t>::epsilon();
    const real_t   tol  = params.tol() * eps/2;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    //bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    scalar_t rho = params.alpha.get<scalar_t>();
    std::string deflate = params.deflate();

    // mark non-standard output values
    params.time();
    params.ref_time();

    params.error();
    params.error2();
    params.error3();
    params.error4();
    params.error5();
    params.error .name( "D err" );
    params.error2.name( "z err" );
    params.error3.name( "Q err" );
    params.error4.name( "Qbar err" );
    params.error5.name( "idx err" );

    if (! run)
        return;

    if (n <= nb) {
        params.msg() = "skipping: requires n > nb (i.e., multiple tiles)";
        return;
    }

    slate::Options const opts =  {
        // {slate::Option::Target, target}
        {slate::Option::PrintVerbose, verbose},
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    gridinfo( mpi_rank, p, q, &myrow, &mycol );

    // Matrix Q: figure out local size.
    int64_t mlocQ = num_local_rows_cols( n, nb, myrow, p );
    int64_t nlocQ = num_local_rows_cols( n, nb, mycol, q );
    int64_t lldQ  = blas::max( 1, mlocQ ); // local leading dimension of Q

    std::vector<scalar_t> Q_data, Qbar_data;
    slate::Matrix<scalar_t> Q, Qbar;
    if (origin != slate::Origin::ScaLAPACK) {
        Q = slate::Matrix<scalar_t>(
                n, n, nb, p, q, MPI_COMM_WORLD);
        Q.insertLocalTiles( origin2target( origin ) );
        Qbar = slate::Matrix<scalar_t>(
                   n, n, nb, p, q, MPI_COMM_WORLD);
        Qbar.insertLocalTiles( origin2target( origin ) );
    }
    else {
        Q_data.resize( lldQ*nlocQ );
        Qbar_data.resize( lldQ*nlocQ );
        Q    = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Q_data[0],    lldQ, nb, p, q, MPI_COMM_WORLD );
        Qbar = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Qbar_data[0], lldQ, nb, p, q, MPI_COMM_WORLD );
    }
    slate::set( zero, Q );
    int64_t nt = Q.nt();
    int64_t nt1 = int64_t( nt/2 );
    int64_t nt2 = nt - nt1;
    auto Q1 = Q.sub( 0,   nt1 - 1, 0,   nt1 - 1 );
    auto Q2 = Q.sub( nt1, nt  - 1, nt1, nt  - 1 );
    slate::generate_matrix( params.matrix, Q1 );
    slate::generate_matrix( params.matrix, Q2 );
    if (verbose >= 1 && mpi_rank == 0)
        printf( "nt %lld, nt1 %lld, nt2 %lld\n",
                llong( nt ), llong( nt1 ), llong( nt2 ) );

    std::vector<scalar_t> D( n ), Dsecular( n, nan("") );
    std::vector<scalar_t> z( n ), zsecular( n, nan("") );
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( 3, iseed, n, &D[0] );  // idist 3: normal
    lapack::larnv( 1, iseed, n, &z[0] );  // idist 1: uniform [0, 1]

    //--------------------------------------------------------------------------
    // D1 is odds  + 0.1, z1 is odds  + 0.2
    // D2 is evens + 0.1, z2 is evens + 0.2
    int64_t n1 = nt1 * nb;
    int64_t n2 = n - n1;
    for (int i = 0; i < n1; ++i) {
        D[ i ] = 1.1 + 2*i;
        z[ i ] = 1.2 + 2*i;
    }
    for (int i = n1; i < n; ++i) {
        D[ i ] = 2.1 + 2*(i - n1);
        z[ i ] = 2.2 + 2*(i - n1);
    }
    // Q1( i, : ) = z1
    for (int j = 0; j < nt1; ++j) {
        for (int i = 0; i < nt1; ++i) {
            if (Q1.tileIsLocal( i, j )) {
                auto T = Q1( i, j );
                int mb_ = T.mb();
                int nb_ = T.nb();
                for (int j_ = 0; j_ < nb_; ++j_)
                    for (int i_ = 0; i_ < mb_; ++i_)
                        T.at( i_, j_ ) = z[ j*nb + j_ ];
            }
        }
    }
    // Q2( i, : ) = z2
    for (int j = 0; j < nt2; ++j) {
        for (int i = 0; i < nt2; ++i) {
            if (Q2.tileIsLocal( i, j )) {
                auto T = Q2( i, j );
                int mb_ = T.mb();
                int nb_ = T.nb();
                for (int j_ = 0; j_ < nb_; ++j_)
                    for (int i_ = 0; i_ < mb_; ++i_)
                        T.at( i_, j_ ) = z[ (j + nt1)*nb + j_ ];
            }
        }
    }
    //--------------------------------------------------------------------------
    // Parse deflations.
    std::vector<int64_t> deflate1;
    std::vector< std::pair<int64_t, int64_t> > deflate2;
    const char* deflate_ = deflate.c_str();
    while (*deflate_ != '\0') {
        int j1, j2, bytes1, bytes2;
        int cnt = sscanf( deflate_, "%d %n , %d %n",
                          &j1, &bytes1, &j2, &bytes2 );
        //printf( "cnt %d, j1 %2d (%2d), j2 %2d (%2d), deflate_ '%s'\n",
        //        cnt, j1, j1-1, j2, j2-1, deflate_ );
        if (cnt == 1) {
            deflate1.push_back( j1 );
            deflate_ += bytes1;
        }
        else if (cnt == 2) {
            deflate2.push_back( { j1, j2 } );
            deflate_ += bytes2;
        }
        else {
            throw slate::Exception(
                std::string("unrecognized --deflate: '") + deflate_ + "'" );
        }
    }

    // Apply type 1 deflation, z[j] ~ 0, within +- 10 eps.
    for (int j : deflate1) {
        slate_assert( 0 <= j && j < n );
        z[ j ] = (rand() % 20 - 10)*eps;
    }

    // Apply type 2 deflation, D[j2] ~ D[j1], within +- 10 eps.
    for (auto j12 : deflate2) {
        int64_t j1 = j12.first;
        int64_t j2 = j12.second;
        slate_assert( 0 <= j1 && j1 < n );
        slate_assert( 0 <= j2 && j2 < n );
        D[ j2 ] = D[ j1 ] * (1 + (rand() % 20 - 10)*eps);
    }
    //--------------------------------------------------------------------------

    std::vector<scalar_t> Dorig( D );
    std::vector<scalar_t> zorig( z );

    // Save copy in ScaLAPACK format for reference.
    std::vector<scalar_t> Qref_data, Qbar_ref_data, Dref, zref;
    slate::Matrix<scalar_t> Qref, Qbar_ref;
    scalar_t rho_ref = rho;
    if (ref) {
        Qref_data.resize( lldQ*nlocQ );
        Qbar_ref_data.resize( lldQ*nlocQ, nan("") );
        Qref     = slate::Matrix<scalar_t>::fromScaLAPACK(
                       n, n, &Qref_data[0],     lldQ, nb, p, q, MPI_COMM_WORLD );
        Qbar_ref = slate::Matrix<scalar_t>::fromScaLAPACK(
                       n, n, &Qbar_ref_data[0], lldQ, nb, p, q, MPI_COMM_WORLD );
        copy( Q, Qref );
        Dref = D;
        zref = z;
    }

    if (verbose >= 1 && mpi_rank == 0)
        printf( "-------------------- SLATE input\n" );
    print_matrix( "Q", Q, params );
    if (verbose >= 1 && mpi_rank == 0) {
        print_vector( "D", D );
        print_vector( "z", z );
    }

    std::vector<int64_t> ibar( n, -1 );
    int64_t nsecular = 0;
    int64_t nU123 = -1;
    int64_t Q12_begin = -1, Q12_end = -1;
    int64_t Q23_begin = -1, Q23_end = -1;

    //std::vector<int64_t> isort   ( n, -1 );
    //std::vector<int64_t> ideflate( n, -1 );
    //std::vector<int64_t> iglobal ( n, -1 );
    //std::vector<int> pcols       ( n, -1 );
    //std::vector<int> coltype     ( n, -1 );
    //slate::internal::Array2D<int64_t> ct_count( q, 5 );

    if (trace)
        slate::trace::Trace::on();

    double time = barrier_get_wtime( MPI_COMM_WORLD );

    //==================================================
    // Run SLATE test.
    //==================================================
    stedc_deflate( n, n1, rho,
                   &D[0], &Dsecular[0], &z[0], &zsecular[0],
                   Q, Qbar, &ibar[0],
                   nsecular, nU123,
                   Q12_begin, Q12_end, Q23_begin, Q23_end,
                   opts );

    params.time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

    if (trace)
        slate::trace::Trace::finish();

    // SLATE output is repeated in ref for easy comparison.
    if (! ref) {
        if (verbose >= 1 && mpi_rank == 0)
            printf( "-------------------- SLATE output\n" );
        print_matrix( "Q", Q, params );
        if (verbose >= 1 && mpi_rank == 0) {
            printf( "rank     = [ " );
            for (int j = 0; j < n; ++j) {
                printf( " %10d", Q.tileRank( 0, j/nb ) );
            }
            printf( " ]';\n" );
            print_vector( "Dorig   ", Dorig );
            print_vector( "zorig   ", zorig );
            print_vector( "Dout    ", D );
            print_vector( "zout    ", z );
            print_vector( "Dsecular", Dsecular );
            print_vector( "zsecular", zsecular );
            //print_vector( "ibar", ibar );

            //print_vector( "isort",    isort    );
            //print_vector( "iglobal",  iglobal  );
            //print_vector( "ideflate", ideflate );
            //print_vector( "pcols   ", pcols    );
            //print_vector( "coltype ", coltype  );
        }
        MPI_Barrier( MPI_COMM_WORLD );

        if (verbose >= 1 && mpi_rank == 0) {
            printf( "n %lld, n1 %lld, n2 %lld, nsecular %lld, nU123 %lld, "
                    "Q12 %lld : %lld (%lld), Q23 %lld : %lld (%lld)\n",
                    llong( n ), llong( n1 ), llong( n2 ),
                    llong( nsecular ), llong( nU123 ),
                    llong( Q12_begin ), llong( Q12_end ), llong( Q12_end - Q12_begin ),
                    llong( Q23_begin ), llong( Q23_end ), llong( Q23_end - Q23_begin ) );
        }
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads( omp_num_threads );

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_;

            // initialize BLACS and ScaLAPACK
            Cblacs_get( -1, 0, &ictxt );
            Cblacs_gridinit( &ictxt, "Col", p, q );
            Cblacs_gridinfo( ictxt, &p_, &q_, &myrow_, &mycol_ );

            if (verbose >= 1 && mpi_rank == 0)
                printf( "-------------------- ScaLAPACK input\n" );
            print_matrix( "Qref", Qref, params );
            if (verbose >= 1 && mpi_rank == 0) {
                print_vector( "Dref", Dref );
                print_vector( "zref", zref );
            }

            // q = npcol
            int nsecular_ref = -1, nU123_ref = -1,
                nQ12_ref = -1, Q12_begin_ref = -1,
                nQ23_ref = -1, Q23_begin_ref = -1;
            std::vector<scalar_t>
                Qbuf( 3*n, nan("") ),
                Dsecular_ref( n, nan("") ),
                zsecular_ref( n, nan("") );
            std::vector<int>
                ct_count_ref( q*4, -1 ),
                ct_idx_local( q*4, -1 ),
                ibar_ref    ( n, -1 ),
                iglobal_ref ( n, -1 ),
                ideflate_ref( n, -1 ),
                pcols_ref   ( n, -1 ),
                coltype_ref ( n, -1 );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime( MPI_COMM_WORLD );

            scalapack_plaed2(
                ictxt, &nsecular_ref, n, n1, nb,
                &Dref[0], 0, 0,
                &Qref_data[0], lldQ, &rho_ref,
                &zref[0], &zsecular_ref[0], &Dsecular_ref[0],
                &Qbar_ref_data[0], lldQ,
                &Qbuf[0],
                &ct_count_ref[0], &ct_idx_local[0], q,
                &ibar_ref[0], &iglobal_ref[0], &ideflate_ref[0],
                &pcols_ref[0], &coltype_ref[0], &nU123_ref,
                &nQ12_ref, &nQ23_ref, &Q12_begin_ref, &Q23_begin_ref );

            params.ref_time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            slate_set_num_blas_threads( saved_num_threads );

            // convert to 1-based
            for (int j = 0; j < n; ++j) {
                ibar_ref[j]     -= 1;
                iglobal_ref[j]  -= 1;
                ideflate_ref[j] -= 1;
            }
            // check errors in int arrays.
            int err_ibar     = 0;
            int err_iglobal  = 0;
            int err_ideflate = 0;
            int err_pcols    = 0;
            int err_coltype  = 0;
            for (int j = 0; j < n; ++j) {
                err_ibar     = std::abs( ibar[j]     - ibar_ref[j]     );
                //err_iglobal = std::abs( iglobal[j]  - iglobal_ref[j]  );
                //err_ideflate = std::abs( ideflate[j] - ideflate_ref[j] );
                //err_pcols    = std::abs( pcols[j]    - pcols_ref[j]    );
                //err_coltype  = std::abs( coltype[j]  - coltype_ref[j]  );
            }
            params.error5() = err_ibar + err_iglobal + err_ideflate + err_pcols
                              + err_coltype;

            if (verbose >= 1 && mpi_rank == 0)
                printf( "-------------------- ScaLAPACK output\n" );

            print_matrix( "Q ", Q,    params );
            print_matrix( "Q*", Qref, params );

            // Print col indices to help parse Q matrices.
            if (verbose >= 2 && mpi_rank == 0) {
                for (int j = 0; j < n; ++j) {
                    if (j > 0 && j % nb == 0)
                        printf( "    " );
                    printf( " %10d", j );
                }
                printf( "\n" );
            }

            print_matrix( "Qbar ", Qbar,     params );
            print_matrix( "Qbar*", Qbar_ref, params );
            if (verbose >= 1 && mpi_rank == 0) {
                printf( "index     = [ " );
                for (int j = 0; j < n; ++j) {
                    printf( " %10d", j );
                }
                printf( " ]';\n" );

                printf( "rank      = [ " );
                for (int j = 0; j < n; ++j) {
                    int jj = j/nb;
                    printf( " %10d", Q.tileRank( 0, jj ) );
                }
                printf( " ]';\n" );

                print_vector( "Dorig    ", Dorig );
                print_vector( "zorig    ", zorig );
                print_vector( "D        ", D    );
                print_vector( "D*       ", Dref );
                print_vector( "z        ", z    );
                print_vector( "z*       ", zref );
                print_vector( "Dsecular ", Dsecular     );
                print_vector( "Dsecular*", Dsecular_ref );
                print_vector( "zsecular ", zsecular     );
                print_vector( "zsecular*", zsecular_ref );

                //print_vector( "ibar ", ibar     );
                //print_vector( "ibar*", ibar_ref );
                //warn_if( err_ibar != 0 );

                //print_vector( "iglobal ", iglobal     );
                //print_vector( "iglobal*", iglobal_ref );
                //warn_if( err_iglobal != 0 );

                //print_vector( "ideflate ", ideflate     );
                //print_vector( "ideflate*", ideflate_ref );
                //warn_if( err_ideflate != 0 );

                //print_vector( "pcols    ", pcols     );
                //print_vector( "pcols*   ", pcols_ref );
                //warn_if( err_pcols != 0 );

                //print_vector( "coltype  ", coltype     );
                //print_vector( "coltype* ", coltype_ref );
                //warn_if( err_coltype != 0 );

                //printf( "ct_count\n" );
                //for (int i = 0; i < q; ++i) {
                //    printf( "   " );
                //    for (int j = 0; j < 5; ++j)
                //        printf( " %4lld", llong( ct_count( i, j ) ) );
                //    printf( "\n" );
                //}
                //print_vector( "ct_count*", ct_count_ref );
                //print_vector( "ct_idx   ", ct_idx_local );
            }
            MPI_Barrier( MPI_COMM_WORLD );

            if (verbose >= 1 && mpi_rank == 0) {
                printf( "n %lld, n1 %lld, n2 %lld, nsecular %lld, nU123 %lld, "
                        "Q12 %lld : %lld (%lld), Q23 %lld : %lld (%lld)\n",
                        llong( n ), llong( n1 ), llong( n2 ),
                        llong( nsecular ), llong( nU123 ),
                        llong( Q12_begin ), llong( Q12_end ), llong( Q12_end - Q12_begin ),
                        llong( Q23_begin ), llong( Q23_end ), llong( Q23_end - Q23_begin ) );
                printf( "n %lld, n1 %lld, n2 %lld, nsecular %lld, nU123 %lld, "
                        "Q12 %lld : %lld (%lld), Q23 %lld : %lld (%lld)\n",
                        llong( n ), llong( n1 ), llong( n2 ),
                        llong( nsecular_ref ), llong( nU123_ref ),
                        llong( Q12_begin_ref ), llong( Q12_begin_ref + nQ12_ref ), llong( nQ12_ref ),
                        llong( Q23_begin_ref ), llong( Q23_begin_ref + nQ23_ref ), llong( nQ23_ref  ));
            }

            blas::axpy( nsecular, -one, &Dsecular[0], 1, &Dsecular_ref[0], 1 );
            params.error() = blas::nrm2( nsecular, &Dsecular_ref[0], 1 );

            blas::axpy( nsecular, -one, &zsecular[0], 1, &zsecular_ref[0], 1 );
            params.error2() = blas::nrm2( nsecular, &zsecular_ref[0], 1 );

            slate::add( -one, Q, one, Qref );
            params.error3() = slate::norm( slate::Norm::One, Qref );

            slate::add( -one, Qbar, one, Qbar_ref );
            params.error4() = slate::norm( slate::Norm::One, Qbar_ref );

            warn_if( nU123 != nU123_ref );
            warn_if( Q12_begin != Q12_begin_ref - 1 );  // _ref is 1-based
            warn_if( Q23_begin != Q23_begin_ref - 1 );
            warn_if( Q12_end - Q12_begin != nQ12_ref );
            warn_if( Q23_end - Q23_begin != nQ23_ref );
        #endif

        params.okay() = (params.error() <= tol)
                        && (params.error2() <= tol)
                        && (params.error3() <= tol)
                        && (params.error4() <= tol)
                        && (params.error5() <= tol);
    }

    // Repeat header.
    if (verbose >= 1) {
        printf( "type     origin       n        alpha     nb       p       q  deflate           D err      z err      Q err   Qbar err    idx err       time(s)   ref_time(s)  status\n" );
    }
}

//------------------------------------------------------------------------------
void test_stedc_deflate( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            //test_stedc_deflate_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_stedc_deflate_work<double> (params, run);
            break;

        default:
            throw std::exception();
            break;
    }
}
