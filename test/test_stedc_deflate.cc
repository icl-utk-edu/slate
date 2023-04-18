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
    params.error4.name( "Qtype err" );
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
        {slate::Option::PrintWidth,     params.print_width() },
        {slate::Option::PrintPrecision, params.print_precision() },
    };

    // MPI variables
    int mpi_rank, mpi_size, myrow, mycol;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
    gridinfo( mpi_rank, p, q, &myrow, &mycol );

    if (verbose >= 1 && mpi_rank == 0)
        printf( "\n%%-------------------------------------------------------------------------------\n" );

    // Matrix Q: figure out local size.
    int64_t mlocQ = num_local_rows_cols( n, nb, myrow, p );
    int64_t nlocQ = num_local_rows_cols( n, nb, mycol, q );
    int64_t lldQ  = blas::max( 1, mlocQ ); // local leading dimension of Q

    std::vector<scalar_t> Q_data, Qtype_data;
    slate::Matrix<scalar_t> Q, Qtype;
    if (origin != slate::Origin::ScaLAPACK) {
        Q = slate::Matrix<scalar_t>(
                n, n, nb, p, q, MPI_COMM_WORLD);
        Q.insertLocalTiles( origin2target( origin ) );
        Qtype = slate::Matrix<scalar_t>(
                   n, n, nb, p, q, MPI_COMM_WORLD);
        Qtype.insertLocalTiles( origin2target( origin ) );
    }
    else {
        Q_data.resize( lldQ*nlocQ );
        Qtype_data.resize( lldQ*nlocQ );
        Q    = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Q_data[0],    lldQ, nb, p, q, MPI_COMM_WORLD );
        Qtype = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Qtype_data[0], lldQ, nb, p, q, MPI_COMM_WORLD );
    }
    slate::set( zero, Q, opts );
    int64_t nt = Q.nt();
    int64_t nt1 = int64_t( nt/2 );
    int64_t nt2 = nt - nt1;
    auto Q1 = Q.sub( 0,   nt1 - 1, 0,   nt1 - 1 );
    auto Q2 = Q.sub( nt1, nt  - 1, nt1, nt  - 1 );
    slate::generate_matrix( params.matrix, Q1 );
    slate::generate_matrix( params.matrix, Q2 );
    if (verbose >= 1 && mpi_rank == 0)
        printf( "%% nt %lld, nt1 %lld, nt2 %lld\n",
                llong( nt ), llong( nt1 ), llong( nt2 ) );

    std::vector<scalar_t> D( n ), Dsecular( n, nan("") );
    std::vector<scalar_t> z( n ), zsecular( n, nan("") );
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( 3, iseed, n, &D[0] );  // idist 3: normal
    lapack::larnv( 1, iseed, n, &z[0] );  // idist 1: uniform [0, 1]

    //--------------------------------------------------------------------------
    if (verbose >= 1 && mpi_rank == 0) {
        printf( "%% D1 is odds + 0.11, D2 is evens + 0.33\n"
                "%% z1 is odds + 0.1,  z2 is evens + 0.3\n" );
    }
    int64_t n1 = nt1 * nb;
    int64_t n2 = n - n1;
    for (int i = 0; i < n1; ++i) {
        D[ i ] = 0.10 + 2*(i + 1);
        z[ i ] = 0.1 * D[ i ];
    }
    for (int i = n1; i < n; ++i) {
        D[ i ] = 0.30 + 3*(i - n1 + 1);
        z[ i ] = 0.1 * D[ i ];
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
        int cnt = sscanf( deflate_, "%d %n / %d %n",
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
    for (int64_t j : deflate1) {
        if (verbose >= 2 && mpi_rank == 0)
            printf( "%% deflate type 1, j = %3lld\n", llong( j ) );
        slate_assert( 0 <= j && j < n );
        z[ j ] = (rand() % 20 - 10)*eps * 1e-2;
    }

    // Apply type 2 deflation, D[j2] = D[j1] * (1 + tau), 0 <= tau < 10 eps.
    // Make D[j1] <= D[j2] so D[j1] will sort first, hence be deflated.
    for (auto j12 : deflate2) {
        int64_t j1 = j12.first;
        int64_t j2 = j12.second;
        slate_assert( 0 <= j1 && j1 < n );
        slate_assert( 0 <= j2 && j2 < n );
        D[ j2 ] = D[ j1 ] * (1 + (rand() % 10)*eps);
        if (verbose >= 2 && mpi_rank == 0) {
            printf( "deflate type 2, j1 = %3lld, j2 = %3lld, delta %.2e\n",
                    llong( j1 ), llong( j2 ), D[ j2 ] - D[ j1 ] );
        }
    }
    //--------------------------------------------------------------------------

    std::vector<scalar_t> Dorig( D );
    std::vector<scalar_t> zorig( z );

    // Save copy in ScaLAPACK format for reference.
    std::vector<scalar_t> Qref_data, Qtype_ref_data, Dref, zref;
    slate::Matrix<scalar_t> Qref, Qtype_ref;
    scalar_t rho_ref = rho;
    if (ref) {
        Qref_data.resize( lldQ*nlocQ );
        Qtype_ref_data.resize( lldQ*nlocQ, nan("") );
        Qref     = slate::Matrix<scalar_t>::fromScaLAPACK(
                       n, n, &Qref_data[0],     lldQ, nb, p, q, MPI_COMM_WORLD );
        Qtype_ref = slate::Matrix<scalar_t>::fromScaLAPACK(
                       n, n, &Qtype_ref_data[0], lldQ, nb, p, q, MPI_COMM_WORLD );
        copy( Q, Qref );
        Dref = D;
        zref = z;
    }

    if (verbose >= 1 && mpi_rank == 0) {
        printf( "%%-------------------- SLATE input\n" );
        printf( "n = %lld; n1 = %lld; n2 = %lld;\n", llong( n ), llong( n1 ), llong( n2 ) );
    }
    if (verbose >= 1 && mpi_rank == 0) {
        print_vector( "D", D, params );
        print_vector( "z", z, params );
    }
    print_matrix( "Q", Q, params );

    std::vector<int64_t> itype( n, -1 );
    int64_t nsecular = 0;
    int64_t Qt12_begin = -1, Qt12_end = -1;
    int64_t Qt23_begin = -1, Qt23_end = -1;

    //std::vector<int64_t> isort   ( n, -1 );
    //std::vector<int64_t> ideflate( n, -1 );
    //std::vector<int64_t> iglobal ( n, -1 );
    //std::vector<int> pcols       ( n, -1 );
    //std::vector<int> coltype     ( n, -1 );
    //slate::internal::Array2D<int64_t> ct_count( q, 5 );

    if (trace)
        slate::trace::Trace::on();

    double time = barrier_get_wtime( MPI_COMM_WORLD );

    if (verbose >= 1 && mpi_rank == 0)
        printf( "%%-------------------- SLATE run\n" );

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::stedc_deflate( n, n1, rho,
                          &D[0], &Dsecular[0], &z[0], &zsecular[0],
                          Q, Qtype, &itype[0],
                          nsecular, Qt12_begin, Qt12_end, Qt23_begin, Qt23_end,
                          opts );
    int64_t nU123 = std::max( Qt12_end, Qt23_end )
                  - std::min( Qt12_begin, Qt23_begin );

    params.time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

    if (trace)
        slate::trace::Trace::finish();

    int nsecular_ref = -1, nU123_ref = -1,
        nQt12_ref = -1, Qt12_begin_ref = -1,
        nQt23_ref = -1, Qt23_begin_ref = -1,
        err_itype = 0;
    std::vector<scalar_t>
        Dsecular_ref( n, nan("") ),
        zsecular_ref( n, nan("") );
    std::vector<int>
        itype_ref( n, -1 );

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_;

            // initialize BLACS and ScaLAPACK
            Cblacs_get( -1, 0, &ictxt );
            Cblacs_gridinit( &ictxt, "Col", p, q );
            Cblacs_gridinfo( ictxt, &p_, &q_, &myrow_, &mycol_ );

            if (verbose >= 1 && mpi_rank == 0)
                printf( "%%-------------------- ScaLAPACK input\n" );
            print_matrix( "Qref", Qref, params );
            if (verbose >= 1 && mpi_rank == 0) {
                print_vector( "Dref", Dref, params );
                print_vector( "zref", zref, params );
            }

            // q = npcol
            std::vector<scalar_t>
                Qbuf( 3*n, nan("") );
            std::vector<int>
                ct_count_ref( q*4, -1 ),
                ct_idx_local( q*4, -1 ),
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
                &Qtype_ref_data[0], lldQ,
                &Qbuf[0],
                &ct_count_ref[0], &ct_idx_local[0], q,
                &itype_ref[0], &iglobal_ref[0], &ideflate_ref[0],
                &pcols_ref[0], &coltype_ref[0], &nU123_ref,
                &nQt12_ref, &nQt23_ref, &Qt12_begin_ref, &Qt23_begin_ref );

            params.ref_time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            // convert to 1-based
            for (int j = 0; j < n; ++j) {
                itype_ref[j]     -= 1;
                iglobal_ref[j]  -= 1;
                ideflate_ref[j] -= 1;
            }
            // check errors in int arrays.
            int err_iglobal  = 0;
            int err_ideflate = 0;
            int err_pcols    = 0;
            int err_coltype  = 0;
            for (int j = 0; j < n; ++j) {
                err_itype     = std::abs( itype[j]     - itype_ref[j]     );
                //err_iglobal = std::abs( iglobal[j]  - iglobal_ref[j]  );
                //err_ideflate = std::abs( ideflate[j] - ideflate_ref[j] );
                //err_pcols    = std::abs( pcols[j]    - pcols_ref[j]    );
                //err_coltype  = std::abs( coltype[j]  - coltype_ref[j]  );
            }
            params.error5() = err_itype + err_iglobal + err_ideflate + err_pcols
                              + err_coltype;

            blas::axpy( nsecular, -one, &Dsecular[0], 1, &Dsecular_ref[0], 1 );
            params.error() = blas::nrm2( nsecular, &Dsecular_ref[0], 1 );

            blas::axpy( nsecular, -one, &zsecular[0], 1, &zsecular_ref[0], 1 );
            params.error2() = blas::nrm2( nsecular, &zsecular_ref[0], 1 );

            slate::add( -one, Q, one, Qref, opts );
            params.error3() = slate::norm( slate::Norm::One, Qref, opts );

            slate::add( -one, Qtype, one, Qtype_ref, opts );
            params.error4() = slate::norm( slate::Norm::One, Qtype_ref, opts );

            warn_if( nU123 != nU123_ref );
            warn_if( Qt12_begin != Qt12_begin_ref - 1 );  // _ref is 1-based
            warn_if( Qt23_begin != Qt23_begin_ref - 1 );
            warn_if( Qt12_end - Qt12_begin != nQt12_ref );
            warn_if( Qt23_end - Qt23_begin != nQt23_ref );
        #endif

        params.okay() = (params.error() <= tol)
                        && (params.error2() <= tol)
                        && (params.error3() <= tol)
                        && (params.error4() <= tol)
                        && (params.error5() <= tol);
    }

    //==================================================
    // Print output.
    //==================================================
    int width = params.print_width();
    if (verbose >= 1 && mpi_rank == 0)
        printf( "%%-------------------- SLATE and ScaLAPACK (ref) output\n" );

    print_matrix(     "Qout", Q,    params );
    if (ref)
        print_matrix( "Qref", Qref, params );

    // Print col indices to help parse Q matrices.
    if (verbose >= 2 && mpi_rank == 0) {
        for (int j = 0; j < n; ++j) {
            if (j > 0 && j % nb == 0)
                printf( "    " );
            printf( " %*d", width, j );
        }
        printf( "\n" );
    }

    print_matrix( "Qtype", Qtype, params );
    if (ref)
        print_matrix( "Qtype_ref", Qtype_ref, params );

    if (verbose >= 1 && mpi_rank == 0) {
        // Print vectors only on rank 0.
        // Also print z*sqrt(2), to undo normalization and match zorig.
        std::vector<scalar_t> zsecular2( zsecular ), zsecular2_ref( zsecular_ref );
        blas::scal( zsecular2.size(), sqrt( 2 ), zsecular2.data(), 1 );
        blas::scal( zsecular2_ref.size(), sqrt( 2 ), zsecular2_ref.data(), 1 );

        printf(       "index          = [ " );
        for (int j = 0; j < n; ++j) {
            printf( " %*d", width, j );
        }
        printf( " ]';\n" );

        printf(       "MPI_rank       = [ " );
        for (int j = 0; j < n; ++j) {
            int jj = j/nb;
            printf( " %*d", width, Q.tileRank( 0, jj ) );
        }
        printf( " ]';\n" );

        print_vector(     "Dorig         ", Dorig,         params );
        print_vector(     "% zout = Dorig", z,             params );
        if (ref)
            print_vector( "% zref = Dorig", zref,          params );
        print_vector(     "Dout          ", D,             params );
        if (ref)
            print_vector( "Dref          ", Dref,          params );
        print_vector(     "Dsecular      ", Dsecular,      params );
        if (ref)
            print_vector( "Dsecular_ref  ", Dsecular_ref,  params );
        printf( "\n" );
        print_vector(     "zorig         ", zorig,         params );
        print_vector(     "zsecular      ", zsecular,      params );
        if (ref)
            print_vector( "zsecular_ref  ", zsecular_ref,  params );
        print_vector(     "zsecular_2    ", zsecular2,     params );
        if (ref)
            print_vector( "zsecular_ref_2", zsecular2_ref, params );
        printf( "\n" );
        print_vector(     "itype         ", itype,         params );
        if (ref)
            print_vector( "itype_ref     ", itype_ref,     params );
        //warn_if( err_itype != 0 );

        //print_vector( "iglobal ", iglobal,     params );
        //print_vector( "iglobal*", iglobal_ref, params );
        //warn_if( err_iglobal != 0 );

        //print_vector( "ideflate ", ideflate,     params );
        //print_vector( "ideflate*", ideflate_ref, params );
        //warn_if( err_ideflate != 0 );

        //print_vector( "pcols    ", pcols,     params );
        //print_vector( "pcols*   ", pcols_ref, params );
        //warn_if( err_pcols != 0 );

        //print_vector( "coltype  ", coltype,     params );
        //print_vector( "coltype* ", coltype_ref, params );
        //warn_if( err_coltype != 0 );

        //printf( "ct_count\n" );
        //for (int i = 0; i < q; ++i) {
        //    printf( "   " );
        //    for (int j = 0; j < 5; ++j)
        //        printf( " %4lld", llong( ct_count( i, j ) ) );
        //    printf( "\n" );
        //}
        //print_vector( "ct_count*", ct_count_ref, params );
        //print_vector( "ct_idx   ", ct_idx_local, params );

        printf( "%% SLATE:\n"
                "nsecular = %lld; nU123 = %lld;\n"
                "Qt12 = [ %3lld : %3lld-1 ]+1; nQt12 = %lld;\n"
                "Qt23 = [ %3lld : %3lld-1 ]+1; nQt23 = %lld;\n",
                llong( nsecular ), llong( nU123 ),
                llong( Qt12_begin ), llong( Qt12_end ), llong( Qt12_end - Qt12_begin ),
                llong( Qt23_begin ), llong( Qt23_end ), llong( Qt23_end - Qt23_begin ) );
        if (ref) {
            printf( "%% ScaLAPACK:\n"
                    "nsecular_ref = %lld; nU123_ref = %lld;\n"
                    "Qt12_ref = [ %3lld : %3lld ]; nQt12_ref = %lld;\n"
                    "Qt23_ref = [ %3lld : %3lld ]; nQt23_ref = %lld;\n",
                    llong( nsecular_ref ), llong( nU123_ref ),
                    llong( Qt12_begin_ref ), llong( Qt12_begin_ref + nQt12_ref - 1 ), llong( nQt12_ref ),
                    llong( Qt23_begin_ref ), llong( Qt23_begin_ref + nQt23_ref - 1 ), llong( nQt23_ref  ));
        }
        printf( "done\n" );
    }
}

//------------------------------------------------------------------------------
void test_stedc_deflate( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_stedc_deflate_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_stedc_deflate_work<double> (params, run);
            break;

        default:
            throw std::exception();
            break;
    }
}
