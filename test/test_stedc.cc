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
#include "matrix_generator.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_stedc_work( Params& params, bool run )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::max;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    const scalar_t nan_ = nan("");
    const real_t   eps  = std::numeric_limits<real_t>::epsilon();
    const real_t   tol  = params.tol() * eps/2;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    int64_t iseed[4] = { 83, 76, 84, 69 };

    // mark non-standard output values
    params.time();
    //params.gflops();
    params.ref_time();
    //params.ref_gflops();
    params.error2();
    params.ortho();
    params.error.name("back err");
    params.error2.name("value err");
    params.ortho.name("Z ortho");

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target,         target },
        {slate::Option::PrintVerbose,   params.verbose() },
        {slate::Option::PrintPrecision, params.print_precision() },
        {slate::Option::PrintWidth,     params.print_width() },
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    gridinfo( mpi_rank, p, q, &myrow, &mycol );

    // Initialize the diagonal and subdiagonal
    // All ranks must generate _same_ D and E!
    std::vector<real_t> D( n ), E( n - 1 );
    int64_t idist = 3; // normal

    if (params.matrix.kind() == "identity") {
        std::fill( &D[0], &D[n],   one  );
        std::fill( &E[0], &E[n-1], zero );
    }
    else if (params.matrix.kind() == "diag") {
        lapack::larnv( idist, iseed, D.size(), D.data() );
        std::fill( &E[0], &E[n-1], zero );
    }
    else {
        lapack::larnv( idist, iseed, D.size(), D.data() );
        lapack::larnv( idist, iseed, E.size(), E.data() );
    }
    std::vector<real_t> Dref = D;
    std::vector<real_t> Eref = E;
    real_t Anorm = lapack::lanht( lapack::Norm::One, n, D.data(), E.data() );

    // Matrix Z: figure out local size.
    int64_t mlocZ = num_local_rows_cols( n, nb, myrow, p );
    int64_t nlocZ = num_local_rows_cols( n, nb, mycol, q );
    int64_t lldZ  = blas::max( 1, mlocZ ); // local leading dimension of Z

    slate::Matrix<scalar_t> Z, Zref; // Matrix of the eigenvectors
    std::vector<scalar_t> Z_data, Zref_data;

    if (origin != slate::Origin::ScaLAPACK) {
        Z = slate::Matrix<scalar_t>(
                n, n, nb, p, q, MPI_COMM_WORLD);
        Z.insertLocalTiles( origin2target( origin ) );
    }
    else {
        Z_data.resize( lldZ*nlocZ );
        Z = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Z_data[0], lldZ, nb, p, q, MPI_COMM_WORLD );
    }
    set( nan_, nan_, Z );

    if (ref) {
        Zref_data.resize( lldZ*nlocZ );
        Zref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Zref_data[0], lldZ, nb, p, q, MPI_COMM_WORLD );
        // undocumented, ScaLAPACK seems to require Z = Identity on input,
        // and pdsyevd sets it that way. Otherwise, deflated eigvecs
        // have entries as set here (say, -1).
        set( zero, one, Zref );
    }

    //print_matrix( "Z", Z, params );
    //print_vector( "D", D, params );
    //print_vector( "E", E, params );

    if (trace)
        slate::trace::Trace::on();

    double time = barrier_get_wtime( MPI_COMM_WORLD );

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::stedc( D, E, Z, opts );

    params.time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

    if (trace)
        slate::trace::Trace::finish();

    if (check) {
        //==================================================
        // Test results by checking the orthogonality of Z.
        //
        //     || Z^H Z - I ||_f
        //     ----------------- < tol * epsilon
        //           n
        //
        //==================================================
        slate::Matrix<scalar_t> R( n, n, nb, p, q, MPI_COMM_WORLD );
        R.insertLocalTiles();
        slate::set( zero, one, R, opts );
        auto ZT = conj_transpose( Z );
        slate::gemm( one, ZT, Z, -one, R, opts );
        params.ortho() = slate::norm( slate::Norm::Fro, R, opts ) / n;
        //printf( "ortho %.2e\n", params.ortho() );

        //==================================================
        // Test results by checking the backward error.
        //
        //     || A - Z Lambda Z^H ||_1
        //     ------------------------- < tol * epsilon
        //           n || A ||_1
        //
        //==================================================
        // Reset R with D on diag, E on sub- & super-diagonal.
        slate::set( zero, R, opts );
        int64_t jj = 0;  // global index of 1st col of tile j.
        int64_t nt = R.nt();
        for (int64_t j = 0; j < nt; ++j) {
            if (R.tileIsLocal( j, j )) {  // diag tile
                auto Rjj = R( j, j );
                scalar_t* data = Rjj.data();
                int64_t nb_ = Rjj.nb();
                int64_t lda = Rjj.stride();
                // Copy diag, sub-diag, super-diag.
                blas::copy( nb_,     &Dref[ jj ], 1, &data[ 0   ], lda+1 );
                blas::copy( nb_ - 1, &Eref[ jj ], 1, &data[ 1   ], lda+1 );
                blas::copy( nb_ - 1, &Eref[ jj ], 1, &data[ lda ], lda+1 );
            }
            if (j > 0 && R.tileIsLocal( j-1, j )) {  // super-diag tile
                auto Rij = R( j-1, j );
                Rij.at( Rij.mb() - 1, 0 ) = Eref[ jj-1 ];
            }
            if (j > 0 && R.tileIsLocal( j, j-1 )) {  // sub-diag tile
                auto Rij = R( j, j-1 );
                Rij.at( 0, Rij.nb() - 1 ) = Eref[ jj-1 ];
            }
            jj += R.tileNb( j );
        }
        print_matrix( "R", R, params );
        slate::Matrix<scalar_t> Z_Lambda( n, n, nb, p, q, MPI_COMM_WORLD );
        Z_Lambda.insertLocalTiles();
        slate::copy( Z, Z_Lambda, opts );
        slate::scale_row_col( slate::Equed::Col, D, D, Z_Lambda, opts );
        slate::gemm( -one, Z_Lambda, ZT, one, R, opts );
        params.error() = slate::norm( slate::Norm::One, R, opts )
                       / (n * Anorm);
        params.okay() = (params.ortho() <= tol
                         && params.error() <= tol);
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            if (verbose) {
                printf( "\n\n\n"
                        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                        "\n\n\n" );
            }

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int Zref_desc[9];
            int mpi_rank_, nprocs;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo( &mpi_rank_, &nprocs );
            Cblacs_get( -1, 0, &ictxt );
            Cblacs_gridinit( &ictxt, "Col", p, q );
            Cblacs_gridinfo( ictxt, &p_, &q_, &myrow_, &mycol_ );
            slate_assert( mpi_rank == mpi_rank_ );
            slate_assert( p*q <= nprocs );
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit( Zref_desc, n, n, nb, nb, 0, 0, ictxt, lldZ, &info );
            slate_assert( info == 0 );

            // Query for workspace size.
            //printf( "query\n" );
            scalar_t work_query;
            blas_int iwork_query;
            //printf( "call scalapack_pstedc query\n" );
            scalapack_pstedc( "i", n, &Dref[0], &Eref[0],
                              &Zref_data[0], 1, 1, Zref_desc,
                              &work_query, -1, &iwork_query, -1, &info );
            //printf( "done scalapack_pstedc query, info=%d\n", info );
            slate_assert( info == 0 );

            int64_t lwork  = work_query;
            int64_t liwork = iwork_query;
            //printf( "query lwork %lld, liwork %lld\n", llong( lwork ), llong( liwork ) );
            std::vector<scalar_t> work( lwork );
            std::vector<blas_int> iwork( liwork );
            // undocumented, needs max( 6 n + 2 mloc nloc, mloc (nb + nloc) )
            lwork = std::max( lwork, mlocZ * (nb + nlocZ) );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime( MPI_COMM_WORLD );

            //print_matrix( "Zref_in", Zref, params );
            //print_vector( "Dref_in", Dref, params );
            //print_vector( "Eref_in", Eref, params );

            //printf( "call scalapack_pstedc\n" );
            scalapack_pstedc( "i", n, &Dref[0], &Eref[0],
                              &Zref_data[0], 1, 1, Zref_desc,
                              &work[0], lwork, &iwork[0], liwork, &info );
            //printf( "done scalapack_pstedc, info=%d\n", info );
            slate_assert( info == 0 );

            params.ref_time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            print_matrix( "Zout", Z,    params );
            print_matrix( "Zref", Zref, params );
            //print_vector( "Dout", D,    params );
            //print_vector( "Dref", Dref, params );

            // Relative forward error: || D - Dref || / || Dref || .
            real_t D_norm = blas::nrm2( n, &Dref[0], 1 );
            blas::axpy( n, -1.0, &D[0], 1, &Dref[0], 1 );
            params.error2() = blas::nrm2( n, &Dref[0], 1 ) / D_norm;
            params.okay() = params.okay() && (params.error2() <= tol);
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_stedc( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_stedc_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_stedc_work<double> (params, run);
            break;

        default:
            throw std::exception();
            break;
    }
}
