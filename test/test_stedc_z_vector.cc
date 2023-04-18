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
template <typename scalar_t>
void test_stedc_z_vector_work( Params& params, bool run )
{
    //using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::max;

    // Constants
    const scalar_t zero = 0.0;
    //const scalar_t one  = 1.0;
    //const real_t tol = params.tol() * unit_roundoff<real_t>();

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

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    slate::Options const opts =  {
        // {slate::Option::Target,         target },
        {slate::Option::PrintVerbose,   params.verbose() },
        {slate::Option::PrintPrecision, params.print_precision() },
        {slate::Option::PrintWidth,     params.print_width() },
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    gridinfo( mpi_rank, p, q, &myrow, &mycol );

    // Matrix Q: figure out local size.
    int64_t mlocQ = num_local_rows_cols( n, nb, myrow, p );
    int64_t nlocQ = num_local_rows_cols( n, nb, mycol, q );
    int64_t lldQ  = blas::max( 1, mlocQ ); // local leading dimension of Q
    std::vector<scalar_t> Q_data;

    slate::Matrix<scalar_t> Q;
    if (origin != slate::Origin::ScaLAPACK) {
        Q = slate::Matrix<scalar_t>(
                n, n, nb, p, q, MPI_COMM_WORLD);
        Q.insertLocalTiles( origin2target( origin ) );
    }
    else {
        Q_data.resize( lldQ*nlocQ );
        Q = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Q_data[0], lldQ, nb, p, q, MPI_COMM_WORLD );
    }
    slate::set( zero, Q, opts );
    int64_t nt = Q.nt();
    int64_t nt1 = int64_t( nt/2 );
    auto Q1 = Q.sub( 0,   nt1 - 1, 0,   nt1 - 1 );
    auto Q2 = Q.sub( nt1, nt  - 1, nt1, nt  - 1 );
    slate::generate_matrix( params.matrix, Q1 );
    slate::generate_matrix( params.matrix, Q2 );

    // todo: n+3 overallocate so we can check that it doesn't overwrite.
    std::vector<scalar_t> z( n+3, nan("") );

    // Save copy in ScaLAPACK format for reference.
    std::vector<scalar_t> Qref_data;
    slate::Matrix<scalar_t> Qref;
    if (check || ref) {
        Qref_data.resize( lldQ*nlocQ );
        Qref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Qref_data[0], lldQ, nb, p, q, MPI_COMM_WORLD );
        copy( Q, Qref );
    }

    print_matrix( "Q", Q, params );
    print_matrix( "Qref", Qref, params );

    if (trace)
        slate::trace::Trace::on();

    double time = barrier_get_wtime( MPI_COMM_WORLD );

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::stedc_z_vector( Q, z, opts );

    params.time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

    if (trace)
        slate::trace::Trace::finish();

    if (verbose >= 2 && mpi_rank == 0) {
        print_vector( "zout", z.size(), &z[0], 1, params );
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int Q_desc[9];
            //int mpi_rank_ = 0, nprocs = 1;

            // initialize BLACS and ScaLAPACK
            //Cblacs_pinfo( &mpi_rank_, &nprocs );
            Cblacs_get( -1, 0, &ictxt );
            Cblacs_gridinit( &ictxt, "Col", p, q );
            Cblacs_gridinfo( ictxt, &p_, &q_, &myrow_, &mycol_ );

            //slate_assert( mpi_rank == mpi_rank_ );
            //slate_assert( p*q <= nprocs );
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            scalapack_descinit( Q_desc, n, n, nb, nb, 0, 0, ictxt, lldQ, &info );
            slate_assert( info == 0 );

            // plaedz work size is undocumented; guess n.
            std::vector<scalar_t> zref( n+3, nan("") ), work( n, nan("") );
            int64_t n1 = Q1.n();

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime( MPI_COMM_WORLD );

            scalapack_plaedz( n, n1, 1,
                              &Qref_data[0], 1, 1, lldQ, Q_desc,
                              &zref[0], &work[0] );

            params.ref_time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            if (verbose >= 2 && mpi_rank == 0) {
                print_vector( "zref", zref.size(), &zref[0], 1, params );
                print_vector( "work", work.size(), &work[0], 1, params );
            }

            // Forward error || z - zref || should be zero.
            blas::axpy( n, -1.0, &z[0], 1, &zref[0], 1 );
            params.error() = blas::nrm2( n, &zref[0], 1 );
            params.okay() = (params.error() == 0.0);
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_stedc_z_vector( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_stedc_z_vector_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_stedc_z_vector_work<double> (params, run);
            break;

        default:
            throw std::exception();
            break;
    }
}
