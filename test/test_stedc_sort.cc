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
void test_stedc_sort_work( Params& params, bool run )
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t one  = 1.0;
    const scalar_t nan_ = nan("");

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    //bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    //int verbose = params.verbose();
    slate::Origin origin = params.origin();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.ref_time();
    params.error2();
    params.error.name("D err");
    params.error2.name("Z err");

    if (! run) {
        params.matrix.kind.set_default( "ij" );
        return;
    }

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

    // Initialize eigenvalues.
    // All ranks must generate _same_ D!
    std::vector<real_t> D( n );
    int64_t idist = 1;  // uniform [0, 1]
    int64_t iseed[4] = { 0, 0, 0, 3 };
    lapack::larnv( idist, iseed, D.size(), D.data() );
    std::vector<real_t> Dref = D;

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
    slate::generate_matrix( params.matrix, Z );

    // For ease of checking, add D to each row of Z
    int64_t jg = 0;
    for (int64_t j = 0; j < Z.nt(); ++j) {
        for (int64_t i = 0; i < Z.mt(); ++i) {
            if (Z.tileIsLocal( i, j )) {
                auto Zij = Z( i, j );
                for (int64_t jj = 0; jj < Zij.nb(); ++jj) {
                    for (int64_t ii = 0; ii < Zij.mb(); ++ii) {
                        Zij.at( ii, jj ) = int( Zij( ii, jj ) ) + D[ jg + jj ];
                    }
                }
            }
        }
        jg += Z.tileNb( j );
    }

    if (ref) {
        Zref_data.resize( lldZ*nlocZ );
        Zref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, n, &Zref_data[0], lldZ, nb, p, q, MPI_COMM_WORLD );
        slate::copy( Z, Zref, opts );
    }

    auto Zout = Z.emptyLike();
    Zout.insertLocalTiles();
    set( nan_, Zout );

    //print_vector( "D", D, params );
    print_matrix( "Z", Z, params );

    if (trace)
        slate::trace::Trace::on();

    double time = barrier_get_wtime( MPI_COMM_WORLD );

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::stedc_sort( D, Z, Zout, opts );

    params.time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

    if (trace)
        slate::trace::Trace::finish();

    //print_vector( "Dout", D,    params );
    print_matrix( "Zout", Zout, params );

    // todo: check that D & Z are sorted.

    if (ref && m == n) {
        #ifdef SLATE_HAVE_SCALAPACK
            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int Zref_desc[9];
            int mpi_rank_, nprocs;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo( &mpi_rank_, &nprocs );
            Cblacs_get( -1, 0, &ictxt );
            Cblacs_gridinit( &ictxt, "Col", p, q );
            Cblacs_gridinfo( ictxt, &p_, &q_, &myrow_, &mycol_ );

            scalapack_descinit( Zref_desc, n, n, nb, nb, 0, 0, ictxt, lldZ, &info );
            slate_assert( info == 0 );

            // Undocumented: liwork needs max( n, 2*nb + 2*q )
            int64_t lwork  = std::max( n, mlocZ * (nb + nlocZ) );
            int64_t liwork = n + std::max( n, 2*nb + 2*q );
            std::vector<scalar_t> work( lwork );
            std::vector<blas_int> iwork( liwork );

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime( MPI_COMM_WORLD );

            scalapack_plasrt( "i", n, &Dref[0],
                              &Zref_data[0], 1, 1, Zref_desc,
                              &work[0], lwork, &iwork[0], liwork, &info );
            slate_assert( info == 0 );

            params.ref_time() = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            //print_vector( "Dout", D,    params );
            //print_vector( "Dref", Dref, params );
            print_matrix( "Zout", Zout, params );
            print_matrix( "Zref", Zref, params );

            // || D - Dref || should be exactly 0.
            blas::axpy( n, -one, &D[0], 1, &Dref[0], 1 );
            params.error() = blas::nrm2( n, &Dref[0], 1 );
            params.okay() = params.okay() && (params.error2() == 0);

            // || Zout - Zref || should be exactly 0.
            slate::add( -one, Zout, one, Zref, opts );
            params.error2() = slate::norm( slate::Norm::Max, Zref, opts );
            params.okay() = (params.error() == 0
                             && params.error2() == 0);
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_stedc_sort( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_stedc_sort_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_stedc_sort_work<double> (params, run);
            break;

        default:
            throw std::exception();
            break;
    }
}
