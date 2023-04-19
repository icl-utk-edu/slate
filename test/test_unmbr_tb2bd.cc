// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "lapack/flops.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
#include "matrix_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>


//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmbr_tb2bd_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;
    const auto mpi_real_type = slate::mpi_type< blas::real_type<scalar_t> >::value;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t band = nb;  // for now use band == nb.
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    slate::Uplo uplo = slate::Uplo::Upper;
    bool upper = uplo == slate::Uplo::Upper;
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho();

    if (! run)
        return;

    if (origin == slate::Origin::ScaLAPACK) {
        params.msg() = "skipping: currently origin=scalapack is not supported";
        return;
    }
    if (origin == slate::Origin::Devices) {
        params.msg() = "skipping: currently origin=devices is not supported";
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    int64_t lda = n;
    int64_t seed[] = {0, 1, 2, 3};

    std::vector<scalar_t> Afull_data( lda*n );
    lapack::larnv(1, seed, Afull_data.size(), &Afull_data[0]);

    // Zero outside the band.
    for (int64_t j = 0; j < n; ++j) {
        if (upper) {
            for (int64_t i = 0; i < n; ++i) {
                if (j > i+band || j < i)
                    Afull_data[i + j*lda] = 0;
            }
        }
        else { // lower
            for (int64_t i = 0; i < n; ++i) {
                if (j < i-band || j > i)
                    Afull_data[i + j*lda] = 0;
            }
        }
        // Diagonal from ge2tb is real.
        Afull_data[j + j*lda] = real( Afull_data[j + j*lda] );
    }

    print_matrix( "Afull_data", n, n, &Afull_data[0], lda, params );

    slate::Target origin_target = origin2target(origin);
    auto Afull = slate::Matrix<scalar_t>::fromLAPACK(
        m, n, &Afull_data[0], lda, nb, p, q, MPI_COMM_WORLD);

    // Copy band of Afull, currently to rank 0.
    auto Aband = slate::TriangularBandMatrix<scalar_t>(
        slate::Uplo::Upper, slate::Diag::NonUnit, n, band, nb,
        1, 1, MPI_COMM_WORLD);
    Aband.insertLocalTiles(origin_target);
    Aband.ge2tbGather( Afull );

    //--------------------
    // [code copied from heev.cc]
    // Matrix to store Householder vectors.
    // Could pack into a lower triangular matrix, but we store each
    // parallelogram in a 2nb-by-nb tile, with nt(nt + 1)/2 tiles.
    int64_t vm = 2*nb;
    int64_t nt = Afull.nt();
    int64_t vn = nt*(nt + 1)/2*nb;
    slate::Matrix<scalar_t> V1(vm, vn, vm, nb, 1, 1, MPI_COMM_WORLD);
    V1.insertLocalTiles(origin_target);

    slate::Matrix<scalar_t> U1(vm, vn, vm, nb, 1, 1, MPI_COMM_WORLD);
    U1.insertLocalTiles(origin_target);
    //--------------------

    // Compute tridiagonal and Householder vectors V.
    if (mpi_rank == 0) {
        slate::tb2bd(Aband, U1, V1);
    }
    //print_matrix( "V1", V1, params );

    // Set V = Identity.
    slate::Matrix<scalar_t> VT(n, n, nb, p, q, MPI_COMM_WORLD);
    VT.insertLocalTiles(origin_target);
    set(zero, one, VT);

    // 1-d V matrix
    slate::Matrix<scalar_t> V1d(VT.m(), VT.n(), VT.tileNb(0), 1, p*q, MPI_COMM_WORLD);
    V1d.insertLocalTiles(target);

    // Set U = Identity.
    slate::Matrix<scalar_t> U(m, n, nb, p, q, MPI_COMM_WORLD);
    U.insertLocalTiles(origin_target);
    set(zero, one, U);

    // 1-d U matrix
    slate::Matrix<scalar_t> U1d(U.m(), U.n(), U.tileNb(0), 1, p*q, MPI_COMM_WORLD);
    U1d.insertLocalTiles(target);

    if (trace)
        slate::trace::Trace::on();
    else
        slate::trace::Trace::off();

    // To compute U, S and VT
    // Copy diagonal & super-diagonal.
    int64_t D_index = 0;
    int64_t E_index = 0;
    int64_t info;
    int64_t min_mn = std::min(m, n);
    std::vector<real_t> Sigma(min_mn);
    std::vector<real_t> E(min_mn - 1);  // super-diagonal
    if (mpi_rank == 0) {
        for (int64_t i = 0; i < std::min(Aband.mt(), Aband.nt()); ++i) {
            // Copy 1 element from super-diagonal tile to E.
            if (i > 0) {
                auto T = Aband(i-1, i);
                E[E_index++] = real( T(T.mb()-1, 0) );
            }

            // Copy main diagonal to Sigma.
            auto T = Aband(i, i);
            for (int64_t j = 0; j < T.nb(); ++j) {
                Sigma[D_index++] = real( T(j, j) );
            }

            // Copy super-diagonal to E.
            for (int64_t j = 0; j < T.nb()-1; ++j) {
                E[E_index++] = real( T(j, j+1) );
            }
        }
    }
    MPI_Bcast( &Sigma[0], min_mn, mpi_real_type, 0, MPI_COMM_WORLD );
    MPI_Bcast( &E[0], min_mn-1, mpi_real_type, 0, MPI_COMM_WORLD );

    print_matrix("D", 1, min_mn,   &Sigma[0], 1, params);
    print_matrix("E", 1, min_mn-1, &E[0],  1, params);

    // Call bdsqr to compute the singular values Sigma
    slate::bdsqr<scalar_t>(slate::Job::Vec, slate::Job::Vec, Sigma, E, U, VT, opts);

    print_matrix("Sigma", 1, min_mn, &Sigma[0],  1, params);
    print_matrix("U_bdsqr", U, params);
    print_matrix("VT_bdsqr", VT, params);

    double time = barrier_get_wtime(MPI_COMM_WORLD);
    // V V1
    // V1T VT
    //auto V = conj_transpose(VT);
    //

    //==================================================
    // Run SLATE test.
    //==================================================
    print_matrix( "U", U, params );
    U1d.redistribute(U);
    print_matrix( "U1d", U1d, params );
    slate::unmtr_hb2st(slate::Side::Left, slate::Op::NoTrans, U1, U1d, opts);
    U.redistribute(U1d);

    print_matrix( "VT", VT, params );
    auto R = conj_transpose(VT);
    print_matrix( "R", R, params );

    //auto V = conj_transpose(VT);
    //auto R = V.emptyLike();
    //R.insertLocalTiles();
    //copy(V, R);
    //print_matrix("V", V, params);
    int64_t nb_A = VT.tileNb( 0 );
    slate::GridOrder grid_order;
    int nprow, npcol;
    VT.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    std::function<int64_t (int64_t j)>
        tileNb = [n, nb_A] (int64_t j) {
            return (j + 1)*nb_A > n ? n%nb_A : nb_A;
        };

    std::function<int (std::tuple<int64_t, int64_t> ij)>
        tileRank = [nprow, npcol]( std::tuple<int64_t, int64_t> ij ) {
            int64_t i = std::get<0>( ij );
            int64_t j = std::get<1>( ij );
            return int( (i%nprow)*npcol + j%npcol );
        };

    int num_devices = blas::get_device_count();
    std::function<int (std::tuple<int64_t, int64_t> ij)>
        tileDevice = [nprow, num_devices]( std::tuple<int64_t, int64_t> ij ) {
            int64_t i = std::get<0>( ij );
            return int( i/nprow )%num_devices;
        };


    slate::Matrix<scalar_t> V(
           n, n, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD );
    V.insertLocalTiles();
    copy(R, V);
    //
    V1d.redistribute(V);
    print_matrix( "V1d", V1d, params );
    //V1d.redistribute(VT);

    //slate::unmbr_tb2bd(slate::Side::Left, slate::Op::NoTrans, V1, V, opts);
    slate::unmtr_hb2st(slate::Side::Left, slate::Op::NoTrans, V1, V1d, opts);
    //VT.redistribute(V1d);
    V.redistribute(V1d);
    print_matrix( "V", V, params );

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace)
        slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;
    // todo: using unmqr's flop count, which is an estimation for unmbr_tb2bd
    double gflop = lapack::Gflop<scalar_t>::unmqr(lapack::Side::Left, n, n, n);
    params.gflops() = gflop / time;

    if (check) {
        //==================================================
        // Test results
        // || I - Q^H Q || / n < tol
        //==================================================
        slate::Matrix<scalar_t> Iden( n, n, nb, p, q, MPI_COMM_WORLD );
        Iden.insertLocalTiles();
        set(zero, one, Iden);

        //auto VT2 = conj_transpose(VT);
        auto VT2 = conj_transpose(V);

        //slate::gemm(-one, VT, VT2, one, Iden);
        slate::gemm(-one, VT2, V, one, Iden);
        print_matrix( "I_VH*V", Iden, params );

        params.ortho() = slate::norm(slate::Norm::One, Iden) / n;

        // If slate::unmbr_tb2bd() fails to update Q, then Q=I.
        // Q is still orthogonal but this should be a failure.
        // todo remove this if block when the backward error
        // check is implemented.
        if (slate::norm(slate::Norm::One, Iden) == zero) {
            params.okay() = false;
            return;
        }

        //==================================================
        // Test results
        // || A - U S V^H || / (n || A ||) < tol
        //==================================================

        // Compute norm Afull to scale error
        real_t Anorm = slate::norm( slate::Norm::One, Afull );

        // Scale V by Sigma to get Sigma V
        //copy(VT2, R);
        slate::scale_row_col( slate::Equed::Col, Sigma, Sigma, U );

        slate::gemm( -one, U, VT2, one, Afull );

        print_matrix( "Aful_-U*S*VT", Afull, params );

        params.error() = slate::norm( slate::Norm::One, Afull ) / (Anorm * n);

        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon() / 2;
        params.okay() = (params.ortho() <= tol && params.error() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_unmbr_tb2bd(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmbr_tb2bd_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmbr_tb2bd_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmbr_tb2bd_work< std::complex<float> > (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmbr_tb2bd_work< std::complex<double> > (params, run);
            break;
    }
}
