// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_gels_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0, one = 1;

    // get & mark input values
    slate::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::Method methodGels = params.method_gels();
    slate::Method methodCholqr = params.method_cholQR();
    bool consistent = true;
    params.matrix.mark();
    params.matrixB.mark();
    params.matrixC.mark();

    // mark non-standard output values
    params.error.name("leastsqr");
    params.error2();
    params.error2.name("min norm");
    params.error3();
    params.error3.name("residual");
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib},
        {slate::Option::MethodCholQR, methodCholqr},
        {slate::Option::MethodGels, methodGels}
    };

    // A is m-by-n, BX is max(m, n)-by-nrhs.
    // If op == NoTrans, op(A) is m-by-n, X is n-by-nrhs, B is m-by-nrhs
    // otherwise,        op(A) is n-by-m, X is m-by-nrhs, B is n-by-nrhs.
    int64_t opAm = (trans == slate::Op::NoTrans ? m : n);
    int64_t opAn = (trans == slate::Op::NoTrans ? n : m);
    int64_t maxmn = std::max(m, n);

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Figure out local size.
    // matrix A, m-by-n
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A

    // matrix X0, opAn-by-nrhs
    // used if making a consistent equation, B = A*X0
    int64_t mlocX0 = num_local_rows_cols(opAn, nb, myrow, p);
    int64_t nlocX0 = num_local_rows_cols(nrhs, nb, mycol, q);
    int64_t lldX0  = blas::max(1, mlocX0); // local leading dimension of A

    // matrix BX, which stores B (input) and X (output), max(m, n)-by-nrhs
    int64_t mlocBX = num_local_rows_cols(maxmn, nb, myrow, p);
    int64_t nlocBX = num_local_rows_cols(nrhs,  nb, mycol, q);
    int64_t lldBX  = blas::max(1, mlocBX); // local leading dimension of A

    // ScaLAPACK data if needed.
    std::vector<scalar_t> A_data, X0_data, BX_data;

    slate::Matrix<scalar_t> A, X0, BX;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);

        X0 = slate::Matrix<scalar_t>(opAn, nrhs, nb, p, q, MPI_COMM_WORLD);
        X0.insertLocalTiles(origin_target);

        BX = slate::Matrix<scalar_t>(maxmn, nrhs, nb, p, q, MPI_COMM_WORLD);
        BX.insertLocalTiles(origin_target);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A_data.resize( lldA * nlocA );
        X0_data.resize( lldX0 * nlocX0 );
        BX_data.resize( lldBX * nlocBX );

        A  = slate::Matrix<scalar_t>::fromScaLAPACK(
                 m,     n,    &A_data[0],  lldA,  nb, p, q, MPI_COMM_WORLD);
        X0 = slate::Matrix<scalar_t>::fromScaLAPACK(
                 opAn,  nrhs, &X0_data[0], lldX0, nb, p, q, MPI_COMM_WORLD);
        BX = slate::Matrix<scalar_t>::fromScaLAPACK(
                 maxmn, nrhs, &BX_data[0], lldBX, nb, p, q, MPI_COMM_WORLD);
    }
    // Create SLATE matrix from the ScaLAPACK layouts
    // slate::TriangularFactors<scalar_t> T;
    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrixB, BX);
    slate::generate_matrix(params.matrixC, X0);

    // In square case, B = X = BX. In rectangular cases, B or X is sub-matrix.
    auto B = BX;
    auto X = BX;
    if (opAm > opAn) {
        // over-determined
        X = BX.slice(0, opAn-1, 0, nrhs-1);
    }
    else if (opAm < opAn) {
        // under-determined
        B = BX.slice(0, opAm-1, 0, nrhs-1);
    }

    // Apply trans
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        opA = conjTranspose(A);

    if (verbose >= 1) {
        printf( "%% A   %6lld-by-%6lld\n", llong(   A.m() ), llong(   A.n() ) );
        printf( "%% opA %6lld-by-%6lld\n", llong( opA.m() ), llong( opA.n() ) );
        printf( "%% X0  %6lld-by-%6lld\n", llong(  X0.m() ), llong(  X0.n() ) );
        printf( "%% B   %6lld-by-%6lld\n", llong(   B.m() ), llong(   B.n() ) );
        printf( "%% X   %6lld-by-%6lld\n", llong(   X.m() ), llong(   X.n() ) );
        printf( "%% BX  %6lld-by-%6lld\n", llong(  BX.m() ), llong(  BX.n() ) );
    }

    // Form consistent RHS, B = A * X0.
    if (consistent) {
        slate::multiply(one, opA, X0, zero, B);
        // Using traditional BLAS/LAPACK name
        // slate::gemm(one, opA, X0, zero, B);
    }

    print_matrix( "A", A, params );
    print_matrix( "X0", X0, params );
    print_matrix( "B", B, params );
    print_matrix( "X", X, params );
    print_matrix( "BX", BX, params );

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> Aref_data( lldA*nlocA );
    std::vector<scalar_t> BXref_data( lldBX*nlocBX );
    slate::Matrix<scalar_t> Aref, opAref, BXref, Bref;
    if (check || ref) {
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   m, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        slate::copy(A, Aref);

        BXref = slate::Matrix<scalar_t>::fromScaLAPACK(
                    maxmn, nrhs, &BXref_data[0], lldBX, nb, p, q, MPI_COMM_WORLD);
        slate::copy(BX, BXref);

        if (opAm >= opAn) {
            Bref = BXref;
        }
        else if (opAm < opAn) {
            Bref = BXref.slice(0, opAm-1, 0, nrhs-1);
        }

        // Apply trans
        opAref = Aref;
        if (trans == slate::Op::Trans)
            opAref = transpose(Aref);
        else if (trans == slate::Op::ConjTrans)
            opAref = conjTranspose(Aref);
    }

    double gflop = lapack::Gflop<scalar_t>::gels(m, n, nrhs);

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(MPI_COMM_WORLD);

        //==================================================
        // Run SLATE test.
        //==================================================
        slate::least_squares_solve(opA, BX, opts);
        // Using traditional BLAS/LAPACK name
        // slate::gels(opA, T, BX, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
        params.gflops() = gflop / time;

        print_matrix( "A2", A, params );
        print_matrix( "BX2", BX, params );
    }

    if (check) {
        //==================================================
        // Test results.
        //==================================================
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

        real_t opA_norm = slate::norm(slate::Norm::One, opAref);
        real_t   B_norm = slate::norm(slate::Norm::One, Bref);

        // todo: need to store Bref for reference ScaLAPACK run?
        // residual = B - op(A) X, stored in Bref
        slate::multiply(-one, opAref, X, one, Bref);
        // Using traditional BLAS/LAPACK name
        // slate::gemm(-one, opAref, X, one, Bref);

        if (opAm >= opAn) {
            //--------------------------------------------------
            // Over-determined case, least squares solution.
            // Check that the residual Res = AX - B is orthogonal to op(A):
            //
            //      || Res^H op(A) ||_1
            //     ------------------------------------------- < tol * epsilon
            //      max(m, n, nrhs) || op(A) ||_1 * || B ||_1

            // todo: scale residual to unit max, and scale error below
            // see LAPACK [sdcz]qrt17.f
            //real_t R_max = slate::norm(slate::Norm::Max, B);
            //slate::scale(1, R_max, B);

            // RA = R^H op(A)
            slate::Matrix<scalar_t> RA(nrhs, opAn, nb, p, q, MPI_COMM_WORLD);
            RA.insertLocalTiles();
            auto RT = conjTranspose(Bref);
            slate::multiply(one, RT, opA, zero, RA);
            // Using traditional BLAS/LAPACK name
            // slate::gemm(one, RT, opA, zero, RA);

            real_t error = slate::norm(slate::Norm::One, RA);
            if (opA_norm != 0)
                error /= opA_norm;
            // todo: error *= R_max
            if (B_norm != 0)
                error /= B_norm;
            error /= blas::max(m, n, nrhs);
            params.error() = error;
            params.okay() = (params.error() <= tol);
        }
        else {
            //--------------------------------------------------
            // opAm < opAn
            // Under-determined case, minimum norm solution.
            // Check that X is in the row-span of op(A),
            // i.e., it has no component in the null space of op(A),
            // by doing QR factorization of D = [ op(A)^H, X ] and examining
            // E = R( opAm : opAm+nrhs-1, opAm : opAm+nrhs-1 ).
            //
            //      || E ||_max / max(m, n, nrhs) < tol * epsilon

            // op(A)^H is opAn-by-opAm, X is opAn-by-nrhs
            // D = [ op(A)^H, X ] is opAn-by-(opAm + pad + nrhs)
            // Xstart = (opAm + pad) / nb
            // padding with zero columns so X starts on nice boundary to need only local copy.
            int64_t Xstart = slate::ceildiv( slate::ceildiv( opAm, nb ), q ) * q;
            slate::Matrix<scalar_t> D(opAn, Xstart*nb + nrhs, nb, p, q, MPI_COMM_WORLD);
            D.insertLocalTiles();

            // zero D.
            // todo: only need to zero the padding tiles in D.
            set(zero, D);

            // copy op(A)^H -> D
            // todo: support op(A)^H = A^H. Requires distributed copy of A^H to D.
            slate_assert(trans != slate::Op::NoTrans);
            auto DS = D.slice(0, Aref.m()-1, 0, Aref.n()-1);
            copy(Aref, DS);
            auto DX = D.sub(0, D.mt()-1, Xstart, D.nt()-1);
            copy(X, DX);

            if (verbose >= 1)
                printf( "%% D %lld-by-%lld\n", llong( D.mt() ), llong( D.nt() ) );

            print_matrix("D", D, params);

            slate::TriangularFactors<scalar_t> TD;
            slate::qr_factor(D, TD);
            // Using traditional BLAS/LAPACK name
            // slate::geqrf(D, TD);

            if (verbose > 1) {
                auto DR = slate::TrapezoidMatrix<scalar_t>(
                              slate::Uplo::Upper, slate::Diag::NonUnit, D );
                print_matrix("DR", DR, params);
            }

            // error = || R_{opAm : opAn-1, opAm : opAm+nrhs-1} ||_max
            // todo: if we can slice R at arbitrary row, just do norm(Max, R_{...}).
            // todo: istart/ioffset assumes fixed nb
            int64_t istart  = opAm / nb; // row m's tile
            int64_t ioffset = opAm % nb; // row m's offset in tile
            real_t local_error = 0;
            for (int64_t i = istart; i < D.mt(); ++i) {
                for (int64_t j = std::max(i, Xstart); j < D.nt(); ++j) { // upper
                    if (D.tileIsLocal(i, j)) {
                        auto T = D(i, j);
                        if (i == j) {
                            for (int64_t jj = 0; jj < T.nb(); ++jj)
                                for (int64_t ii = ioffset; ii < jj && ii < T.mb(); ++ii) // upper
                                    local_error = std::max( local_error, std::abs( T(ii, jj) ) );
                        }
                        else {
                            for (int64_t jj = 0; jj < T.nb(); ++jj)
                                for (int64_t ii = ioffset; ii < T.mb(); ++ii)
                                    local_error = std::max( local_error, std::abs( T(ii, jj) ) );
                        }
                    }
                }
                ioffset = 0; // no offset for subsequent block rows
            }
            real_t error2;
            MPI_Allreduce(&local_error, &error2, 1, slate::mpi_type<real_t>::value, MPI_MAX, BX.mpiComm());
            error2 /= blas::max(m, n, nrhs);
            params.error2() = error2;
            params.okay() = (params.error2() <= tol);
        }

        //--------------------------------------------------
        // If op(A) X = B is consistent, because either B = op(A) X0
        // or opAm <= opAn, check the residual:
        //
        //      || Res ||_1
        //     ----------------------------------- < tol * epsilon
        //      max(m, n) || op(A) ||_1 || X ||_1
        if (consistent || opAm <= opAn) {
            real_t X_norm = slate::norm(slate::Norm::One, X);
            real_t error3 = slate::norm(slate::Norm::One, Bref);
            if (opA_norm != 0)
                error3 /= opA_norm;
            if (X_norm != 0)
                error3 /= X_norm;
            error3 /= std::max(m, n);
            params.error3() = error3;
            params.okay() = (params.okay() && params.error3() <= tol);
        }
    }

    if (ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // A comparison with a reference routine from ScaLAPACK for timing only

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int mpi_rank_ = 0, nprocs = 1;

            // initialize BLACS and ScaLAPACK
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank_ == mpi_rank );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            // workspace for ScaLAPACK
            int64_t lwork;
            std::vector<scalar_t> work;

            int Aref_desc[9], BXref_desc[9];
            scalapack_descinit(Aref_desc, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            scalapack_descinit(BXref_desc, maxmn, nrhs, nb, nb, 0, 0, ictxt, mlocBX, &info);
            slate_assert(info == 0);

            int64_t info_ref = 0;

            // query for workspace size
            scalar_t dummy;
            scalapack_pgels(op2str(trans), m, n, nrhs,
                            &Aref_data[0],  1, 1, Aref_desc,
                            &BXref_data[0], 1, 1, BXref_desc,
                            &dummy, -1, &info_ref);
            slate_assert(info_ref == 0);
            lwork = int64_t( real( dummy ) );
            work.resize(lwork);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            scalapack_pgels(op2str(trans), m, n, nrhs,
                            &Aref_data[0],  1, 1, Aref_desc,
                            &BXref_data[0], 1, 1, BXref_desc,
                            work.data(), lwork, &info_ref);
            slate_assert(info_ref == 0);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            params.ref_time() = time;
            params.ref_gflops() = gflop / time;

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_gels(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gels_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gels_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gels_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gels_work<std::complex<double>> (params, run);
            break;
    }
}
