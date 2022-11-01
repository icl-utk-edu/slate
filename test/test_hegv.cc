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
#include <limits>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hegv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // get & mark input values
    slate::Job jobz = params.jobz();
    slate::Uplo uplo = params.uplo();
    int64_t itype = params.itype();
    int64_t n = params.dim.n();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();
    params.matrixB.mark();
    params.matrixC.mark();

    params.time();
    params.ref_time();
    params.error2();

    if (! run) {
        // B matrix must be Symmetric Positive Definite (SPD) for scalapack_phegvx
        params.matrixB.kind.set_default( "rand_dominant" );
        return;
    }

    // MPI variables
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Skip invalid or unimplemented options.
    if (uplo == slate::Uplo::Upper) {
        params.msg() = "skipping: Uplo::Upper isn't supported.";
        return;
    }
    if (p != q) {
        params.msg() = "skipping: requires square process grid (p == q).";
        return;
    }

    // Figure out local size.
    // matrix A (local input/local output), n-by-n, Hermitian
    int64_t mlocA = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data;

    // matrix B (local input/local output), n-by-n, Hermitian
    int64_t mlocB = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocB = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldB  = blas::max(1, mlocB); // local leading dimension of B
    std::vector<scalar_t> B_data;

    // vector Lambda (global output), gets eigenvalues in decending order
    std::vector<real_t> Lambda(n);

    // matrix Z (local output), n-by-n, gets orthonormal eigenvectors corresponding to Lambda
    int64_t mlocZ = num_local_rows_cols(n, nb, myrow, p);
    int64_t nlocZ = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldZ  = blas::max(1, mlocZ); // local leading dimension of Z
    std::vector<scalar_t> Z_data;

    // Initialize SLATE data structures
    slate::HermitianMatrix<scalar_t> A;
    slate::HermitianMatrix<scalar_t> B;
    slate::Matrix<scalar_t> Z;

    // Copy data from ScaLAPACK as needed
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, mpi_comm);
        A.insertLocalTiles(origin_target);

        B = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, mpi_comm);
        B.insertLocalTiles(origin_target);

        Z = slate::Matrix<scalar_t>(n, n, nb, p, q, mpi_comm);
        Z.insertLocalTiles(origin_target);
    }
    else {
        A_data.resize( lldA * nlocA );
        B_data.resize( lldB * nlocB );
        Z_data.resize( lldZ * nlocZ );
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, &A_data[0], lldA, nb, p, q, mpi_comm);
        B = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, &B_data[0], lldB, nb, p, q, mpi_comm);
        Z = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Z_data[0], lldZ, nb, p, q, mpi_comm);
    }

    slate::generate_matrix( params.matrix,  A );
    slate::generate_matrix( params.matrixB, B );

    if (verbose >= 1) {
        printf("%% A   %6lld-by-%6lld\n", llong( A.m() ), llong( A.n() ));
        printf("%% B   %6lld-by-%6lld\n", llong( B.m() ), llong( B.n() ));
        printf("%% Z   %6lld-by-%6lld\n", llong( Z.m() ), llong( Z.n() ));
    }

    print_matrix("A", A, params);
    print_matrix("B", B, params);
    print_matrix("Z", Z, params);

    std::vector<scalar_t> Aref_data, Bref_data, Zref_data;
    std::vector<real_t> Lambda_ref;
    slate::HermitianMatrix<scalar_t> Aref;
    slate::HermitianMatrix<scalar_t> Bref;
    if (ref || check) {
        Aref_data.resize( lldA * nlocA );
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                   uplo, n, &Aref_data[0], lldA, nb, p, q, mpi_comm);
        slate::copy(A, Aref);

        Bref_data.resize( lldB * nlocB );
        Bref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                   uplo, n, &Bref_data[0], lldB, nb, p, q, mpi_comm);
        slate::copy(B, Bref);

        Zref_data.resize( lldZ * nlocZ );
        Lambda_ref.resize( Lambda.size() );
    }

    slate::HermitianMatrix<scalar_t> A_orig;
    slate::HermitianMatrix<scalar_t> B_orig;
    if (check) {
        A_orig = A.emptyLike();
        A_orig.insertLocalTiles();
        copy(A, A_orig);

        B_orig = B.emptyLike();
        B_orig.insertLocalTiles();
        copy(B, B_orig);
    }

    slate::Options const opts = {
        {slate::Option::Lookahead,       lookahead},
        {slate::Option::Target,          target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking,   ib}
    };

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        double time = barrier_get_wtime(mpi_comm);

        //==================================================
        // Run SLATE test.
        //==================================================
        if (jobz == slate::Job::NoVec) {
            slate::eig_vals( itype, A, B, Lambda, opts );
            // Using traditional BLAS/LAPACK name
            // slate::hegv( itype, A, B, Lambda, opts );
        }
        else {
            slate::eig( itype, A, B, Lambda, Z, opts );
            // Using traditional BLAS/LAPACK name
            // slate::hegv( itype, A, B, Lambda, Z, opts );
        }

        time = barrier_get_wtime(mpi_comm) - time;
        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
    }

    print_matrix("A", A, params);
    print_matrix("B", B, params);
    print_matrix("Z", Z, params);

    if (check && jobz == slate::Job::Vec) {
        // do error checks for the operations
        // from ScaLAPACK testing (pzgsepchk.f)
        // where A is a symmetric matrix,
        // B is symmetric positive definite,
        // Z is orthogonal containing eigenvectors
        // and D is diagonal containing eigenvalues
        // One of the following test ratios is computed:
        // itype = 1: R_norm = | A Z - B Z D | / ( |A| |Z| n ulp )
        // itype = 2: R_norm = | A B Z - Z D | / ( |A| |Z| n ulp )
        // itype = 3: R_norm = | B A Z - Z D | / ( |A| |Z| n ulp )

        // create C as a empty allocated matrix
        slate::Matrix<scalar_t> C = Z.emptyLike();
        C.insertLocalTiles();

        // calculate some norms
        real_t A_norm = slate::norm(slate::Norm::One, A_orig);
        real_t Z_norm = slate::norm(slate::Norm::One, Z);
        real_t R_norm = 0;

        if (itype == 1) {
            // C = AZ + 0*C = AZ
            slate::hemm(slate::Side::Left, one, A_orig, Z, zero, C, opts);
            // Z = ZD
            // todo: Does the Z matrix need to be forced back to the CPU if it is not there?
            int64_t joff = 0;
            for (int64_t j = 0; j < Z.nt(); ++j) {
                int64_t ioff = 0;
                for (int64_t i = 0; i < Z.mt(); ++i) {
                    if (Z.tileIsLocal(i, j)) {
                        auto T = Z.at(i, j);
                        for (int jj = 0; jj < T.nb(); ++jj)
                            for (int ii = 0; ii < T.mb(); ++ii)
                                T.at(ii, jj) *= Lambda[ jj + joff ];
                    }
                    ioff += Z.tileMb(i);
                }
                joff += Z.tileNb(j);
            }
            // C = C - BZ  (i.e. AZ - BZD)
            slate::hemm(slate::Side::Left, one, B_orig, Z, -one, C, opts);
            // R_norm = | A Z - B Z D | / ( |A| |Z| n )
            R_norm = slate::norm(slate::Norm::One, C) / A_norm / Z_norm / n;
        }
        else if (itype == 2) {
            // C = Bz + 0*C = AZ
            slate::hemm(slate::Side::Left, one, B_orig, Z, zero, C, opts);
            // Z = ZD
            int64_t joff = 0;
            for (int64_t j = 0; j < Z.nt(); ++j) {
                int64_t ioff = 0;
                for (int64_t i = 0; i < Z.mt(); ++i) {
                    if (Z.tileIsLocal(i, j)) {
                        auto T = Z.at(i, j);
                        for (int jj = 0; jj < T.nb(); ++jj)
                            for (int ii = 0; ii < T.mb(); ++ii)
                                T.at(ii, jj) *= Lambda[ jj + joff ];
                    }
                    ioff += Z.tileMb(i);
                }
                joff += Z.tileNb(j);
            }
            // Z = AC - Z
            slate::hemm(slate::Side::Left, one, A_orig, C, -one, Z, opts);
            // R_norm = | A B Z - Z D | / ( |A| |Z| n )
            R_norm = slate::norm(slate::Norm::One, Z) / A_norm / Z_norm / n;
        }
        else if (itype == 3) {
            // C = AZ + 0*C = AZ
            slate::hemm(slate::Side::Left, one, A_orig, Z, zero, C, opts);
            // Z = ZD
            int64_t joff = 0;
            for (int64_t j = 0; j < Z.nt(); ++j) {
                int64_t ioff = 0;
                for (int64_t i = 0; i < Z.mt(); ++i) {
                    if (Z.tileIsLocal(i, j)) {
                        auto T = Z.at(i, j);
                        for (int jj = 0; jj < T.nb(); ++jj)
                            for (int ii = 0; ii < T.mb(); ++ii)
                                T.at(ii, jj) *= Lambda[ jj + joff ];
                    }
                    ioff += Z.tileMb(i);
                }
                joff += Z.tileNb(j);
            }
            // Z = BC - Z   = ( BAZ - ZD )
            slate::hemm(slate::Side::Left, one, B_orig, C, -one, Z, opts);
            // R_norm = | B A Z - Z D | / ( |A| |Z| n )
            R_norm = slate::norm(slate::Norm::One, Z) / A_norm / Z_norm / n;

        }
        params.error() = R_norm;
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref || check) {
        #ifdef SLATE_HAVE_SCALAPACK
            // Run reference routine from ScaLAPACK

            // initialize BLACS
            int mpi_rank_ = 0, nprocs = 1, ictxt;
            int p_, q_, myrow_, mycol_, info;
            Cblacs_pinfo(&mpi_rank_, &nprocs);
            slate_assert( mpi_rank == mpi_rank_ );
            slate_assert(p*q <= nprocs);
            Cblacs_get(-1, 0, &ictxt);
            Cblacs_gridinit(&ictxt, "Col", p, q);
            Cblacs_gridinfo(ictxt, &p_, &q_, &myrow_, &mycol_);
            slate_assert( p == p_ );
            slate_assert( q == q_ );
            slate_assert( myrow == myrow_ );
            slate_assert( mycol == mycol_ );

            int A_desc[9];
            scalapack_descinit(A_desc, n, n, nb, nb, 0, 0, ictxt, mlocA, &info);
            slate_assert(info == 0);

            int B_desc[9];
            scalapack_descinit(B_desc, n, n, nb, nb, 0, 0, ictxt, mlocB, &info);
            slate_assert(info == 0);

            int Z_desc[9];
            scalapack_descinit(Z_desc, n, n, nb, nb, 0, 0, ictxt, mlocZ, &info);
            slate_assert(info == 0);

            const char* range = "A";
            int64_t vl=0, vu=0, il=0, iu=0;
            real_t abstol=0;
            int64_t nfound=0, nzfound=0;
            real_t orfac=0;

            // query for workspace size
            int64_t info_tst = 0;
            int64_t lwork = -1, lrwork = -1, liwork=-1;
            // For the query, it should be work(3) and rwork(3), since
            // work and rwork have dimension max(3,LWORK) and max(3,LRWORK) in
            // http://www.netlib.org/scalapack/explore-html/db/d5b/pdsygvx_8f_source.html
            // http://www.netlib.org/scalapack/explore-html/d7/dff/pzhegvx_8f_source.html
            std::vector<scalar_t> work(3);
            std::vector<real_t> rwork(3);
            std::vector<int> iwork(1);
            std::vector<int> ifail(n);
            std::vector<int> iclustr(2*p*q);
            std::vector<real_t> gap(p*q);
            scalapack_phegvx(itype, job2str(jobz), range, uplo2str(uplo), n,
                             &Aref_data[0], 1, 1, A_desc,
                             &Bref_data[0], 1, 1, B_desc,
                             vl, vu, il, iu, abstol, &nfound, &nzfound,
                             &Lambda_ref[0], orfac,
                             &Zref_data[0], 1, 1, Z_desc,
                             &work[0], lwork,
                             &rwork[0], lrwork,
                             &iwork[0], liwork,
                             &ifail[0], &iclustr[0], &gap[0], &info_tst);

            // resize workspace based on query for workspace sizes
            slate_assert(info_tst == 0);
            lwork = int64_t(real(work[0]));
            work.resize(lwork);
            // The lrwork, rwork parameters are only valid for complex
            if (slate::is_complex<scalar_t>::value) {
                lrwork = int64_t(real(rwork[0]));
                rwork.resize(lrwork);
            }
            liwork = int64_t(iwork[0]);
            iwork.resize(liwork);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(mpi_comm);

            scalapack_phegvx(itype, job2str(jobz), range, uplo2str(uplo), n,
                             &Aref_data[0], 1, 1, A_desc,
                             &Bref_data[0], 1, 1, B_desc,
                             vl, vu, il, iu, abstol, &nfound, &nzfound,
                             &Lambda_ref[0], orfac,
                             &Zref_data[0], 1, 1, Z_desc,
                             &work[0], lwork,
                             &rwork[0], lrwork,
                             &iwork[0], liwork,
                             &ifail[0], &iclustr[0], &gap[0], &info_tst);

            slate_assert(info_tst == 0);
            time = barrier_get_wtime(mpi_comm) - time;

            params.ref_time() = time;

            if (! ref_only) {
                // Reference Scalapack was run, check reference eigenvalues
                // Perform a local operation to get differences Lambda = Lambda - Lambda_ref
                blas::axpy( n, -1.0, &Lambda_ref[0], 1, &Lambda[0], 1 );
                // Relative forward error: || Lambda_ref - Lambda || / || Lambda_ref ||.
                params.error2() = blas::asum( n, &Lambda[0], 1 )
                                / blas::asum( n, &Lambda_ref[0], 1 );
                real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
                params.okay() = params.okay() && (params.error2() <= tol);
            }
            Cblacs_gridexit(ictxt);
        #else  // not SLATE_HAVE_SCALAPACK
            if (mpi_rank == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_hegv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hegv_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_hegv_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hegv_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hegv_work<std::complex<double>> (params, run);
            break;
    }
}
