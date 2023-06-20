// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
#include "scalapack_support_routines.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_tb2bd_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // get & mark input values
    lapack::Job jobu = params.jobu();
    lapack::Job jobvt = params.jobvt();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t ku = nb;  // upper band; for now use ku == nb.
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    int64_t info;

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho_U();
    params.ortho_V();
    params.error2();
    params.error.name( "S - Sref" );
    params.error2.name( "off-diag" );

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target, target},
    };

    bool wantu  = (jobu  == slate::Job::Vec
                   || jobu  == slate::Job::AllVec
                   || jobu  == slate::Job::SomeVec);
    bool wantvt = (jobvt == slate::Job::Vec
                   || jobvt == slate::Job::AllVec
                   || jobvt == slate::Job::SomeVec);

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    int64_t lda = n;
    int64_t seed[] = {0, 1, 2, 3};

    slate::Target origin_target = origin2target(origin);

    std::vector<scalar_t> Afull_data( lda*n );
    lapack::larnv(1, seed, Afull_data.size(), &Afull_data[0]);

    // Zero outside the upper band.
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) {
            if (j > i+ku || j < i)
                Afull_data[i + j*lda] = 0;
        }
        // Diagonal from ge2tb is real.
        Afull_data[j + j*lda] = real( Afull_data[j + j*lda] );
    }
    if (mpi_rank == 0) {
        print_matrix( "Afull_data", n, n, &Afull_data[0], lda, params );
    }

    auto Afull = slate::Matrix<scalar_t>::fromLAPACK(
        n, n, &Afull_data[0], lda, nb, p, q, MPI_COMM_WORLD);

    // Copy band of Afull, currently to rank 0.
    auto Aband = slate::TriangularBandMatrix<scalar_t>(
        slate::Uplo::Upper, slate::Diag::NonUnit, n, ku, nb,
        1, 1, MPI_COMM_WORLD);
    Aband.insertLocalTiles();
    Aband.ge2tbGather( Afull );

    if (verbose) {
        print_matrix("Aband", Aband, params);
    }

    slate::Matrix<scalar_t> U, U1d, VT, V1d;
    // Create U and U1d. Set U to Identity.
    if (wantu) {
        U = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        U.insertLocalTiles(origin_target);
        set(zero, one, U);
        U1d = slate::Matrix<scalar_t>(U.m(), U.n(), U.tileNb(0), 1, p*q, MPI_COMM_WORLD);
        U1d.insertLocalTiles(origin_target);
    }

    // Create VT and V1d. Set VT to Identity.
    if (wantvt) {
        VT = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);
        VT.insertLocalTiles(origin_target);
        set(zero, one, VT);
        // 1-d V matrix
        V1d = slate::Matrix<scalar_t>(VT.m(), VT.n(), VT.tileNb(0), 1, p*q, MPI_COMM_WORLD);
        V1d.insertLocalTiles(origin_target);
    }

    // Singular values.
    std::vector<real_t> Sigma_ref(n);

    // Create U2 and V2 needed for tb2bd.
    int64_t vm = 2*nb;
    int64_t nt = Afull.nt();
    int64_t vn = nt*(nt + 1)/2*nb;
    slate::Matrix<scalar_t> V2(vm, vn, vm, nb, 1, 1, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> U2(vm, vn, vm, nb, 1, 1, MPI_COMM_WORLD);

    if (check && mpi_rank == 0) {
        //==================================================
        // For checking results, compute SVD of original matrix A.
        //==================================================
        std::vector<scalar_t> Afull_copy = Afull_data;
        scalar_t dummy[1];  // U, VT not needed for NoVec
        info = lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec,
                             n, n, &Afull_copy[0], lda, &Sigma_ref[0],
                             dummy, 1, dummy, 1);
        assert(info == 0);
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Currently runs only on rank 0.
    //==================================================
    if (mpi_rank == 0) {
        V2.insertLocalTiles();
        U2.insertLocalTiles();
        slate::tb2bd(Aband, U2, V2);
    }

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;
    params.time() = time;

    if (trace) slate::trace::Trace::finish();

    //==================================================
    // Back transform U and VT of the band matrix..
    //==================================================
    if (wantu) {
        slate::redistribute(U, U1d, opts);
        slate::unmtr_hb2st(slate::Side::Left, slate::Op::NoTrans, U2, U1d, opts);
        slate::redistribute(U1d, U, opts);
    }
    if (wantvt) {
        slate::redistribute(VT, V1d, opts);
        slate::unmtr_hb2st(slate::Side::Left, slate::Op::NoTrans, V2, V1d, opts);
        slate::redistribute(V1d, VT, opts);
    }

    if (check) {
        //==================================================
        // Test results
        // Gather the whole matrix onto rank 0.
        //==================================================
        // Copy Aband back to Afull_data on rank 0.
        Aband.gather(&Afull_data[0], lda);

        std::vector<real_t> Sigma(1);
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        if (mpi_rank == 0) {
            print_matrix( "Afull_data_out", n, n, &Afull_data[0], lda, params );

            // Check that updated Aband is real bidiagonal by finding max value
            // outside bidiagonal, and imaginary parts of bidiagonal.
            // Unclear why this increases modestly with n.
            real_t max_value = 0;
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < n; ++i) {
                    auto val = Afull_data[i + j*lda];
                    if (j > i+1 || j < i)
                        max_value = std::max( std::abs(val), max_value );
                    else
                        max_value = std::max( std::abs(imag(val)), max_value );
                }
            }
            params.error2() = max_value / sqrt(n);

            // Check that the singular values of updated Aband
            // match the singular values of the original Aband.
            Sigma.resize(n);
            std::vector<real_t> E(n - 1);  // super-diagonal
            scalar_t dummy[1];  // U, VT, C not needed for NoVec

            // Copy diagonal & super-diagonal.
            // todo: we can use the following three lines to copy E and D to all ranks.:
            //slate::internal::copytb2bd(Aband, Sigma, E);
            //MPI_Bcast( &Sigma[0], n, mpi_real_type, 0, MPI_COMM_WORLD );
            //MPI_Bcast( &E[0], n-1, mpi_real_type, 0, MPI_COMM_WORLD );
            int64_t D_index = 0;
            int64_t E_index = 0;
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

            print_matrix("D", 1, n,   &Sigma[0], 1, params);
            print_matrix("E", 1, n-1, &E[0],  1, params);

            info = lapack::bdsqr(lapack::Uplo::Upper, n, 0, 0, 0,
                                 &Sigma[0], &E[0], dummy, 1, dummy, 1, dummy, 1);
            assert(info == 0);

            if (verbose) {
                printf( "%% first and last 20 rows of Sigma_ref and Sigma\n" );
                printf( "%9s  %9s\n", "Sigma_ref", "Sigma" );
                for (int64_t i = 0; i < std::min(n, n); ++i) {
                    if (i < 20 || i >= std::min(n, n)-20) {
                        bool okay = std::abs( Sigma_ref[i] - Sigma[i] ) < tol;
                        printf( "%9.6f  %9.6f%s\n",
                                Sigma_ref[i], Sigma[i], (okay ? "" : " !!") );
                    }
                    else if (i == 20) {
                        printf( "--------------------\n" );
                    }
                }
                printf( "\n" );
            }

            // Relative forward error: || Sigma - Sigma_ref || / || Sigma_ref ||.
            blas::axpy(Sigma.size(), -1.0, &Sigma_ref[0], 1, &Sigma[0], 1);
            params.error() = blas::nrm2(Sigma.size(), &Sigma[0], 1)
                           / blas::nrm2(Sigma_ref.size(), &Sigma_ref[0], 1);
            params.okay() = (params.error() <= tol && params.error2() <= tol);
        }
        if (wantu || wantvt) {
            //==================================================
            // Test results orthogonality of U.
            // || I - U^H U || / n < tol
            //==================================================
            slate::Matrix<scalar_t> Iden( n, n, nb, p, q, MPI_COMM_WORLD );
            Iden.insertLocalTiles(origin_target);
            set(zero, one, Iden);
            if (wantu) {
                set(zero, one, Iden);
                auto UH = conj_transpose( U );
                slate::gemm( -one, UH, U, one, Iden );
                params.ortho_U() = slate::norm( slate::Norm::One, Iden ) / n;
                params.okay() = params.okay() && (params.ortho_U() <= tol);
            }
            //==================================================
            // Test results orthogonality of VT.
            // || I - VT^H VT || / n < tol
            //==================================================
            if (wantvt) {
                set(zero, one, Iden);
                auto V = conj_transpose(VT);
                slate::gemm(-one, V, VT, one, Iden);
                params.ortho_V() = slate::norm(slate::Norm::One, Iden) / n;
                params.okay() = params.okay() && (params.ortho_V() <= tol);
            }
        }
    }
}

// -----------------------------------------------------------------------------
void test_tb2bd(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tb2bd_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_tb2bd_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_tb2bd_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tb2bd_work<std::complex<double>> (params, run);
            break;
    }
}
