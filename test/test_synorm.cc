// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include "matrix_utils.hh"
#include "test_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_synorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;

    // get & mark input values
    slate::Norm norm = params.norm();
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int p = params.grid.m();
    int q = params.grid.n();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    bool nonuniform_nb = params.nonuniform_nb() == 'y';
    bool ref_copy = nonuniform_nb && (check || ref);
    int verbose = params.verbose();
    int extended = params.extended();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::GridOrder grid_order = params.grid_order();
    params.matrix.mark();

    mark_params_for_test_SymmetricMatrix( params );

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    // Check for common invalid combinations
    if (is_invalid_parameters( params, true )) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    auto A_alloc = allocate_test_SymmetricMatrix<scalar_t>( ref_copy, false, n, params );

    auto& A = A_alloc.A;
    auto& Aref = A_alloc.Aref;

    slate::generate_matrix(params.matrix, A);

    if (ref_copy) {
        copy_matrix( A, Aref );
    }

    print_matrix("A", A, params);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Compute || A ||_norm.
    //==================================================
    real_t A_norm = slate::norm(norm, A, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time;

    #ifdef SLATE_HAVE_SCALAPACK
        // comparison with reference routine from ScaLAPACK

        // initialize BLACS and ScaLAPACK
        blas_int ictxt, A_desc[9];
        A_alloc.create_ScaLAPACK_context( &ictxt );
        A_alloc.ScaLAPACK_descriptor( ictxt, A_desc );

        auto& A_data = ref_copy ? A_alloc.Aref_data : A_alloc.A_data;

        if (origin != slate::Origin::ScaLAPACK && !ref_copy) {
            A_data.resize( A_alloc.lld * A_alloc.nloc );

            copy(A, &A_data[0], A_desc);
        }

        if (check || ref) {
            // comparison with reference routine from ScaLAPACK

            // allocate work space
            int64_t ldw = nb*ceildiv( ceildiv( A_alloc.nloc, nb ),
                                      scalapack_ilcm( p, q ) / p );
            int64_t lwork = 2*A_alloc.mloc + A_alloc.nloc + ldw;
            std::vector<real_t> worklansy(lwork);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            real_t A_norm_ref = scalapack_plansy(
                                    norm2str(norm), uplo2str(A.uplo()),
                                    n, &A_data[0], 1, 1, A_desc, &worklansy[0]);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            // difference between norms
            real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
            if (norm == slate::Norm::One || norm == slate::Norm::Inf) {
                error /= sqrt(n);
            }
            else if (norm == slate::Norm::Fro) {
                error /= n;  // = sqrt( n*n );
            }

            if (verbose && A.mpiRank() == 0) {
                printf("norm %15.8e, ref %15.8e, ref - norm %5.2f, error %9.2e\n",
                       A_norm, A_norm_ref, A_norm_ref - A_norm, error);
            }

            // Allow for difference, except max norm in real should be exact.
            real_t eps = std::numeric_limits<real_t>::epsilon();
            real_t tol;
            if (norm == slate::Norm::Max && ! slate::is_complex<scalar_t>::value)
                tol = 0;
            else
                tol = 10*eps;

            params.ref_time() = time;
            params.error() = error;

            // Allow for difference
            params.okay() = (params.error() <= tol);
        }

        //---------- extended tests
        if (extended) {
            if (grid_order != slate::GridOrder::Col) {
                printf("WARNING: cannot do extended tests with row-major grid\n");
            }
            else {
                // allocate work space
                int64_t ldw = nb*ceildiv( ceildiv( A_alloc.nloc, nb ),
                                          scalapack_ilcm( p, q ) / p );
                int64_t lwork = 2*A_alloc.mloc + A_alloc.nloc + ldw;
                std::vector<real_t> worklansy(lwork);

                // seed all MPI processes the same
                srand(1234);

                // Test tiles in 2x2 in all 4 corners, and 4 random rows and cols,
                // up to 64 tiles total.
                // Indices may be out-of-bounds if nt is small, so check in loops.
                int64_t nt = A.nt();
                std::set<int64_t> j_indices = { 0, 1, nt - 2, nt - 1 };
                for (size_t k = 0; k < 4; ++k) {
                    j_indices.insert(rand() % nt);
                }
                for (auto j : j_indices) {
                    if (j < 0 || j >= nt)
                        continue;
                    int64_t jb = std::min(n - j*nb, nb);
                    slate_assert(jb == A.tileNb(j));

                    for (auto i : j_indices) {
                        // lower requires i >= j
                        // upper requires i <= j
                        if (i < 0 || i >= nt || (uplo == slate::Uplo::Lower ? i < j : i > j))
                            continue;
                        int64_t ib = std::min(n - i*nb, nb);
                        slate_assert(ib == A.tileMb(i));

                        // Test entries in 2x2 in all 4 corners, and 1 other random row and col,
                        // up to 25 entries per tile.
                        // Indices may be out-of-bounds if ib or jb is small, so check in loops.
                        std::set<int64_t> ii_indices = { 0, 1, ib - 2, ib - 1, rand() % ib };
                        std::set<int64_t> jj_indices = { 0, 1, jb - 2, jb - 1, rand() % jb };

                        // todo: complex peak
                        scalar_t peak = rand() / double(RAND_MAX)*1e6 + 1e6;
                        if (rand() < RAND_MAX / 2)
                            peak *= -1;
                        if (rand() < RAND_MAX / 20)
                            peak = nan("");
                        scalar_t save = 0;

                        for (auto jj : jj_indices) {
                            if (jj < 0 || jj >= jb)
                                continue;

                            for (auto ii : ii_indices) {
                                if (ii < 0 || ii >= ib
                                    || (i == j && (uplo == slate::Uplo::Lower
                                                   ? ii < jj
                                                   : ii > jj))) {
                                    continue;
                                }

                                int64_t ilocal = int(i / p)*nb + ii;
                                int64_t jlocal = int(j / q)*nb + jj;
                                if (A.tileIsLocal(i, j)) {
                                    A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
                                    auto T = A(i, j);
                                    save = T(ii, jj);
                                    T.at(ii, jj) = peak;
                                    A_data[ ilocal + jlocal*A_alloc.lld ] = peak;
                                    // todo: this move shouldn't be required -- the trnorm should copy data itself.
                                    A.tileGetForWriting(i, j, A.tileDevice(i, j), slate::LayoutConvert::ColMajor);
                                }

                                A_norm = slate::norm(norm, A, opts);

                                real_t A_norm_ref = scalapack_plansy(
                                                        norm2str(norm), uplo2str(A.uplo()),
                                                        n, &A_data[0], 1, 1, A_desc, &worklansy[0]);

                                // difference between norms
                                real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
                                if (norm == slate::Norm::One || norm == slate::Norm::Inf) {
                                    error /= sqrt(n);
                                }
                                else if (norm == slate::Norm::Fro) {
                                    error /= sqrt(n*n);
                                }

                                // Allow for difference, except max norm in real should be exact.
                                real_t eps = std::numeric_limits<real_t>::epsilon();
                                real_t tol;
                                if (norm == slate::Norm::Max && ! slate::is_complex<scalar_t>::value)
                                    tol = 0;
                                else
                                    tol = 10*eps;

                                if (A.mpiRank() == 0) {
                                    // if peak is nan, expect A_norm to be nan.
                                    bool okay = (std::isnan(real(peak))
                                                 ? std::isnan(A_norm)
                                                 : error <= tol);
                                    params.okay() = params.okay() && okay;
                                    if (verbose || ! okay) {
                                        printf("i %5lld, j %5lld, ii %3lld, jj %3lld, peak %15.8e, norm %15.8e, ref %15.8e, error %9.2e, %s\n",
                                               llong( i ), llong( j ), llong( ii ), llong( jj ),
                                               real( peak ), A_norm, A_norm_ref, error,
                                               (okay ? "pass" : "failed"));
                                    }
                                }

                                if (A.tileIsLocal(i, j)) {
                                    A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
                                    auto T = A(i, j);
                                    T.at(ii, jj) = save;
                                    A_data[ ilocal + jlocal*A_alloc.lld ] = save;
                                    // todo: this move shouldn't be required -- the trnorm should copy data itself.
                                    A.tileGetForWriting(i, j, A.tileDevice(i, j), slate::LayoutConvert::ColMajor);
                                }
                            }
                        }
                    }
                }
            }
        }
        Cblacs_gridexit(ictxt);
        //Cblacs_exit(1) does not handle re-entering
    #else // not SLATE_HAVE_SCALAPACK
        SLATE_UNUSED( A_norm );
        SLATE_UNUSED( check );
        SLATE_UNUSED( ref );
        SLATE_UNUSED( extended );
        SLATE_UNUSED( verbose );
        if ((check || ref) && A.mpiRank() == 0)
            printf( "ScaLAPACK not available\n" );
    #endif
}

// -----------------------------------------------------------------------------
void test_synorm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_synorm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_synorm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_synorm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_synorm_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
