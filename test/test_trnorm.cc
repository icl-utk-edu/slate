// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include "grid_utils.hh"
#include "matrix_utils.hh"
#include "test_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_trnorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;

    // get & mark input values
    slate::Norm norm = params.norm();
    slate::Uplo uplo = params.uplo();
    slate::Diag diag = params.diag();
    int64_t m = params.dim.m();
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

    mark_params_for_test_TrapezoidMatrix( params );

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

    auto A_alloc = allocate_test_TrapezoidMatrix<scalar_t>( ref_copy, false, m, n, params );

    auto& A = A_alloc.A;
    auto& Aref = A_alloc.Aref;

    slate::generate_matrix(params.matrix, A);

    if (ref_copy) {
        copy_matrix( A, Aref );
    }

    print_matrix("A", A, params);

    if (trace)
        slate::trace::Trace::on();
    else
        slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    //==================================================
    // Run SLATE test.
    // Compute || A ||_norm.
    //==================================================
    real_t A_norm = slate::norm(norm, A, opts);

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;

    if (trace)
        slate::trace::Trace::finish();

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

        // TODO move the above into this if statement
        if (check || ref) {
            // comparison with reference routine from ScaLAPACK

            // allocate work space
            std::vector<real_t> worklantr(std::max(A_alloc.mloc, A_alloc.nloc));

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            time = barrier_get_wtime(MPI_COMM_WORLD);
            real_t A_norm_ref = scalapack_plantr(
                                    norm2str(norm), uplo2str(A.uplo()), diag2str(diag),
                                    m, n, &A_data[0], 1, 1, A_desc, &worklantr[0]);
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            // difference between norms
            real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
            if (norm == slate::Norm::One) {
                error /= sqrt(m);
            }
            else if (norm == slate::Norm::Inf) {
                error /= sqrt(n);
            }
            else if (norm == slate::Norm::Fro) {
                error /= sqrt(m*n);
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
                std::vector<real_t> worklantr(std::max(A_alloc.mloc, A_alloc.nloc));

                // seed all MPI processes the same
                srand(1234);

                // Test tiles in 2x2 in all 4 corners, and 4 random rows and cols,
                // up to 64 tiles total.
                // Indices may be out-of-bounds if mt or nt is small, so check in loops.
                int64_t mt = A.mt();
                int64_t nt = A.nt();
                std::set<int64_t> i_indices = { 0, 1, mt - 2, mt - 1 };
                std::set<int64_t> j_indices = { 0, 1, nt - 2, nt - 1 };
                for (size_t k = 0; k < 4; ++k) {
                    i_indices.insert(rand() % mt);
                    j_indices.insert(rand() % nt);
                }
                for (auto j : j_indices) {
                    if (j < 0 || j >= nt)
                        continue;
                    int64_t jb = std::min(n - j*nb, nb);
                    slate_assert(jb == A.tileNb(j));

                    for (auto i : i_indices) {
                        // lower requires i >= j
                        // upper requires i <= j
                        if (i < 0 || i >= mt || (uplo == slate::Uplo::Lower ? i < j : i > j))
                            continue;
                        int64_t ib = std::min(m - i*nb, nb);
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
                                    slate_assert(A_data[ ilocal + jlocal*A_alloc.lld ] == save);
                                    T.at(ii, jj) = peak;
                                    A_data[ ilocal + jlocal*A_alloc.lld ] = peak;
                                    // todo: this move shouldn't be required -- the trnorm should copy data itself.
                                    A.tileGetForWriting(i, j, A.tileDevice(i, j), slate::LayoutConvert::ColMajor);
                                }

                                A_norm = slate::norm(norm, A, opts);

                                real_t A_norm_ref = scalapack_plantr(
                                                        norm2str(norm), uplo2str(A.uplo()), diag2str(diag),
                                                        m, n, &A_data[0], 1, 1, A_desc, &worklantr[0]);

                                // difference between norms
                                real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
                                if (norm == slate::Norm::One) {
                                    error /= sqrt(m);
                                }
                                else if (norm == slate::Norm::Inf) {
                                    error /= sqrt(n);
                                }
                                else if (norm == slate::Norm::Fro) {
                                    error /= sqrt(m*n);
                                }

                                // Allow for difference, except max norm in real should be exact.
                                real_t eps = std::numeric_limits<real_t>::epsilon();
                                real_t tol;
                                if (norm == slate::Norm::Max && ! slate::is_complex<scalar_t>::value)
                                    tol = 0;
                                else
                                    tol = 10*eps;

                                if (A.mpiRank() == 0) {
                                    // if peak is nan, expect A_norm to be nan,
                                    // except in Unit case with i == j and ii == jj,
                                    // where peak shouldn't affect A_norm.
                                    bool okay = (std::isnan(real(peak)) && ! (diag == slate::Diag::Unit && i == j && ii == jj)
                                                 ? std::isnan(A_norm)
                                                 : error <= tol);
                                    params.okay() = params.okay() && okay;
                                    if (verbose || ! okay) {
                                        printf("i %5lld, j %5lld, ii %3lld, jj %3lld, peak %15.8e, norm %15.8e, ref %15.8e, error %9.2e, %s\n",
                                               llong( i ), llong( j ), llong( ii ), llong( jj ),
                                               real(peak), A_norm, A_norm_ref, error,
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
    #else  // not SLATE_HAVE_SCALAPACK
        SLATE_UNUSED( A_norm );
        SLATE_UNUSED( check );
        SLATE_UNUSED( ref );
        SLATE_UNUSED( extended );
        SLATE_UNUSED( verbose );
        if (A.mpiRank() == 0)
            printf( "ScaLAPACK not available\n" );
    #endif
}

// -----------------------------------------------------------------------------
void test_trnorm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trnorm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trnorm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trnorm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trnorm_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
