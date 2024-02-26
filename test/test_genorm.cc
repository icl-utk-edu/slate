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
void test_genorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;

    // get & mark input values
    slate::Norm norm = params.norm();
    slate::NormScope scope = params.scope();
    slate::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    bool nonuniform_nb = params.nonuniform_nb() == 'y';
    bool ref_copy = nonuniform_nb && (check || ref);
    int verbose = params.verbose();
    int extended = params.extended();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::GridOrder grid_order = params.grid_order();
    params.matrix.mark();

    mark_params_for_test_Matrix( params );

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

    auto A_alloc = allocate_test_Matrix<scalar_t>( ref_copy, false, m, n, params );

    auto& A = A_alloc.A;
    auto& Aref = A_alloc.Aref;

    slate::generate_matrix(params.matrix, A);

    if (ref_copy) {
        copy_matrix( A, Aref );
    }

    std::vector<real_t> values;
    if (scope == slate::NormScope::Columns) {
        values.resize(A.n());
    }
    else if (scope == slate::NormScope::Rows) {
        values.resize(A.m());
    }

    if (trans == slate::Op::Trans)
        A = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        A = conj_transpose( A );

    print_matrix("A", A, params);

    real_t A_norm = 0;
    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test.
        // Compute || A ||_norm.
        //==================================================
        double time = barrier_get_wtime(MPI_COMM_WORLD);

        if (scope == slate::NormScope::Matrix) {
            A_norm = slate::norm(norm, A, opts);
        }
        else if (scope == slate::NormScope::Columns) {
            slate::colNorms(norm, A, values.data(), opts);
        }
        else if (scope == slate::NormScope::Rows) {
            slate_error("Not implemented yet");
            // slate::rowNorms(norm, A, values.data(), opts);
        }

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
    }

    if (check || ref) {
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

            // allocate work space
            std::vector<real_t> worklange(std::max(A_alloc.mloc, A_alloc.nloc));

            // (Sca)LAPACK norms don't support trans; map One <=> Inf norm.
            slate::Norm op_norm = norm;
            if (trans == slate::Op::Trans || trans == slate::Op::ConjTrans) {
                if (norm == slate::Norm::One)
                    op_norm = slate::Norm::Inf;
                else if (norm == slate::Norm::Inf)
                    op_norm = slate::Norm::One;
            }

            // difference between norms
            real_t error = 0.;
            real_t A_norm_ref = 0;

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);
            if (scope == slate::NormScope::Matrix) {
                A_norm_ref = scalapack_plange(
                                 norm2str(op_norm),
                                 m, n, &A_data[0], 1, 1, A_desc, &worklange[0]);
            }
            else if (scope == slate::NormScope::Columns) {
                for (int64_t c = 0; c < n; ++c) {
                    A_norm_ref = scalapack_plange(
                                     norm2str(norm),
                                     m, 1, &A_data[0], 1, c+1, A_desc, &worklange[0]);
                    error += std::abs(values[c] - A_norm_ref) / A_norm_ref;
                }
            }
            else if (scope == slate::NormScope::Rows) {
                // todo
            }
            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            if (scope == slate::NormScope::Matrix) {
                // difference between norms
                error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
                if (op_norm == slate::Norm::One) {
                    error /= sqrt(m);
                }
                else if (op_norm == slate::Norm::Inf) {
                    error /= sqrt(n);
                }
                else if (op_norm == slate::Norm::Fro) {
                    error /= sqrt(m*n);
                }

                if (verbose && A.mpiRank() == 0) {
                    printf("norm %15.8e, ref %15.8e, ref - norm %5.2f, error %9.2e\n",
                           A_norm, A_norm_ref, A_norm_ref - A_norm, error);
                }
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

            //---------- extended tests
            if (extended && scope == slate::NormScope::Matrix) {
                if (grid_order != slate::GridOrder::Col) {
                    printf("WARNING: cannot do extended tests with row-major grid\n");
                }
                else {

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
                            if (i < 0 || i >= mt)
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
                                    if (ii < 0 || ii >= ib) {
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
                                        // todo: this move shouldn't be required -- the genorm should copy data itself.
                                        A.tileGetForWriting(i, j, A.tileDevice(i, j), slate::LayoutConvert::ColMajor);
                                    }

                                    A_norm = slate::norm(norm, A, opts);

                                    A_norm_ref = scalapack_plange(
                                                     norm2str(norm), m, n,
                                                     &A_data[0], 1, 1, A_desc,
                                                     &worklange[0]);

                                    // difference between norms
                                    error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
                                    if (norm == slate::Norm::One) {
                                        error /= sqrt(m);
                                    }
                                    else if (norm == slate::Norm::Inf) {
                                        error /= sqrt(n);
                                    }
                                    else if (norm == slate::Norm::Fro) {
                                        error /= sqrt(m*n);
                                    }

                                    if (A.mpiRank() == 0) {
                                        // if peak is nan, expect A_norm to be nan.
                                        bool okay = (std::isnan(real(peak))
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
                                        // todo: this move shouldn't be required -- the genorm should copy data itself.
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
            SLATE_UNUSED( extended );
            SLATE_UNUSED( verbose );
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

// -----------------------------------------------------------------------------
void test_genorm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_genorm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_genorm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_genorm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_genorm_work<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
