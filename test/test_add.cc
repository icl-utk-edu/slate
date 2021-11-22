// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

// -----------------------------------------------------------------------------
// The following routine : subtract_matrix : takes input matrices A and B,
// assumed to be of the same dimension and performs the operation B = A - B.
// This was developed for checking slate::add without using slate::add to check.
// It is a CPU-only implementation and assumes column-major.
//template <typename scalar_t>
//void subtract_matrix( slate::Matrix<scalar_t>& A, slate::Matrix<scalar_t>& B )

template<typename matrix_type>
void subtract_matrix(matrix_type &A, matrix_type &B)
{
    using scalar_t = typename matrix_type::value_type;
    int64_t mt = A.mt();
    int64_t nt = A.nt();
    if constexpr (std::is_same<matrix_type, slate::Matrix<scalar_t>>::value) {
        #pragma omp parallel for collapse(2)
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal( i, j )) {
                    A.tileGetForReading( i, j, slate::LayoutConvert::None );
                    B.tileGetForWriting( i, j, slate::LayoutConvert::None );
                    auto TA = A( i, j );
                    auto TB = B( i, j );
                    int64_t mb = TA.mb();
                    int64_t nb = TA.nb();
                    int64_t lda = TA.stride();
                    int64_t ldb = TB.stride();
                    scalar_t const* TA_data = TA.data();
                    scalar_t*       TB_data = TB.data();
                    for (int64_t jj = 0; jj < nb; ++jj) {
                        for (int64_t ii = 0; ii < mb; ++ii) {
                            TB_data[ ii + jj*ldb ] -= TA_data[ ii + jj*lda ];
                        }
                    }
                }
            }
        }
    }
    else {
        if (A.uploPhysical() == slate::Uplo::Upper) {
            for (int64_t i = 0; i < mt; ++i) {
                for (int64_t j = i; j < nt; ++j) {
                    if (A.tileIsLocal( i, j )) {
                        auto Aij = A(i,j);
                        if (Aij.uploPhysical() == slate::Uplo::Upper) {
                            A.tileGetForReading( i, j, slate::LayoutConvert::None );
                            B.tileGetForWriting( i, j, slate::LayoutConvert::None );
                            auto TA = A( i, j );
                            auto TB = B( i, j );
                            int64_t mb = TA.mb();
                            int64_t nb = TA.nb();
                            int64_t lda = TA.stride();
                            int64_t ldb = TB.stride();
                            scalar_t const* TA_data = TA.data();
                            scalar_t*       TB_data = TB.data();
                            for (int64_t ii = 0; ii < mb; ++ii) {
                                for (int64_t jj = ii; jj < nb; ++jj) {
                                        TB_data[ ii + jj*ldb ] -= TA_data[ ii + jj*lda ];
                                }
                            }
                        }
                        else {
                            A.tileGetForReading( i, j, slate::LayoutConvert::None );
                            B.tileGetForWriting( i, j, slate::LayoutConvert::None );
                            auto TA = A( i, j );
                            auto TB = B( i, j );
                            int64_t mb = TA.mb();
                            int64_t nb = TA.nb();
                            int64_t lda = TA.stride();
                            int64_t ldb = TB.stride();
                            scalar_t const* TA_data = TA.data();
                            scalar_t*       TB_data = TB.data();
                            for (int64_t jj = 0; jj < nb; ++jj) {
                                for (int64_t ii = 0; ii < mb; ++ii) {
                                    TB_data[ ii + jj*ldb ] -= TA_data[ ii + jj*lda ];
                                }
                            }
                        }
                    }
                }
            }
        } else if (A.uploPhysical() == slate::Uplo::Lower) {
            for (int64_t j = 0; j < nt; ++j) {
                for (int64_t i = j; i < mt; ++i) {
                    if (A.tileIsLocal( i, j )) {
                        auto Aij = A(i,j);
                        if (Aij.uploPhysical() == slate::Uplo::Lower) {
                            A.tileGetForReading( i, j, slate::LayoutConvert::None );
                            B.tileGetForWriting( i, j, slate::LayoutConvert::None );
                            auto TA = A( i, j );
                            auto TB = B( i, j );
                            int64_t mb = TA.mb();
                            int64_t nb = TA.nb();
                            int64_t lda = TA.stride();
                            int64_t ldb = TB.stride();
                            scalar_t const* TA_data = TA.data();
                            scalar_t*       TB_data = TB.data();
                            for (int64_t jj = 0; jj < nb; ++jj) {
                                for (int64_t ii = jj; ii < mb; ++ii) {
                                    TB_data[ ii + jj*ldb ] -= TA_data[ ii + jj*lda ];
                                }
                            }
                        }
                        else {
                            A.tileGetForReading( i, j, slate::LayoutConvert::None );
                            B.tileGetForWriting( i, j, slate::LayoutConvert::None );
                            auto TA = A( i, j );
                            auto TB = B( i, j );
                            int64_t mb = TA.mb();
                            int64_t nb = TA.nb();
                            int64_t lda = TA.stride();
                            int64_t ldb = TB.stride();
                            scalar_t const* TA_data = TA.data();
                            scalar_t*       TB_data = TB.data();
                            for (int64_t jj = 0; jj < nb; ++jj) {
                                for (int64_t ii = 0; ii < mb; ++ii) {
                                    TB_data[ ii + jj*ldb ] -= TA_data[ ii + jj*lda ];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#define SLATE_HAVE_SCALAPACK
//------------------------------------------------------------------------------
template<typename matrix_type>
void test_add_work(Params& params, bool run)
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;

    // get & mark input values
    //
    // if transposition operations are included in add.cc,
    // the following slate::Op should change.
    // slate::Op transA = params.trans();
    slate::Op transA = slate::Op::NoTrans;
    scalar_t alpha = params.alpha.get<real_t>();
    scalar_t beta = params.beta.get<real_t>();
    int64_t n = params.dim.n();
    int64_t m;
    if constexpr (std::is_same<matrix_type, slate::TriangularMatrix<scalar_t>>::value ||
                  std::is_same<matrix_type, slate::SymmetricMatrix<scalar_t>>::value ||
                  std::is_same<matrix_type, slate::HermitianMatrix<scalar_t>>::value)
        m = n;
    else
        m = params.dim.m();
    int64_t nb = params.nb();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::Uplo uplo;
    if constexpr (std::is_same<matrix_type, slate::Matrix<scalar_t>>::value)
        SLATE_UNUSED( uplo );
    else
        uplo = params.uplo();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    // MPI variables
    int mpi_rank, myrow, mycol;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    gridinfo(mpi_rank, p, q, &myrow, &mycol);

    // Matrix A: figure out local size.
    int64_t mlocA = num_local_rows_cols(m, nb, myrow, p);
    int64_t nlocA = num_local_rows_cols(n, nb, mycol, q);
    int64_t lldA  = blas::max(1, mlocA); // local leading dimension of A
    std::vector<scalar_t> A_data,B_data;

    matrix_type A,B;
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target(origin);
        if constexpr (std::is_same<matrix_type, slate::Matrix<scalar_t>>::value) {
            // General mxn matrix
            A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::TrapezoidMatrix<scalar_t>>::value) {
            // Trapezoidal mxn matrix
            A = slate::TrapezoidMatrix<scalar_t>(uplo, slate::Diag::NonUnit, m, n, nb, p, q, MPI_COMM_WORLD);
            B = slate::TrapezoidMatrix<scalar_t>(uplo, slate::Diag::NonUnit, m, n, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::TriangularMatrix<scalar_t>>::value) {
            // Triangular nxn matrix
            A = slate::TriangularMatrix<scalar_t>(uplo, slate::Diag::NonUnit, n, nb, p, q, MPI_COMM_WORLD);
            B = slate::TriangularMatrix<scalar_t>(uplo, slate::Diag::NonUnit, n, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::SymmetricMatrix<scalar_t>>::value) {
            // Symmetric nxn matrix
            A = slate::SymmetricMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
            B = slate::SymmetricMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::HermitianMatrix<scalar_t>>::value) {
            // Hermitian nxn matrix
            A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
            B = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        }
        else {
            throw slate::Exception("unknown routine: not compatible with add");
        }
        A.insertLocalTiles(origin_target);
        B.insertLocalTiles(origin_target);
    }
    else {
         // Allocate necessary space here to avoid using 2x memory
         A_data.resize( lldA*nlocA );
         B_data.resize( lldA*nlocA );
        // Create SLATE matrix from the ScaLAPACK layout.
        if constexpr (std::is_same<matrix_type, slate::Matrix<scalar_t>>::value) {
            // General mxn matrix
            A = slate::Matrix<scalar_t>::fromScaLAPACK(
                    m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>::fromScaLAPACK(
                    m, n, &B_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::TrapezoidMatrix<scalar_t>>::value) {
            // Trapezoidal mxn matrix
            A = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(
                    uplo, slate::Diag::NonUnit, m, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            B = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(
                    uplo, slate::Diag::NonUnit, m, n, &B_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::TriangularMatrix<scalar_t>>::value) {
            // Triangular nxn matrix
            A = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
                    uplo, slate::Diag::NonUnit, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            B = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
                    uplo, slate::Diag::NonUnit, n, &B_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::SymmetricMatrix<scalar_t>>::value) {
            // Symmetric nxn matrix
            A = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            B = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, &B_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::HermitianMatrix<scalar_t>>::value) {
            // Hermitian nxn matrix
            A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, &A_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            B = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, &B_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else {
            throw slate::Exception("unknown routine: not compatible with add");
        }
    }

    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrix, B);

    // if reference run is required, copy test data
    matrix_type Aref,Bref;
    std::vector<scalar_t> Aref_data,Bref_data;
    if (check || ref) {
        // For simplicity, always use ScaLAPACK format for ref matrices.
        Aref_data.resize( lldA*nlocA );
        Bref_data.resize( lldA*nlocA );
        if constexpr (std::is_same<matrix_type, slate::Matrix<scalar_t>>::value) {
            // General mxn matrix
            Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
                       m,  n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                      m,  n, &Bref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::TrapezoidMatrix<scalar_t>>::value) {
            // Trapezoidal mxn matrix
            Aref = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(
                       uplo, slate::Diag::NonUnit, m,  n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            Bref = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(
                       uplo, slate::Diag::NonUnit, m,  n, &Bref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::TriangularMatrix<scalar_t>>::value) {
            // Triangular nxn matrix
            Aref = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
                       uplo, slate::Diag::NonUnit,  n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            Bref = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
                       uplo, slate::Diag::NonUnit,  n, &Bref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::SymmetricMatrix<scalar_t>>::value) {
            // Symmetric nxn matrix
            Aref = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(
                       uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            Bref = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(
                       uplo, n, &Bref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else if constexpr (std::is_same<matrix_type, slate::HermitianMatrix<scalar_t>>::value) {
             // Hermitian nxn matrix
            Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                       uplo, n, &Aref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
            Bref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                       uplo, n, &Bref_data[0], lldA, nb, p, q, MPI_COMM_WORLD);
        }
        else {
            throw slate::Exception("unknown routine: not compatible with add");
        }
        slate::copy(A, Aref);
        slate::copy(B, Bref);
    }

/*
    if (trans == slate::Op::Trans)
        A = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        A = conjTranspose(A);
*/

    if (verbose >= 1) {
        print_matrix("A", A, params);
        print_matrix("B", B, params);
    }

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test.
        // Add alpha A by beta B into B.
        //==================================================
        double time = barrier_get_wtime(MPI_COMM_WORLD);

        slate::add(alpha, A, beta, B, opts);

        time = barrier_get_wtime(MPI_COMM_WORLD) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // BLACS/MPI variables
            int ictxt, p_, q_, myrow_, mycol_, info;
            int A_desc[9], B_desc[9];
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

            scalapack_descinit(A_desc, m, n, nb, nb, 0, 0, ictxt, lldA, &info);
            scalapack_descinit(B_desc, m, n, nb, nb, 0, 0, ictxt, lldA, &info);
            slate_assert(info == 0);

            if (origin == slate::Origin::ScaLAPACK) {
                // todo: the copy needs to be fixed for transpose case.
                copy(A, &A_data[0], A_desc);
                copy(B, &B_data[0], B_desc);
            }
            real_t A_norm = slate::norm(slate::Norm::One, A);
            real_t B_norm = slate::norm(slate::Norm::One, B);
            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime(MPI_COMM_WORLD);

            scalapack_pgeadd(op2str(transA), m, n,
                             alpha, &Aref_data[0], 1, 1, A_desc,
                             beta,  &Bref_data[0], 1, 1, B_desc, &info);

            slate_assert(info == 0);

            time = barrier_get_wtime(MPI_COMM_WORLD) - time;

            if (verbose >= 1) {
                print_matrix("Aref", mlocA, nlocA, &Aref_data[0], lldA, p, q, MPI_COMM_WORLD);
                print_matrix("Bref", mlocA, nlocA, &Bref_data[0], lldA, p, q, MPI_COMM_WORLD);
            }

            // get differences A = A - Aref
            subtract_matrix(Aref,A);

            // get differences B = B - Bref
            subtract_matrix(Bref,B);

            if (verbose >= 1) {
                print_matrix("DiffA", A, params);
                print_matrix("DiffB", B, params);
            }

            // norm(A - Aref)
            real_t A_diff_norm = slate::norm(slate::Norm::One, A);
            // norm(B - Bref)
            real_t B_diff_norm = slate::norm(slate::Norm::One, B);

            params.ref_time() = time;

            real_t errorA = A_diff_norm / (n * A_norm);
            real_t errorB = B_diff_norm / (n * B_norm);

            params.error() = errorA + errorB;

            slate_set_num_blas_threads(saved_num_threads);

            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
            // Allow for difference
            params.okay() = (params.error() <= tol);

            Cblacs_gridexit(ictxt);
            //Cblacs_exit(1) does not handle re-entering
        #endif
    }
}

// -----------------------------------------------------------------------------
template <typename scalar_t >
void test_add_dispatch(Params& params, bool run )
{
    std::string routine = params.routine;
    if (routine == "add") {
        test_add_work< slate::Matrix<scalar_t> >( params, run );
    }
    else if (routine == "tzadd") {
        test_add_work< slate::TrapezoidMatrix<scalar_t> >( params, run );
    }
    else if (routine == "tradd") {
        test_add_work< slate::TriangularMatrix<scalar_t> >( params, run );
    }
    else if (routine == "syadd") {
        test_add_work< slate::SymmetricMatrix<scalar_t> >( params, run );
    }
    else if (routine == "headd") {
        test_add_work< slate::HermitianMatrix<scalar_t> >( params, run );
    }
    else {
        throw slate::Exception("unknown routine: " + routine);
    }
}

// -----------------------------------------------------------------------------
void test_add(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_add_dispatch<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_add_dispatch<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_add_dispatch<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_add_dispatch<std::complex<double>> (params, run);
            break;
    }
}
