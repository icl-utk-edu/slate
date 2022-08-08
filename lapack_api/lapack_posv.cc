// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack_slate.hh"

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------

// Local function
template <typename scalar_t>
void slate_posv(const char* uplostr, const int n, const int nrhs, scalar_t* a, const int lda, scalar_t* b, const int ldb, int* info);

using lld = long long int;

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sposv BLAS_FORTRAN_NAME( slate_sposv, SLATE_SPOSV )
#define slate_dposv BLAS_FORTRAN_NAME( slate_dposv, SLATE_DPOSV )
#define slate_cposv BLAS_FORTRAN_NAME( slate_cposv, SLATE_CPOSV )
#define slate_zposv BLAS_FORTRAN_NAME( slate_zposv, SLATE_ZPOSV )

extern "C" void slate_sposv(const char* uplo, const int* n, const int* nrhs, float* a, const int* lda, float* b, const int* ldb, int* info)
{
    slate_posv(uplo, *n, *nrhs, a, *lda, b, *ldb, info);
}

extern "C" void slate_dposv(const char* uplo, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb, int* info)
{
    slate_posv(uplo, *n, *nrhs, a, *lda, b, *ldb, info);
}

extern "C" void slate_cposv(const char* uplo, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, int* info)
{
    slate_posv(uplo, *n, *nrhs, a, *lda, b, *ldb, info);
}

extern "C" void slate_zposv(const char* uplo, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, int* info)
{
    slate_posv(uplo, *n, *nrhs, a, *lda, b, *ldb, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_posv(const char* uplostr, const int n, const int nrhs, scalar_t* a, const int lda, scalar_t* b, const int ldb, int* info)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // Check and initialize MPI, else SLATE calls to MPI will fail
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);
    slate::Pivots pivots;

    // create SLATE matrices from the LAPACK data
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(uplo, n, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(n, nrhs, b, ldb, nb, p, q, MPI_COMM_WORLD);

    // computes the solution to the system of linear equations with a square coefficient matrix A and multiple right-hand sides.
    slate::posv(A, B, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });

    // todo:  get a real value for info
    *info = 0;

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "posv(" <<  uplostr << "," << n << "," <<  nrhs << "," << (void*)a << "," <<  lda << "," << (void*)b << "," << ldb << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
