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
void slate_syrk(const char* uplostr, const char* transastr, const int n, const int k, const scalar_t alpha, scalar_t* a, const int lda, const scalar_t beta, scalar_t* c, const int ldc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_ssyrk BLAS_FORTRAN_NAME( slate_ssyrk, SLATE_SSYRK )
#define slate_dsyrk BLAS_FORTRAN_NAME( slate_dsyrk, SLATE_DSYRK )
#define slate_csyrk BLAS_FORTRAN_NAME( slate_csyrk, SLATE_CSYRK )
#define slate_zsyrk BLAS_FORTRAN_NAME( slate_zsyrk, SLATE_ZSYRK )

extern "C" void slate_ssyrk(const char* uplo, const char* transa, const int* n, const int* k, const float* alpha, float* a, const int* lda, const float* beta, float* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

extern "C" void slate_dsyrk(const char* uplo, const char* transa, const int* n, const int* k, const double* alpha, double* a, const int* lda, const double* beta, double* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

extern "C" void slate_csyrk(const char* uplo, const char* transa, const int* n, const int* k, const std::complex<float>* alpha, std::complex<float>* a, const int* lda, const std::complex<float>* beta, std::complex<float>* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

extern "C" void slate_zsyrk(const char* uplo, const char* transa, const int* n, const int* k, const std::complex<double>* alpha, std::complex<double>* a, const int* lda, const std::complex<double>* beta, std::complex<double>* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_syrk(const char* uplostr, const char* transastr, const int n, const int k, const scalar_t alpha, scalar_t* a, const int lda, const scalar_t beta, scalar_t* c, const int ldc)
{
    // start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // need a dummy MPI_Init for SLATE to proceed
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);

    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transastr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);

    // setup so op(A) is n-by-k
    int64_t Am = (transA == blas::Op::NoTrans ? n : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : n);
    int64_t Cn = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::SymmetricMatrix<scalar_t>::fromLAPACK(uplo, Cn, c, ldc, nb, p, q, MPI_COMM_WORLD);

    if (transA == blas::Op::Trans)
        A = transpose(A);
    else if (transA == blas::Op::ConjTrans)
        A = conjTranspose(A);
    assert(A.mt() == C.mt());

    slate::syrk(alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "syrk(" << uplostr[0] << "," << transastr[0] <<  "," << n << "," << k << "," << alpha << "," << (void*)a << "," << lda << "," << beta << "," << (void*)c << "," << ldc << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
