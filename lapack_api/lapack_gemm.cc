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
void slate_gemm(const char* transa, const char* transb, int m, int n, int k, scalar_t alpha, scalar_t* a, int lda, scalar_t* b, int ldb, scalar_t beta, scalar_t* c, int ldc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgemm BLAS_FORTRAN_NAME( slate_sgemm, SLATE_SGEMM )
#define slate_dgemm BLAS_FORTRAN_NAME( slate_dgemm, SLATE_DGEMM )
#define slate_cgemm BLAS_FORTRAN_NAME( slate_cgemm, SLATE_CGEMM )
#define slate_zgemm BLAS_FORTRAN_NAME( slate_zgemm, SLATE_ZGEMM )

extern "C" void slate_sgemm(const char* transa, const char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc)
{
    slate_gemm(transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_dgemm(const char* transa, const char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc)
{
    slate_gemm(transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_cgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* lda, std::complex<float>* b, int* ldb, std::complex<float>* beta, std::complex<float>* c, int* ldc)
{
    slate_gemm(transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_zgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* lda, std::complex<double>* b, int* ldb, std::complex<double>* beta, std::complex<double>* c, int* ldc)
{
    slate_gemm(transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_gemm(const char* transastr, const char* transbstr, int m, int n, int k, scalar_t alpha, scalar_t* a, int lda, scalar_t* b, int ldb, scalar_t beta, scalar_t* c, int ldc)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // Need a dummy MPI_Init for SLATE to proceed
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);

    int64_t p = 1;
    int64_t q = 1;
    int64_t lookahead = 1;
    static slate::Target target = slate_lapack_set_target();

    // sizes
    blas::Op transA = blas::char2op(transastr[0]);
    blas::Op transB = blas::char2op(transbstr[0]);
    int64_t Am = (transA == blas::Op::NoTrans ? m : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : m);
    int64_t Bm = (transB == blas::Op::NoTrans ? k : n);
    int64_t Bn = (transB == blas::Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;
    static int64_t nb = slate_lapack_set_nb(target);

    // create SLATE matrices from the Lapack layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::Matrix<scalar_t>::fromLAPACK(Cm, Cn, c, ldc, nb, p, q, MPI_COMM_WORLD);

    if (transA == blas::Op::Trans)
        A = transpose(A);
    else if (transA == blas::Op::ConjTrans)
        A = conjTranspose(A);

    if (transB == blas::Op::Trans)
        B = transpose(B);
    else if (transB == blas::Op::ConjTrans)
        B = conjTranspose(B);

    slate::gemm(alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "gemm(" << transastr[0] << "," << transbstr[0] << "," <<  m << "," <<  n << "," <<  k << "," <<  alpha << "," << (void*)a << "," <<  lda << "," << (void*)b << "," << ldb << "," << beta << "," << (void*)c << "," << ldc << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

}

} // namespace lapack_api
} // namespace slate
