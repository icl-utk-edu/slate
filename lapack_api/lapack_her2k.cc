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
void slate_her2k(const char* uplostr, const char* transastr, const int n, const int k, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb, const blas::real_type<scalar_t> beta, scalar_t* c, const int ldc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_cher2k BLAS_FORTRAN_NAME( slate_cher2k, SLATE_CHER2K )
#define slate_zher2k BLAS_FORTRAN_NAME( slate_zher2k, SLATE_ZHER2K )

extern "C" void slate_cher2k(const char* uplo, const char* transa, const int* n, const int* k, const std::complex<float>* alpha, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, const float* beta, std::complex<float>* c, const int* ldc)
{
    slate_her2k(uplo, transa, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_zher2k(const char* uplo, const char* transa, const int* n, const int* k, const std::complex<double>* alpha, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, const double* beta, std::complex<double>* c, const int* ldc)
{
    slate_her2k(uplo, transa, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_her2k(const char* uplostr, const char* transastr, const int n, const int k, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb, const blas::real_type<scalar_t> beta, scalar_t* c, const int ldc)
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
    blas::Op trans = blas::char2op(transastr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);

    // setup so op(A) and op(B) are n-by-k
    int64_t Am = (trans == blas::Op::NoTrans ? n : k);
    int64_t An = (trans == blas::Op::NoTrans ? k : n);
    int64_t Bm = Am;
    int64_t Bn = An;
    int64_t Cn = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::HermitianMatrix<scalar_t>::fromLAPACK(uplo, Cn, c, ldc, nb, p, q, MPI_COMM_WORLD);

    if (trans == blas::Op::Trans) {
        A = transpose(A);
        B = transpose(B);
    }
    else if (trans == blas::Op::ConjTrans) {
        A = conjTranspose(A);
        B = conjTranspose(B);
    }
    assert(A.mt() == C.mt());
    assert(B.mt() == C.mt());
    assert(A.nt() == B.nt());

    slate::her2k(alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "her2k(" << uplostr[0] << "," << transastr[0] << "," <<  n << "," <<  k << "," <<  alpha << "," << (void*)a << "," <<  lda << "," << (void*)b << "," << ldb << "," << beta << "," << (void*)c << "," << ldc << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
