// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack_slate.hh"

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------

// Local function
template <typename scalar_t>
void slate_trsm(const char* sidestr, const char* uplostr, const char* transastr, const char* diagstr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_strsm BLAS_FORTRAN_NAME( slate_strsm, SLATE_STRSM )
#define slate_dtrsm BLAS_FORTRAN_NAME( slate_dtrsm, SLATE_DTRSM )
#define slate_ctrsm BLAS_FORTRAN_NAME( slate_ctrsm, SLATE_CTRSM )
#define slate_ztrsm BLAS_FORTRAN_NAME( slate_ztrsm, SLATE_ZTRSM )

extern "C" void slate_strsm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const float* alpha, float* a, const int* lda, float* b, const int* ldb)
{
    slate_trsm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

extern "C" void slate_dtrsm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const double* alpha, double* a, const int* lda, double* b, const int* ldb)
{
    slate_trsm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

extern "C" void slate_ctrsm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const std::complex<float>* alpha, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb)
{
    slate_trsm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

extern "C" void slate_ztrsm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const std::complex<double>* alpha, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb)
{
    slate_trsm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_trsm(const char* sidestr, const char* uplostr, const char* transastr, const char* diagstr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb)
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

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);
    blas::Side side = blas::char2side(sidestr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transastr[0]);
    blas::Diag diag = blas::char2diag(diagstr[0]);

    // setup so op(B) is m-by-n
    int64_t An  = (side == blas::Side::Left ? m : n);
    int64_t Bm  = m;
    int64_t Bn  = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::TriangularMatrix<scalar_t>::fromLAPACK(uplo, diag, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);

    if (transA == Op::Trans)
        A = transpose(A);
    else if (transA == Op::ConjTrans)
        A = conj_transpose( A );

    slate::trsm(side, alpha, A, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "trsm(" << sidestr[0] << "," << uplostr[0] << "," << transastr[0] <<  "," << diagstr[0] <<  "," << m << "," << n << "," << alpha << "," << (void*)a << "," << lda << "," << (void*)b << "," << ldb << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
