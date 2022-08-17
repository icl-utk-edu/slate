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
void slate_symm(const char* sidestr, const char* uplostr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb, const scalar_t beta, scalar_t* c, const int ldc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_ssymm BLAS_FORTRAN_NAME( slate_ssymm, SLATE_SSYMM )
#define slate_dsymm BLAS_FORTRAN_NAME( slate_dsymm, SLATE_DSYMM )
#define slate_csymm BLAS_FORTRAN_NAME( slate_csymm, SLATE_CSYMM )
#define slate_zsymm BLAS_FORTRAN_NAME( slate_zsymm, SLATE_ZSYMM )


extern "C" void slate_ssymm(const char* side, const char* uplo, const int* m, const int* n, float* alpha, float* a, const int* lda, float* b, const int* ldb, float* beta, float* c, const int* ldc)
{
    slate_symm(side, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_dsymm(const char* side, const char* uplo, const int* m, const int* n, double* alpha, double* a, const int* lda, double* b, const int* ldb, double* beta, double* c, const int* ldc)
{
    slate_symm(side, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_csymm(const char* side, const char* uplo, const int* m, const int* n, std::complex<float>* alpha, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, std::complex<float>* beta, std::complex<float>* c, const int* ldc)
{
    slate_symm(side, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_zsymm(const char* side, const char* uplo, const int* m, const int* n, std::complex<double>* alpha, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, std::complex<double>* beta, std::complex<double>* c, const int* ldc)
{
    slate_symm(side, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_symm(const char* sidestr, const char* uplostr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb, const scalar_t beta, scalar_t* c, const int ldc)
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

    blas::Side side = blas::char2side(sidestr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);

    // sizes of data
    int64_t An = (side == blas::Side::Left ? m : n);
    int64_t Bm = m;
    int64_t Bn = n;
    int64_t Cm = m;
    int64_t Cn = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::SymmetricMatrix<scalar_t>::fromLAPACK(uplo, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::Matrix<scalar_t>::fromLAPACK(Cm, Cn, c, ldc, nb, p, q, MPI_COMM_WORLD);

    if (side == blas::Side::Left)
        assert(A.mt() == C.mt());
    else
        assert(A.mt() == C.nt());
    assert(B.mt() == C.mt());
    assert(B.nt() == C.nt());

    slate::symm(side, alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "symm(" << sidestr[0] << "," << uplostr[0] <<  "," << m << "," << n << "," << alpha << "," << (void*)a << "," << lda << "," << (void*)b << "," << ldb << "," << beta << "," << (void*)c << "," << ldc << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

}

} // namespace lapack_api
} // namespace slate
