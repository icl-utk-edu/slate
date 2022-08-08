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
void slate_hemm(const char* sidestr, const char* uplostr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb, const scalar_t beta, scalar_t* c, const int ldc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_chemm BLAS_FORTRAN_NAME( slate_chemm, SLATE_CHEMM )
#define slate_zhemm BLAS_FORTRAN_NAME( slate_zhemm, SLATE_ZHEMM )

extern "C" void slate_chemm(const char* side, const char* uplo, const int* m, const int* n, std::complex<float>* alpha, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb, std::complex<float>* beta, std::complex<float>* c, const int* ldc)
{
    slate_hemm(side, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

extern "C" void slate_zhemm(const char* side, const char* uplo, const int* m, const int* n, std::complex<double>* alpha, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb, std::complex<double>* beta, std::complex<double>* c, const int* ldc)
{
    slate_hemm(side, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_hemm(const char* sidestr, const char* uplostr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb, const scalar_t beta, scalar_t* c, const int ldc)
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

    // create SLATE matrices from the Lapack layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(uplo, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::Matrix<scalar_t>::fromLAPACK(Cm, Cn, c, ldc, nb, p, q, MPI_COMM_WORLD);

    if (side == blas::Side::Left)
        assert(A.mt() == C.mt());
    else
        assert(A.mt() == C.nt());
    assert(B.mt() == C.mt());
    assert(B.nt() == C.nt());

    slate::hemm(side, alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "hemm(" << sidestr[0] << "," << uplostr[0] << "," <<  m << "," <<  n << "," <<  alpha << "," << (void*)a << "," <<  lda << "," << (void*)b << "," << ldb << "," << beta << "," << (void*)c << "," << ldc << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

}

} // namespace lapack_api
} // namespace slate
