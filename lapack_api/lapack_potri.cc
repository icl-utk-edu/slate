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
void slate_potri(const char* uplostr, const int n, scalar_t* a, const int lda, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_spotri BLAS_FORTRAN_NAME( slate_spotri, SLATE_SPOTRI )
#define slate_dpotri BLAS_FORTRAN_NAME( slate_dpotri, SLATE_DPOTRI )
#define slate_cpotri BLAS_FORTRAN_NAME( slate_cpotri, SLATE_CPOTRI )
#define slate_zpotri BLAS_FORTRAN_NAME( slate_zpotri, SLATE_ZPOTRI )

extern "C" void slate_spotri(const char* uplo, const int* n, float* a, const int* lda, int* info)
{
    return slate_potri(uplo, *n, a, *lda, info);
}

extern "C" void slate_dpotri(const char* uplo, const int* n, double* a, const int* lda, int* info)
{
    return slate_potri(uplo, *n, a, *lda, info);
}

extern "C" void slate_cpotri(const char* uplo, const int* n, std::complex<float>* a, const int* lda, int* info)
{
    return slate_potri(uplo, *n, a, *lda, info);
}

extern "C" void slate_zpotri(const char* uplo, const int* n, std::complex<double>* a, const int* lda, int* info)
{
    return slate_potri(uplo, *n, a, *lda, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_potri(const char* uplostr, const int n, scalar_t* a, const int lda, int* info)
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
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);

    // sizes of data
    int64_t An = n;

    // create SLATE matrices from the Lapack layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(uplo, An, a, lda, nb, p, q, MPI_COMM_WORLD);

    slate::potri(A, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });

    // todo get a real value for info
    *info = 0;

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "potri(" << uplostr[0] << "," << n << "," << (void*)a << "," <<  lda << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
