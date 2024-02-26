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
blas::real_type<scalar_t> slate_lanhe(const char* normstr, const char* uplostr, int n, scalar_t* a, int lda, blas::real_type<scalar_t>* work);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_clanhe BLAS_FORTRAN_NAME( slate_clanhe, SLATE_CLANHE )
#define slate_zlanhe BLAS_FORTRAN_NAME( slate_zlanhe, SLATE_ZLANHE )

extern "C" float slate_clanhe(const char* norm, const char* uplo, int* n, std::complex<float>* a, int* lda, float* work)
{
    return slate_lanhe(norm, uplo, *n, a, *lda, work);
}

extern "C" double slate_zlanhe(const char* norm, const char* uplo, int* n, std::complex<double>* a, int* lda, double* work)
{
    return slate_lanhe(norm, uplo, *n, a, *lda, work);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
blas::real_type<scalar_t> slate_lanhe(const char* normstr, const char* uplostr, int n, scalar_t* a, int lda, blas::real_type<scalar_t>* work)
{
    // start timing
    // static int verbose = slate_lapack_set_verbose();
    // double timestart = 0.0;
    // if (verbose) timestart = omp_get_wtime();

    // need a dummy MPI_Init for SLATE to proceed
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);

    lapack::Norm norm = lapack::char2norm(normstr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);

    // sizes of matrices
    int64_t An = n;

    // create SLATE matrix from the Lapack layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromLAPACK(uplo, An, a, lda, nb, p, q, MPI_COMM_WORLD);

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm(norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    // if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "lanhe(" << normstr[0] << "," << uplostr[0] <<  "," << n << "," <<  (void*)a << "," <<  lda << "," <<  (void*)work << ") " <<  (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

    return A_norm;
}

} // namespace lapack_api
} // namespace slate
