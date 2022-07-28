// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack_slate.hh"

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------
using lld = long long int;

// Local function
template< typename scalar_t, typename half_scalar_t >
void slate_gesv(const int n, const int nrhs, scalar_t* a, const int lda, int* ipiv, scalar_t* b, const int ldb, scalar_t* x, const int ldx, scalar_t* work, half_scalar_t* swork, int* iter, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_dsgesv BLAS_FORTRAN_NAME( slate_dsgesv, SLATE_DSGESV )
#define slate_zcgesv BLAS_FORTRAN_NAME( slate_zcgesv, SLATE_ZCGESV )

extern "C" void slate_dsgesv(const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, double* x, const int* ldx, double* work, float* swork, int* iter, int* info)
{
    slate_gesv(*n, *nrhs, a, *lda, ipiv, b, *ldb, x, *ldx, work, swork, iter, info);
}

extern "C" void slate_zcgesv(const int* n, const int* nrhs, std::complex<double>* a, const int* lda, int* ipiv, std::complex<double>* b, const int* ldb, std::complex<double>* x, const int* ldx, std::complex<double>* work, std::complex<float>* swork, int* iter, int* info)
{
    slate_gesv(*n, *nrhs, a, *lda, ipiv, b, *ldb, x, *ldx, work, swork, iter, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t, typename half_scalar_t >
void slate_gesv(const int n, const int nrhs, scalar_t* a, const int lda, int* ipiv, scalar_t* b, const int ldb, scalar_t* x, const int ldx, scalar_t* work, half_scalar_t* swork, int* iter, int* info)
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

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t panel_threads = slate_lapack_set_panelthreads();
    static int64_t nb = slate_lapack_set_nb(target);
    static int64_t ib = std::min({slate_lapack_set_ib(), nb});
    slate::Pivots pivots;

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(n, n, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(n, nrhs, b, ldb, nb, p, q, MPI_COMM_WORLD);
    auto X = slate::Matrix<scalar_t>::fromLAPACK(n, nrhs, b, ldb, nb, p, q, MPI_COMM_WORLD);

    // computes the solution to the system of linear equations with a square coefficient matrix A and multiple right-hand sides.
    int iters;
    slate::gesvMixed(A, pivots, B, X, iters, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target},
            {slate::Option::MaxPanelThreads, panel_threads},
            {slate::Option::InnerBlocking, ib}
        });
    *iter = iters;

    // extract pivots from SLATE's Pivots structure into LAPACK ipiv array
    {
        int64_t p_count = 0;
        int64_t t_iter_add = 0;
        for (auto t_iter = pivots.begin(); t_iter != pivots.end(); ++t_iter) {
            for (auto p_iter = t_iter->begin(); p_iter != t_iter->end(); ++p_iter) {
                ipiv[p_count] = p_iter->tileIndex() * nb + p_iter->elementOffset() + 1 + t_iter_add;
                p_count++;
            }
            t_iter_add += nb;
        }
    }

    // todo:  get a real value for info
    *info = 0;

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << slate_lapack_scalar_t_to_char(swork) << "gesv(" <<  n << "," <<  nrhs << "," << (void*)a << "," <<  lda << "," << (void*)ipiv << "," << (void*)b << "," << ldb << (void*)x << "," << ldx << "," << (void*)work << "," << (void*)swork << "," << iter << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
