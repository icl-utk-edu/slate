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
void slate_getrf(const int m, const int n, scalar_t* a, const int lda, int* ipiv, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgetrf BLAS_FORTRAN_NAME( slate_sgetrf, SLATE_SGETRF )
#define slate_dgetrf BLAS_FORTRAN_NAME( slate_dgetrf, SLATE_DGETRF )
#define slate_cgetrf BLAS_FORTRAN_NAME( slate_cgetrf, SLATE_CGETRF )
#define slate_zgetrf BLAS_FORTRAN_NAME( slate_zgetrf, SLATE_ZGETRF )

extern "C" void slate_sgetrf(const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info)
{
    slate_getrf(*m, *n, a, *lda, ipiv, info);
}

extern "C" void slate_dgetrf(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info)
{
    slate_getrf(*m, *n, a, *lda, ipiv, info);
}

extern "C" void slate_cgetrf(const int* m, const int* n, std::complex<float>* a, const int* lda, int* ipiv, int* info)
{
    slate_getrf(*m, *n, a, *lda, ipiv, info);
}

extern "C" void slate_zgetrf(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv, int* info)
{
    slate_getrf(*m, *n, a, *lda, ipiv, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_getrf(const int m, const int n, scalar_t* a, const int lda, int* ipiv, int* info)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    // Test the input parameters
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < std::max(1, m))
        *info = -4;
    if (*info != 0)
        return;

    // Quick return if possible
    if (m == 0 || n == 0)
        return;

    int64_t p = 1;
    int64_t q = 1;
    int64_t lookahead = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t panel_threads = slate_lapack_set_panelthreads();

    int64_t Am = m;
    int64_t An = n;
    static int64_t nb = slate_lapack_set_nb(target);
    static int64_t ib = std::min({slate_lapack_set_ib(), nb});
    slate::Pivots pivots;

    // create SLATE matrices from the Lapack layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);

    // factorize using slate
    slate::getrf(A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

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

    // todo: get a real value for info
    *info = 0;

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "getrf(" <<  m << "," <<  n << "," << (void*)a << "," <<  lda << "," << (void*)ipiv << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

}

} // namespace lapack_api
} // namespace slate
