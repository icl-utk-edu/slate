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
void slate_pgels(const char* transstr, int m, int n, int nrhs, scalar_t* a, int lda, scalar_t* b, int ldb, scalar_t* work, int lwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgels BLAS_FORTRAN_NAME( slate_sgels, SLATE_SGELS )
#define slate_dgels BLAS_FORTRAN_NAME( slate_dgels, SLATE_DGELS )
#define slate_cgels BLAS_FORTRAN_NAME( slate_cgels, SLATE_CGELS )
#define slate_zgels BLAS_FORTRAN_NAME( slate_zgels, SLATE_ZGELS )

extern "C" void slate_sgels(const char* trans, int* m, int* n, int* nrhs, float* a, int* lda, float* b, int* ldb, float* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

extern "C" void slate_dgels(const char* trans, int* m, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, double* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

extern "C" void slate_cgels(const char* trans, int* m, int* n, int* nrhs, std::complex<float>* a, int* lda, std::complex<float>* b, int* ldb, std::complex<float>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

extern "C" void slate_zgels(const char* trans, int* m, int* n, int* nrhs, std::complex<double>* a, int* lda, std::complex<double>* b, int* ldb, std::complex<double>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_pgels(const char* transstr, int m, int n, int nrhs, scalar_t* a, int lda, scalar_t* b, int ldb, scalar_t* work, int lwork, int* info)
{
    using real_t = blas::real_type<scalar_t>;

    // Respond to workspace query with a minimal value (1); workspace
    // is allocated within the SLATE routine.
    if (lwork == -1) {
        work[0] = (real_t)1.0;
        *info = 0;
        return;
    }

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
    static int64_t nb = slate_lapack_set_nb(target);
    static int64_t panel_threads = slate_lapack_set_panelthreads();
    static int64_t inner_blocking = slate_lapack_set_ib();

    // sizes
    blas::Op trans = blas::char2op(transstr[0]);
    // A is m-by-n, BX is max(m, n)-by-nrhs.
    // If op == NoTrans, op(A) is m-by-n, B is m-by-nrhs
    // otherwise,        op(A) is n-by-m, B is n-by-nrhs.
    int64_t Am = (trans == slate::Op::NoTrans ? m : n);
    int64_t An = (trans == slate::Op::NoTrans ? n : m);
    int64_t Bm = (trans == slate::Op::NoTrans ? m : n);
    int64_t Bn = nrhs;

    // create SLATE matrices from the LAPACK layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);

    // Apply transpose
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        opA = conjTranspose(A);

    slate::gels(opA, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, inner_blocking}
    });

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "gels(" << transstr[0] << "," <<  m << "," <<  n << "," << nrhs << "," <<  (void*)a << "," <<  lda << "," << (void*)b << "," << ldb << "," << (void*)work << "," << lwork << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

    // todo: extract the real info
    *info = 0;
}

} // namespace lapack_api
} // namespace slate
