// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "scalapack_slate.hh"

#include <complex>

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);

// Type generic function calls the SLATE routine
template< typename scalar_t >
blas::real_type<scalar_t> slate_plantr(const char* normstr, const char* uplostr, const char* diagstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" float PSLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pslantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pslantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PDLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" float PCLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pclantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pclantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PZLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
blas::real_type<scalar_t> slate_plantr(const char* normstr, const char* uplostr, const char* diagstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work)
{
    // todo: figure out if the pxq grid is in row or column

    // make blas single threaded
    // todo: does this set the omp num threads correctly
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    lapack::Norm norm = lapack::char2norm(normstr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Diag diag = blas::char2diag(diagstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    int64_t lookahead = 1;

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myrow, &mycol);
    auto A = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(uplo, diag, desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_NB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    if (verbose && myrow == 0 && mycol == 0)
        logprintf("%s target %d\n", "lantr", (int)target);

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm(norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    slate_set_num_blas_threads(saved_num_blas_threads);

    return A_norm;
}

} // namespace scalapack_api
} // namespace slate
