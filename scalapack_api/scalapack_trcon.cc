// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_ptrcon(const char* normstr, const char* uplostr, const char* diagstr, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* rcond, scalar_t* work, int lwork, void* irwork, int lirwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSTRCON(const char* normstr, const char* uplostr, const char* diagstr, int* n, float* a, int* ia, int* ja, int* desca, float* rcond, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void pstrcon(const char* normstr, const char* uplostr, const char* diagstr, int* n, float* a, int* ia, int* ja, int* desca, float* rcond, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void pstrcon_(const char* normstr, const char* uplostr, const char* diagstr, int* n, float* a, int* ia, int* ja, int* desca, float* rcond, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDTRCON(const char* normstr, const char* uplostr, const char* diagstr, int* n, double* a, int* ia, int* ja, int* desca, double* rcond, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void pdtrcon(const char* normstr, const char* uplostr, const char* diagstr, int* n, double* a, int* ia, int* ja, int* desca, double* rcond, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void pdtrcon_(const char* normstr, const char* uplostr, const char* diagstr, int* n, double* a, int* ia, int* ja, int* desca, double* rcond, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCTRCON(const char* normstr, const char* uplostr, const char* diagstr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* rcond, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pctrcon(const char* normstr, const char* uplostr, const char* diagstr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* rcond, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pctrcon_(const char* normstr, const char* uplostr, const char* diagstr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* rcond, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, rwork, *lrwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZTRCON(const char* normstr, const char* uplostr, const char* diagstr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* rcond, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pztrcon(const char* normstr, const char* uplostr, const char* diagstr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* rcond, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pztrcon_(const char* normstr, const char* uplostr, const char* diagstr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* rcond, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* info)
{
    slate_ptrcon(normstr, uplostr, diagstr, *n, a, *ia, *ja, desca, rcond, work, *lwork, rwork, *lrwork, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_ptrcon(const char* normstr, const char* uplostr, const char* diagstr, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* rcond, scalar_t* work, int lwork, void* irwork, int lirwork, int* info)
{
    lapack::Norm norm = lapack::char2norm(normstr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Diag diag = blas::char2diag(diagstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    static int64_t panel_threads = slate_scalapack_set_panelthreads();
    static int64_t ib = slate_scalapack_set_ib();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // todo: extract the real info from getrf
    *info = 0;

    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "trcon");

    if (lwork == -1 || lirwork == -1) {
        *work = 0;
        if constexpr (std::is_same_v<scalar_t, blas::real_type<scalar_t>>) {
            *(int*)irwork = 0;
        }
        else {
            *(blas::real_type<scalar_t>*)irwork = 0;
        }
        return;
    }

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    auto AT = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(uplo, diag, desc_N(desca), a, desc_LLD(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    AT = slate_scalapack_submatrix(Am, An, AT, ia, ja, desca);

    blas::real_type<scalar_t> anorm = slate::norm( norm, AT, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    *rcond = slate::trcondest( norm, AT, anorm, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });
}

} // namespace scalapack_api
} // namespace slate
