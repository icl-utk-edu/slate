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
void slate_pgecon(const char* normstr, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t> anorm, blas::real_type<scalar_t>* rcond, scalar_t* work, int lwork, void* irwork, int lirwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSGECON(const char* normstr, int* n, float* a, int* ia, int* ja, int* desca, float* anorm, float* rcond, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void psgecon(const char* normstr, int* n, float* a, int* ia, int* ja, int* desca, float* anorm, float* rcond, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void psgecon_(const char* normstr, int* n, float* a, int* ia, int* ja, int* desca, float* anorm, float* rcond, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDGECON(const char* normstr, int* n, double* a, int* ia, int* ja, int* desca, double* anorm, double* rcond, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void pdgecon(const char* normstr, int* n, double* a, int* ia, int* ja, int* desca, double* anorm, double* rcond, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, iwork, *liwork, info);
}

extern "C" void pdgecon_(const char* normstr, int* n, double* a, int* ia, int* ja, int* desca, double* anorm, double* rcond, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCGECON(const char* normstr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* anorm, float* rcond, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pcgecon(const char* normstr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* anorm, float* rcond, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pcgecon_(const char* normstr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* anorm, float* rcond, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, rwork, *lrwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZGECON(const char* normstr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* anorm, double* rcond, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pzgecon(const char* normstr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* anorm, double* rcond, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, rwork, *lrwork, info);
}

extern "C" void pzgecon_(const char* normstr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* anorm, double* rcond, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* info)
{
    slate_pgecon(normstr, *n, a, *ia, *ja, desca, *anorm, rcond, work, *lwork, rwork, *lrwork, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgecon(const char* normstr, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t> anorm, blas::real_type<scalar_t>* rcond, scalar_t* work, int lwork, void* irwork, int lirwork, int* info)
{
    Norm norm{};
    from_string( std::string( 1, normstr[0] ), &norm );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // todo: extract the real info from getrf
    *info = 0;

    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gecon");

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
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    *rcond = slate::gecondest(norm, A, anorm, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });
}

} // namespace scalapack_api
} // namespace slate
