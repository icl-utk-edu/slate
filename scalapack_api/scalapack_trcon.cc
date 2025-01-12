// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

//------------------------------------------------------------------------------
/// SLATE ScaLAPACK wrapper sets up SLATE matrices from ScaLAPACK descriptors
/// and calls SLATE.
/// If scalar_t is real,    irwork is integer.
/// If scalar_t is complex, irwork is real.
template <typename scalar_t>
void slate_ptrcon(
    const char* norm_str, const char* uplo_str, const char* diag_str, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* rcond,
    scalar_t* work, blas_int lwork,
    void* irwork, blas_int lirwork,
    blas_int* info )
{
    Norm norm{};
    Uplo uplo{};
    Diag diag{};
    from_string( std::string( 1, norm_str[0] ), &norm );
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, diag_str[0] ), &diag );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // todo: extract the real info from getrf
    *info = 0;

    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "trcon");

    if (lwork == -1 || lirwork == -1) {
        *work = 0;
        if constexpr (std::is_same_v<scalar_t, blas::real_type<scalar_t>>) {
            *(blas_int*)irwork = 0;
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
    auto AT = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
        uplo, diag, desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    AT = slate_scalapack_submatrix( Am, An, AT, ia, ja, descA );

    blas::real_type<scalar_t> Anorm = slate::norm( norm, AT, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    *rcond = slate::trcondest( norm, AT, Anorm, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pstrcon BLAS_FORTRAN_NAME( pstrcon, PSTRCON )
void SCALAPACK_pstrcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* rcond,
    float* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_ptrcon(
        norm, uplo, diag, *n,
        A_data, *ia, *ja, descA,
        rcond, work, *lwork, iwork, *liwork, info );
}

#define SCALAPACK_pdtrcon BLAS_FORTRAN_NAME( pdtrcon, PDTRCON )
void SCALAPACK_pdtrcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* rcond,
    double* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_ptrcon(
        norm, uplo, diag, *n,
        A_data, *ia, *ja, descA,
        rcond, work, *lwork, iwork, *liwork, info );
}

#define SCALAPACK_pctrcon BLAS_FORTRAN_NAME( pctrcon, PCTRCON )
void SCALAPACK_pctrcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* rcond,
    std::complex<float>* work, blas_int const* lwork,
    float* rwork, blas_int const* lrwork,
    blas_int* info )
{
    slate_ptrcon(
        norm, uplo, diag, *n,
        A_data, *ia, *ja, descA,
        rcond, work, *lwork, rwork, *lrwork, info );
}

#define SCALAPACK_pztrcon BLAS_FORTRAN_NAME( pztrcon, PZTRCON )
void SCALAPACK_pztrcon(
    const char* norm, const char* uplo, const char* diag, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* rcond,
    std::complex<double>* work, blas_int const* lwork,
    double* rwork, blas_int const* lrwork,
    blas_int* info )
{
    slate_ptrcon(
        norm, uplo, diag, *n,
        A_data, *ia, *ja, descA,
        rcond, work, *lwork, rwork, *lrwork, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
