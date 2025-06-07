// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "scalapack_slate.hh"

#include <complex>

namespace slate {
namespace scalapack_api {

//------------------------------------------------------------------------------
/// SLATE ScaLAPACK wrapper sets up SLATE matrices from ScaLAPACK descriptors
/// and calls SLATE.
template <typename scalar_t>
void slate_ppotrs(
    const char* uplo_str, blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB,
    blas_int* info )
{
    Uplo uplo{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto Afull = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    auto Asub = slate_scalapack_submatrix( n, n, Afull, ia, ja, descA );
    slate::HermitianMatrix<scalar_t> A( uplo, Asub );

    Cblacs_gridinfo( desc_ctxt( descB ), &nprow, &npcol, &myprow, &mypcol );
    auto Bfull = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descB ), desc_n( descB ), B_data, desc_lld( descB ),
        desc_mb( descB ), desc_nb( descB ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    slate::Matrix<scalar_t> B = slate_scalapack_submatrix( n, nrhs, Bfull, ia, ja, descB );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "potrs");

    slate::potrs( A, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
    });

    // todo: extract the real info
    *info = 0;
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pspotrs BLAS_FORTRAN_NAME( pspotrs, PSPOTRS )
void SCALAPACK_pspotrs(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_ppotrs(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pdpotrs BLAS_FORTRAN_NAME( pdpotrs, PDPOTRS )
void SCALAPACK_pdpotrs(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_ppotrs(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pcpotrs BLAS_FORTRAN_NAME( pcpotrs, PCPOTRS )
void SCALAPACK_pcpotrs(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_ppotrs(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pzpotrs BLAS_FORTRAN_NAME( pzpotrs, PZPOTRS )
void SCALAPACK_pzpotrs(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_ppotrs(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
