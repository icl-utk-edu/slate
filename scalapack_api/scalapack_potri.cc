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
void slate_ppotri(
    const char* uplo_str, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas_int* info )
{
    Uplo uplo{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( An, An, A, ia, ja, descA );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "potri");

    slate::potri( A, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo: extract the real info from potri
    *info = 0;
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pspotri BLAS_FORTRAN_NAME( pspotri, PSPOTRI )
void SCALAPACK_pspotri(
    const char* uplo, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* info )
{
    slate_ppotri(
        uplo, *n,
        A_data, *ia, *ja, descA,
        info );
}

#define SCALAPACK_pdpotri BLAS_FORTRAN_NAME( pdpotri, PDPOTRI )
void SCALAPACK_pdpotri(
    const char* uplo, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* info )
{
    slate_ppotri(
        uplo, *n,
        A_data, *ia, *ja, descA,
        info );
}

#define SCALAPACK_pcpotri BLAS_FORTRAN_NAME( pcpotri, PCPOTRI )
void SCALAPACK_pcpotri(
    const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* info )
{
    slate_ppotri(
        uplo, *n,
        A_data, *ia, *ja, descA,
        info );
}

#define SCALAPACK_pzpotri BLAS_FORTRAN_NAME( pzpotri, PZPOTRI )
void SCALAPACK_pzpotri(
    const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* info )
{
    slate_ppotri(
        uplo, *n,
        A_data, *ia, *ja, descA,
        info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
