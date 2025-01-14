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
void slate_pposv(
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

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;
    int64_t Bm = n;
    int64_t Bn = nrhs;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    Cblacs_gridinfo( desc_ctxt( descB ), &nprow, &npcol, &myprow, &mypcol );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descB ), desc_n( descB ), B_data, desc_lld( descB ),
        desc_mb( descB ), desc_nb( descB ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    B = slate_scalapack_submatrix( Bm, Bn, B, ib, jb, descB );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "posv");

    slate::posv( A, B, {
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

#define SCALAPACK_psposv BLAS_FORTRAN_NAME( psposv, PSPOSV )
void SCALAPACK_psposv(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pposv(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pdposv BLAS_FORTRAN_NAME( pdposv, PDPOSV )
void SCALAPACK_pdposv(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pposv(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pcposv BLAS_FORTRAN_NAME( pcposv, PCPOSV )
void SCALAPACK_pcposv(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pposv(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pzposv BLAS_FORTRAN_NAME( pzposv, PZPOSV )
void SCALAPACK_pzposv(
    const char* uplo, blas_int const* n, blas_int* nrhs,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pposv(
        uplo, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
