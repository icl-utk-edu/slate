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
void slate_ptrmm(
    const char* side_str, const char* uplo_str, const char* transA_str,
    const char* diag_str,
    blas_int m, blas_int n, scalar_t alpha,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB)
{
    Side side{};
    Uplo uplo{};
    Op transA{};
    Diag diag{};
    from_string( std::string( 1, side_str[0] ), &side );
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, transA_str[0] ), &transA );
    from_string( std::string( 1, diag_str[0] ), &diag );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // setup so op(B) is m-by-n
    int64_t An = (side == blas::Side::Left ? m : n);
    int64_t Am = An;
    int64_t Bm = m;
    int64_t Bn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto AT = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
        uplo, diag, desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    AT = slate_scalapack_submatrix( Am, An, AT, ia, ja, descA );

    Cblacs_gridinfo( desc_ctxt( descB ), &nprow, &npcol, &myprow, &mypcol );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descB ), desc_n( descB ), B_data, desc_lld( descB ),
        desc_mb( descB ), desc_nb( descB ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    B = slate_scalapack_submatrix( Bm, Bn, B, ib, jb, descB );

    if (transA == Op::Trans)
        AT = transpose( AT );
    else if (transA == Op::ConjTrans)
        AT = conj_transpose( AT );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "trmm");

    slate::trmm( side, alpha, AT, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pdtrmm BLAS_FORTRAN_NAME( pdtrmm, PDTRMM )
void SCALAPACK_pdtrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n, double* alpha,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB )
{
    slate_ptrmm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB );
}

#define SCALAPACK_pstrmm BLAS_FORTRAN_NAME( pstrmm, PSTRMM )
void SCALAPACK_pstrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n, float* alpha,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB )
{
    slate_ptrmm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB );
}

#define SCALAPACK_pctrmm BLAS_FORTRAN_NAME( pctrmm, PCTRMM )
void SCALAPACK_pctrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n, std::complex<float>* alpha,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB )
{
    slate_ptrmm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB );
}

#define SCALAPACK_pztrmm BLAS_FORTRAN_NAME( pztrmm, PZTRMM )
void SCALAPACK_pztrmm(
    const char* side, const char* uplo, const char* transA, const char* diag,
    blas_int const* m, blas_int const* n, std::complex<double>* alpha,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB )
{
    slate_ptrmm(
        side, uplo, transA, diag, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
