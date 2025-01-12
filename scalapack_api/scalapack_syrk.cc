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
void slate_psyrk(
    const char* uplo_str, const char* trans_str,
    blas_int n, blas_int k, scalar_t alpha,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    scalar_t beta,
    scalar_t* C_data, blas_int ic, blas_int jc, blas_int const* descC )
{
    Uplo uplo{};
    Op transA{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, trans_str[0] ), &transA );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // setup so op(A) is n-by-k
    int64_t Am = (transA == blas::Op::NoTrans ? n : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : n);
    int64_t Cm = n;
    int64_t Cn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    Cblacs_gridinfo( desc_ctxt( descC ), &nprow, &npcol, &myprow, &mypcol );
    auto C = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descC ), C_data, desc_lld( descC ), desc_nb( descC ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    C = slate_scalapack_submatrix( Cm, Cn, C, ic, jc, descC );

    if (transA == blas::Op::Trans)
        A = transpose( A );
    else if (transA == blas::Op::ConjTrans)
        A = conj_transpose( A );
    assert( A.mt() == C.mt() );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "syrk");

    slate::syrk( alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pssyrk BLAS_FORTRAN_NAME( pssyrk, PSSYRK )
void SCALAPACK_pssyrk(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, float* alpha,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* beta,
    float* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psyrk(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pdsyrk BLAS_FORTRAN_NAME( pdsyrk, PDSYRK )
void SCALAPACK_pdsyrk(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, double* alpha,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* beta,
    double* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psyrk(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pcsyrk BLAS_FORTRAN_NAME( pcsyrk, PCSYRK )
void SCALAPACK_pcsyrk(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, std::complex<float>* alpha,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* beta,
    std::complex<float>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psyrk(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pzsyrk BLAS_FORTRAN_NAME( pzsyrk, PZSYRK )
void SCALAPACK_pzsyrk(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, std::complex<double>* alpha,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* beta,
    std::complex<double>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psyrk(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA, *beta,
        C_data, *ic, *jc, descC );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
