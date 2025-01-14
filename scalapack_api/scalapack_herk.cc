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
template <typename scalar_t>
void slate_pherk(
    const char* uplo_str, const char* trans_str,
    blas_int n, blas_int k, blas::real_type<scalar_t> alpha,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t> beta,
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
    auto C = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descC ), C_data, desc_lld( descC ), desc_nb( descC ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    C = slate_scalapack_submatrix( Cm, Cn, C, ic, jc, descC );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "herk");

    if (transA == blas::Op::Trans)
        A = transpose( A );
    else if (transA == blas::Op::ConjTrans)
        A = conj_transpose( A );
    assert( A.mt() == C.mt() );

    slate::herk( alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pcherk BLAS_FORTRAN_NAME( pcherk, PCHERK )
void SCALAPACK_pcherk(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, float* alpha,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* beta,
    std::complex<float>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pherk(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pzherk BLAS_FORTRAN_NAME( pzherk, PZHERK )
void SCALAPACK_pzherk(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, double* alpha,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* beta,
    std::complex<double>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pherk(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA, *beta,
        C_data, *ic, *jc, descC );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
