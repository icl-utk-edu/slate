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
void slate_pher2k(
    const char* uplo_str, const char* trans_str, blas_int n, blas_int k, scalar_t alpha,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB,
    blas::real_type<scalar_t> beta,
    scalar_t* C_data, blas_int ic, blas_int jc, blas_int const* descC )
{
    Uplo uplo{};
    Op trans{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, trans_str[0] ), &trans );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // setup so op(A) and op(B) are n-by-k
    int64_t Am = (trans == blas::Op::NoTrans ? n : k);
    int64_t An = (trans == blas::Op::NoTrans ? k : n);
    int64_t Bm = Am;
    int64_t Bn = An;
    int64_t Cn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    Cblacs_gridinfo( desc_ctxt( descB ), &nprow, &npcol, &myprow, &mypcol );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descB ), desc_n( descB ), B_data, desc_lld( descB ),
        desc_mb( descB ), desc_nb( descB ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    B = slate_scalapack_submatrix( Bm, Bn, B, ib, jb, descB );

    Cblacs_gridinfo( desc_ctxt( descC ), &nprow, &npcol, &myprow, &mypcol );
    auto C = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descC ), C_data, desc_lld( descC ), desc_nb( descC ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    C = slate_scalapack_submatrix( Cn, Cn, C, ic, jc, descC );

    if (trans == blas::Op::Trans) {
        A = transpose( A );
        B = transpose( B );
    }
    else if (trans == blas::Op::ConjTrans) {
        A = conj_transpose( A );
        B = conj_transpose( B );
    }
    assert( A.mt() == C.mt() );
    assert( B.mt() == C.mt() );
    assert( A.nt() == B.nt() );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "her2k");

    slate::her2k( alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pcher2k BLAS_FORTRAN_NAME( pcher2k, PCHER2K )
void SCALAPACK_pcher2k(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, std::complex<float>* alpha,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    float* beta,
    std::complex<float>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pher2k(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pzher2k BLAS_FORTRAN_NAME( pzher2k, PZHER2K )
void SCALAPACK_pzher2k(
    const char* uplo, const char* trans,
    blas_int const* n, blas_int const* k, std::complex<double>* alpha,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    double* beta,
    std::complex<double>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pher2k(
        uplo, trans, *n, *k, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
