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
void slate_pgemm(
    const char* transA_str, const char* transB_str,
    blas_int m, blas_int n, blas_int k, scalar_t alpha,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB,
    scalar_t beta,
    scalar_t* C_data, blas_int ic, blas_int jc, blas_int const* descC )
{
    Op transA{};
    Op transB{};
    from_string( std::string( 1, transA_str[0] ), &transA );
    from_string( std::string( 1, transB_str[0] ), &transB );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // sizes of A and B
    int64_t Am = (transA == blas::Op::NoTrans ? m : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : m);
    int64_t Bm = (transB == blas::Op::NoTrans ? k : n);
    int64_t Bn = (transB == blas::Op::NoTrans ? n : k);
    int64_t Cm = m;
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
    auto C = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descC ), desc_n( descC ), C_data, desc_lld( descC ),
        desc_mb( descC ), desc_nb( descC ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    C = slate_scalapack_submatrix( Cm, Cn, C, ic, jc, descC );

    if (transA == blas::Op::Trans)
        A = transpose( A );
    else if (transA == blas::Op::ConjTrans)
        A = conj_transpose( A );

    if (transB == blas::Op::Trans)
        B = transpose( B );
    else if (transB == blas::Op::ConjTrans)
        B = conj_transpose( B );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gemm");

    slate::gemm( alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type-generic C++ slate_pgemm routine.

extern "C" {

#define SCALAPACK_psgemm BLAS_FORTRAN_NAME( psgemm, PSGEMM )
void SCALAPACK_psgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    float* alpha,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    float* beta,
    float* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pgemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pdgemm BLAS_FORTRAN_NAME( pdgemm, PDGEMM )
void SCALAPACK_pdgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    double* alpha,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    double* beta,
    double* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pgemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pcgemm BLAS_FORTRAN_NAME( pcgemm, PCGEMM )
void SCALAPACK_pcgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    std::complex<float>* alpha,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    std::complex<float>* beta,
    std::complex<float>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pgemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pzgemm BLAS_FORTRAN_NAME( pzgemm, PZGEMM )
void SCALAPACK_pzgemm(
    const char* transA, const char* transB,
    blas_int const* m, blas_int const* n, blas_int const* k,
    std::complex<double>* alpha,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    std::complex<double>* beta,
    std::complex<double>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_pgemm(
        transA, transB, *m, *n, *k, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
