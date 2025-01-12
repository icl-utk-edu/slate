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

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_psymm(const char* side_str, const char* uplo_str, blas_int m, blas_int n, scalar_t alpha,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB, scalar_t beta,
    scalar_t* C_data, blas_int ic, blas_int jc, blas_int const* descC )
{
    Side side{};
    Uplo uplo{};
    from_string( std::string( 1, side_str[0] ), &side );
    from_string( std::string( 1, uplo_str[0] ), &uplo );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    int64_t An = (side == blas::Side::Left ? m : n);
    int64_t Am = An;
    int64_t Bm = m;
    int64_t Bn = n;
    int64_t Cm = m;
    int64_t Cn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto AS = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    AS = slate_scalapack_submatrix( Am, An, AS, ia, ja, descA );

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

    if (side == blas::Side::Left)
        assert( AS.mt() == C.mt() );
    else
        assert( AS.mt() == C.nt() );
    assert( B.mt() == C.mt() );
    assert( B.nt() == C.nt() );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "symm");

    slate::symm( side, alpha, AS, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

//------------------------------------------------------------------------------
// Each Fortran interface for all Fortran interfaces
// Each Fortran interface calls the type-generic C++ slate_pgemm routine.

extern "C" {

#define SCALAPACK_pssymm BLAS_FORTRAN_NAME( pssymm, PSSYMM )
void SCALAPACK_pssymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n, float* alpha,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    float* beta,
    float* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psymm(
        side, uplo, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pdsymm BLAS_FORTRAN_NAME( pdsymm, PDSYMM )
void SCALAPACK_pdsymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n, double* alpha,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    double* beta,
    double* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psymm(
        side, uplo, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pcsymm BLAS_FORTRAN_NAME( pcsymm, PCSYMM )
void SCALAPACK_pcsymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n, std::complex<float>* alpha,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    std::complex<float>* beta,
    std::complex<float>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psymm(
        side, uplo, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

#define SCALAPACK_pzsymm BLAS_FORTRAN_NAME( pzsymm, PZSYMM )
void SCALAPACK_pzsymm(
    const char* side, const char* uplo,
    blas_int const* m, blas_int const* n, std::complex<double>* alpha,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    std::complex<double>* beta,
    std::complex<double>* C_data, blas_int const* ic, blas_int const* jc, blas_int const* descC )
{
    slate_psymm(
        side, uplo, *m, *n, *alpha,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB, *beta,
        C_data, *ic, *jc, descC );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
