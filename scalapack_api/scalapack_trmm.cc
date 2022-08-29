// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "scalapack_slate.hh"

#include <complex>

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);

// Declarations
template< typename scalar_t >
void slate_ptrmm(const char* side, const char* uplo, const char* transa, const char* diag, int m, int n, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_ptrmm

extern "C" void PDTRMM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pdtrmm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pdtrmm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

// -----------------------------------------------------------------------------

extern "C" void PSTRMM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pstrmm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pstrmm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

// -----------------------------------------------------------------------------

extern "C" void PCTRMM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pctrmm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pctrmm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

// -----------------------------------------------------------------------------

extern "C" void PZTRMM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pztrmm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pztrmm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb)
{
    slate_ptrmm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_ptrmm(const char* sidestr, const char* uplostr, const char* transastr, const char* diagstr, int m, int n, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb)
{
    blas::Side side = blas::char2side(sidestr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transastr[0]);
    blas::Diag diag = blas::char2diag(diagstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // setup so op(B) is m-by-n
    int64_t An = (side == blas::Side::Left ? m : n);
    int64_t Am = An;
    int64_t Bm = m;
    int64_t Bn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto AT = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(uplo, diag, desc_N(desca), a, desc_LLD(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    AT = slate_scalapack_submatrix(Am, An, AT, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descb), &nprow, &npcol, &myprow, &mypcol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descb), desc_N(descb), b, desc_LLD(descb), desc_MB(descb), desc_NB(descb), grid_order, nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    if (transA == Op::Trans)
        AT = transpose(AT);
    else if (transA == Op::ConjTrans)
        AT = conjTranspose(AT);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "trmm");

    slate::trmm(side, alpha, AT, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

} // namespace scalapack_api
} // namespace slate
