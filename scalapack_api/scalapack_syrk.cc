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
void slate_psyrk(const char* uplostr, const char* transstr, int n, int k, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t beta, scalar_t* c, int ic, int jc, int* descc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_psyrk

extern "C" void PDSYRK(const char* uplo, const char* trans, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pdsyrk(const char* uplo, const char* trans, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pdsyrk_(const char* uplo, const char* trans, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PSSYRK(const char* uplo, const char* trans, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pssyrk(const char* uplo, const char* trans, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pssyrk_(const char* uplo, const char* trans, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PCSYRK(const char* uplo, const char* trans, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pcsyrk(const char* uplo, const char* trans, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pcsyrk_(const char* uplo, const char* trans, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PZSYRK(const char* uplo, const char* trans, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pzsyrk(const char* uplo, const char* trans, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pzsyrk_(const char* uplo, const char* trans, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_psyrk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_psyrk(const char* uplostr, const char* transstr, int n, int k, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t beta, scalar_t* c, int ic, int jc, int* descc)
{
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // setup so op(A) is n-by-k
    int64_t Am = (transA == blas::Op::NoTrans ? n : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : n);
    int64_t Cm = n;
    int64_t Cn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descc), &nprow, &npcol, &myprow, &mypcol);
    auto C = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(uplo, desc_N(descc), c, desc_LLD(descc), desc_NB(descc), grid_order, nprow, npcol, MPI_COMM_WORLD);
    C = slate_scalapack_submatrix(Cm, Cn, C, ic, jc, descc);

    if (transA == blas::Op::Trans)
        A = transpose(A);
    else if (transA == blas::Op::ConjTrans)
        A = conjTranspose(A);
    assert(A.mt() == C.mt());

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "syrk");

    slate::syrk(alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

} // namespace scalapack_api
} // namespace slate
