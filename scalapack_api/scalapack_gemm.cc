// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);

// Declarations
template< typename scalar_t >
void slate_pgemm(const char* transastr, const char* transbstr, int m, int n, int k, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb, scalar_t beta, scalar_t* c, int ic, int jc, int* descc);

// -----------------------------------------------------------------------------
// Each C interface for all Fortran interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type-generic C++ slate_pgemm routine.

extern "C" void PDGEMM(const char* transa, const char* transb, int* m, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pdgemm(const char* transa, const char* transb, int* m, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pdgemm_(const char* transa, const char* transb, int* m, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PSGEMM(const char* transa, const char* transb, int* m, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void psgemm(const char* transa, const char* transb, int* m, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void psgemm_(const char* transa, const char* transb, int* m, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PCGEMM(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pcgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pcgemm_(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PZGEMM(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pzgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pzgemm_(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------
// Exposed type-specific API

extern "C" void slate_pdgemm(const char* transa, const char* transb, int* m, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void slate_psgemm(const char* transa, const char* transb, int* m, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void slate_pcgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void slate_pzgemm(const char* transa, const char* transb, int* m, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_pgemm(transa, transb, *m, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_pgemm(const char* transastr, const char* transbstr, int m, int n, int k, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb, scalar_t beta, scalar_t* c, int ic, int jc, int* descc)
{
    blas::Op transA = blas::char2op(transastr[0]);
    blas::Op transB = blas::char2op(transbstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // sizes of A and B
    int64_t Am = (transA == blas::Op::NoTrans ? m : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : m);
    int64_t Bm = (transB == blas::Op::NoTrans ? k : n);
    int64_t Bn = (transB == blas::Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descb), &nprow, &npcol, &myprow, &mypcol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descb), desc_N(descb), b, desc_LLD(descb), desc_MB(descb), desc_NB(descb), grid_order, nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    Cblacs_gridinfo(desc_CTXT(descc), &nprow, &npcol, &myprow, &mypcol);
    auto C = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descc), desc_N(descc), c, desc_LLD(descc), desc_MB(descc), desc_NB(descc), grid_order, nprow, npcol, MPI_COMM_WORLD);
    C = slate_scalapack_submatrix(Cm, Cn, C, ic, jc, descc);

    if (transA == blas::Op::Trans)
        A = transpose(A);
    else if (transA == blas::Op::ConjTrans)
        A = conjTranspose(A);

    if (transB == blas::Op::Trans)
        B = transpose(B);
    else if (transB == blas::Op::ConjTrans)
        B = conjTranspose(B);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gemm");

    slate::gemm(alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });
}

} // namespace scalapack_api
} // namespace slate
