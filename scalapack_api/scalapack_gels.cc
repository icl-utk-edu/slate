// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int* np_row, int* np_col, int*  my_row, int*  my_col);

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_pgels(const char* transstr, int m, int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb, scalar_t* work, int lwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSGELS(const char* trans, int* m, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void psgels(const char* trans, int* m, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void psgels_(const char* trans, int* m, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDGELS(const char* trans, int* m, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void pdgels(const char* trans, int* m, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void pdgels_(const char* trans, int* m, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCGELS(const char* trans, int* m, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void pcgels(const char* trans, int* m, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void pcgels_(const char* trans, int* m, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZGELS(const char* trans, int* m, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void pzgels(const char* trans, int* m, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

extern "C" void pzgels_(const char* trans, int* m, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *ia, *ja, desca, b, *ib, *jb, descb, work, *lwork, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgels(const char* transstr, int m, int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb, scalar_t* work, int lwork, int* info)
{
    using real_t = blas::real_type<scalar_t>;

    // Respond to workspace query with a minimal value (1); workspace
    // is allocated within the SLATE routine.
    if (lwork == -1) {
        work[0] = (real_t)1.0;
        *info = 0;
        return;
    }

    blas::Op trans = blas::char2op(transstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t panel_threads = slate_scalapack_set_panelthreads();
    static int64_t inner_blocking = slate_scalapack_set_ib();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // A is m-by-n, BX is max(m, n)-by-nrhs.
    // If op == NoTrans, op(A) is m-by-n, B is m-by-nrhs
    // otherwise,        op(A) is n-by-m, B is n-by-nrhs.
    int64_t Am = (trans == slate::Op::NoTrans ? m : n);
    int64_t An = (trans == slate::Op::NoTrans ? n : m);
    int64_t Bm = (trans == slate::Op::NoTrans ? m : n);
    int64_t Bn = nrhs;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descb), &nprow, &npcol, &myprow, &mypcol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descb), desc_N(descb), b, desc_LLD(descb), desc_MB(descb), desc_NB(descb), grid_order, nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    // Apply transpose
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        opA = conjTranspose(A);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gels");

    slate::gels(opA, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, inner_blocking}
    });

    // todo: extract the real info
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
