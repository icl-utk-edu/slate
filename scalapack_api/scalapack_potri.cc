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

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_ppotri(const char* uplostr, int n, scalar_t* a, int ia, int ja, int* desca, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSPOTRI(const char* uplo, int* n, float* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pspotri(const char* uplo, int* n, float* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pspotri_(const char* uplo, int* n, float* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDPOTRI(const char* uplo, int* n, double* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pdpotri(const char* uplo, int* n, double* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pdpotri_(const char* uplo, int* n, double* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCPOTRI(const char* uplo, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pcpotri(const char* uplo, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pcpotri_(const char* uplo, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZPOTRI(const char* uplo, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pzpotri(const char* uplo, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pzpotri_(const char* uplo, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotri(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_ppotri(const char* uplostr, int n, scalar_t* a, int ia, int ja, int* desca, int* info)
{
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, desc_N(desca), a, desc_LLD(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(An, An, A, ia, ja, desca);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "potri");

    slate::potri(A, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo: extract the real info from potri
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
