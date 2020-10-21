// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
void slate_ppotrf(const char* uplostr, int n, scalar_t* a, int ia, int ja, int* desca, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSPOTRF(const char* uplo, int* n, float* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pspotrf(const char* uplo, int* n, float* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pspotrf_(const char* uplo, int* n, float* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDPOTRF(const char* uplo, int* n, double* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pdpotrf(const char* uplo, int* n, double* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pdpotrf_(const char* uplo, int* n, double* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCPOTRF(const char* uplo, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pcpotrf(const char* uplo, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pcpotrf_(const char* uplo, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZPOTRF(const char* uplo, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pzpotrf(const char* uplo, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

extern "C" void pzpotrf_(const char* uplo, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* info)
{
    slate_ppotrf(uplo, *n, a, *ia, *ja, desca, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_ppotrf(const char* uplostr, int n, scalar_t* a, int ia, int ja, int* desca, int* info)
{
    // todo: figure out if the pxq grid is in row or column

    // make blas single threaded
    // todo: does this set the omp num threads correctly
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    int64_t lookahead = 1;

    // Matrix sizes
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myrow, &mycol);
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, desc_N(desca), a, desc_LLD(desca), desc_MB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(An, An, A, ia, ja, desca);

    if (verbose && myrow == 0 && mycol == 0)
        logprintf("%s\n", "potrf");

    slate::potrf(A, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    slate_set_num_blas_threads(saved_num_blas_threads);

    // todo: extract the real info from potrf
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
