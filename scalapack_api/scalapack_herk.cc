// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
void slate_pherk(const char* uplostr, const char* transstr, int n, int k, blas::real_type<scalar_t> alpha, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t> beta, scalar_t* c, int ic, int jc, int* descc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pherk

extern "C" void PCHERK(const char* uplo, const char* trans, int* n, int* k, float* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, float* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_pherk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pcherk(const char* uplo, const char* trans, int* n, int* k, float* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, float* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_pherk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pcherk_(const char* uplo, const char* trans, int* n, int* k, float* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, float* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_pherk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PZHERK(const char* uplo, const char* trans, int* n, int* k, double* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, double* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_pherk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pzherk(const char* uplo, const char* trans, int* n, int* k, double* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, double* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_pherk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

extern "C" void pzherk_(const char* uplo, const char* trans, int* n, int* k, double* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, double* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_pherk(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_pherk(const char* uplostr, const char* transstr, int n, int k, blas::real_type<scalar_t> alpha, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t> beta, scalar_t* c, int ic, int jc, int* descc)
{
    check_and_assert_blacs_grid_is_column_major();

    // make blas single threaded
    // todo: does this set the omp num threads correctly
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    int64_t lookahead = 1;

    // setup so op(A) is n-by-k
    int64_t Am = (transA == blas::Op::NoTrans ? n : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : n);
    int64_t Cm = n;
    int64_t Cn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descc), &nprow, &npcol, &myprow, &mypcol);
    auto C = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, desc_N(descc), c, desc_LLD(descc), desc_MB(descc), nprow, npcol, MPI_COMM_WORLD);
    C = slate_scalapack_submatrix(Cm, Cn, C, ic, jc, descc);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "herk");

    if (transA == blas::Op::Trans)
        A = transpose(A);
    else if (transA == blas::Op::ConjTrans)
        A = conjTranspose(A);
    assert(A.mt() == C.mt());

    slate::herk(alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    slate_set_num_blas_threads(saved_num_blas_threads);
}

} // namespace scalapack_api
} // namespace slate
