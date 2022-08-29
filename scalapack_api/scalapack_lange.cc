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

// Type generic function calls the SLATE routine
template< typename scalar_t >
blas::real_type<scalar_t> slate_plange(const char* normstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" float PSLANGE(const char* norm, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pslange(const char* norm, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pslange_(const char* norm, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PDLANGE(const char* norm, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlange(const char* norm, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlange_(const char* norm, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" float PCLANGE(const char* norm, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pclange(const char* norm, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pclange_(const char* norm, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PZLANGE(const char* norm, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlange(const char* norm, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlange_(const char* norm, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
blas::real_type<scalar_t> slate_plange(const char* normstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work)
{
    lapack::Norm norm = lapack::char2norm(normstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "lange");

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm(norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    return A_norm;
}

} // namespace scalapack_api
} // namespace slate
