// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "scalapack_slate.hh"

#include <complex>

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------
// Type generic function calls the SLATE routine
template< typename scalar_t >
blas::real_type<scalar_t> slate_plantr(const char* normstr, const char* uplostr, const char* diagstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" float PSLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pslantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pslantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PDLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" float PCLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pclantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" float pclantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PZLANTR(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlantr(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlantr_(const char* norm, const char* uplo, const char* diag, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
blas::real_type<scalar_t> slate_plantr(const char* normstr, const char* uplostr, const char* diagstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work)
{
    Norm norm{};
    Uplo uplo{};
    Diag diag{};
    from_string( std::string( 1, normstr[0] ), &norm );
    from_string( std::string( 1, uplostr[0] ), &uplo );
    from_string( std::string( 1, diagstr[0] ), &diag );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(uplo, diag, desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s target %d\n", "lantr", (int)target);

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm(norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    return A_norm;
}

} // namespace scalapack_api
} // namespace slate
