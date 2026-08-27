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
blas::real_type<scalar_t> slate_plantr(const char* norm_str, const char* uplo_str, const char* diag_str, blas_int m, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* work);

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" float PSLANTR(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pslantr(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pslantr_(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------

extern "C" double PDLANTR(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pdlantr(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pdlantr_(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------

extern "C" float PCLANTR(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pclantr(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pclantr_(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------

extern "C" double PZLANTR(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pzlantr(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pzlantr_(const char* norm, const char* uplo, const char* diag, blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plantr(norm, uplo, diag, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
blas::real_type<scalar_t> slate_plantr(const char* norm_str, const char* uplo_str, const char* diag_str, blas_int m, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* work)
{
    Norm norm{};
    Uplo uplo{};
    Diag diag{};
    from_string( std::string( 1, norm_str[0] ), &norm );
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, diag_str[0] ), &diag );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(
        uplo, diag, desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s target %d\n", "lantr", (blas_int)target);

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm( norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    return A_norm;
}

} // namespace scalapack_api
} // namespace slate
