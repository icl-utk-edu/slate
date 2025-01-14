// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

//------------------------------------------------------------------------------
// Type generic function calls the SLATE routine
template <typename scalar_t>
blas::real_type<scalar_t> slate_planhe(const char* norm_str, const char* uplo_str, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* work);

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

//------------------------------------------------------------------------------

extern "C" float PCLANHE(const char* norm, const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_planhe(norm, uplo, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pclanhe(const char* norm, const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_planhe(norm, uplo, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pclanhe_(const char* norm, const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_planhe(norm, uplo, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------

extern "C" double PZLANHE(const char* norm, const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_planhe(norm, uplo, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pzlanhe(const char* norm, const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_planhe(norm, uplo, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pzlanhe_(const char* norm, const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_planhe(norm, uplo, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
blas::real_type<scalar_t> slate_planhe(const char* norm_str, const char* uplo_str, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* work)
{
    Uplo uplo{};
    Norm norm{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, norm_str[0] ), &norm );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "lanhe");

    blas::real_type<scalar_t> A_norm = 1.0;
    A_norm = slate::norm( norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    return A_norm;
}

} // namespace scalapack_api
} // namespace slate
