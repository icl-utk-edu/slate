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
blas::real_type<scalar_t> slate_plange(const char* norm_str, blas_int m, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* work);

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" float PSLANGE(const char* norm, blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pslange(const char* norm, blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pslange_(const char* norm, blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------

extern "C" double PDLANGE(const char* norm, blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pdlange(const char* norm, blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pdlange_(const char* norm, blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------

extern "C" float PCLANGE(const char* norm, blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pclange(const char* norm, blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" float pclange_(const char* norm, blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------

extern "C" double PZLANGE(const char* norm, blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pzlange(const char* norm, blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

extern "C" double pzlange_(const char* norm, blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* work)
{
    return slate_plange(norm, *m, *n,
        A_data, *ia, *ja, descA,
        work);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
blas::real_type<scalar_t> slate_plange(const char* norm_str, blas_int m, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* work)
{
    Norm norm{};
    from_string( std::string( 1, norm_str[0] ), &norm );

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
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "lange");

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm( norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    return A_norm;
}

} // namespace scalapack_api
} // namespace slate
