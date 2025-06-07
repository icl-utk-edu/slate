// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

//------------------------------------------------------------------------------
/// SLATE ScaLAPACK wrapper sets up SLATE matrices from ScaLAPACK descriptors
/// and calls SLATE.
template <typename scalar_t>
void slate_pheev(
    const char* jobz_str, const char* uplo_str, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* Lambda,
    scalar_t* Z_data, blas_int iz, blas_int jz, blas_int const* descZ,
    scalar_t* work, blas_int lwork,
    blas::real_type<scalar_t>* rwork, blas_int lrwork,
    blas_int* info )
{
    Uplo uplo{};
    Job jobz{};
    from_string( std::string( 1, uplo_str[0] ), &uplo );
    from_string( std::string( 1, jobz_str[0] ), &jobz );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // todo: extract the real info from heev
    *info = 0;

    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "heev");

    if (lwork == -1 || lrwork == -1) {
        *work = 0;
        *rwork = 0;
        return;
    }

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;
    int64_t Zm = n;
    int64_t Zn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, desc_n( descA ), A_data, desc_lld( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    slate::Matrix<scalar_t> Z;
    if (jobz == lapack::Job::Vec) {
        Cblacs_gridinfo( desc_ctxt( descZ ), &nprow, &npcol, &myprow, &mypcol );
        Z = slate::Matrix<scalar_t>::fromScaLAPACK(
            desc_m( descZ ), desc_n( descZ ), Z_data, desc_lld( descZ ),
            desc_mb( descZ ), desc_nb( descZ ),
            grid_order, nprow, npcol, MPI_COMM_WORLD );
        Z = slate_scalapack_submatrix( Zm, Zn, Z, iz, jz, descZ );
    }

    std::vector< blas::real_type<scalar_t> > Lambda_( n );

    slate::heev( A, Lambda_, Z, {
        {slate::Option::MethodEig, MethodEig::QR},
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    std::copy( Lambda_.begin(), Lambda_.end(), Lambda );
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_pssyev BLAS_FORTRAN_NAME( pssyev, PSSYEV )
void SCALAPACK_pssyev(
    const char* jobz, const char* uplo, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* Lambda,
    float* Z_data, blas_int const* iz, blas_int const* jz, blas_int const* descZ,
    float* work, blas_int const* lwork,
    blas_int* info )
{
    float dummy;
    slate_pheev(
        jobz, uplo, *n,
        A_data, *ia, *ja, descA,
        Lambda,
        Z_data, *iz, *jz, descZ,
        work, *lwork, &dummy, 1, info );
}

#define SCALAPACK_pdsyev BLAS_FORTRAN_NAME( pdsyev, PDSYEV )
void SCALAPACK_pdsyev(
    const char* jobz, const char* uplo, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* Lambda,
    double* Z_data, blas_int const* iz, blas_int const* jz, blas_int const* descZ,
    double* work, blas_int const* lwork,
    blas_int* info )
{
    double dummy;
    slate_pheev(
        jobz, uplo, *n,
        A_data, *ia, *ja, descA,
        Lambda,
        Z_data, *iz, *jz, descZ,
        work, *lwork, &dummy, 1, info );
}

#define SCALAPACK_pcheev BLAS_FORTRAN_NAME( pcheev, PCHEEV )
void SCALAPACK_pcheev(
    const char* jobz, const char* uplo, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* Lambda,
    std::complex<float>* Z_data, blas_int const* iz, blas_int const* jz, blas_int const* descZ,
    std::complex<float>* work, blas_int const* lwork,
    float* rwork, blas_int const* lrwork,
    blas_int* info )
{
    slate_pheev(
        jobz, uplo, *n,
        A_data, *ia, *ja, descA,
        Lambda,
        Z_data, *iz, *jz, descZ,
        work, *lwork, rwork, *lrwork, info );
}

#define SCALAPACK_pzheev BLAS_FORTRAN_NAME( pzheev, PZHEEV )
void SCALAPACK_pzheev(
    const char* jobz, const char* uplo, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* Lambda,
    std::complex<double>* Z_data, blas_int const* iz, blas_int const* jz, blas_int const* descZ,
    std::complex<double>* work, blas_int const* lwork,
    double* rwork, blas_int const* lrwork,
    blas_int* info )
{
    slate_pheev(
        jobz, uplo, *n,
        A_data, *ia, *ja, descA,
        Lambda,
        Z_data, *iz, *jz, descZ,
        work, *lwork, rwork, *lrwork, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
