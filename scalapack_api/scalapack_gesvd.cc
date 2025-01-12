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
void slate_pgesvd(
    const char* jobu_str, const char* jobvt_str, blas_int m, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas::real_type<scalar_t>* Sigma,
    scalar_t* U_data, blas_int iu, blas_int ju, blas_int const* descU,
    scalar_t* VT_data, blas_int ivt, blas_int jvt, blas_int const* descVT,
    scalar_t* work, blas_int lwork,
    blas::real_type<scalar_t>* rwork,
    blas_int* info )
{
    Job jobu{};
    Job jobvt{};
    from_string( std::string( 1, jobu_str[0] ), &jobu );
    from_string( std::string( 1, jobvt_str[0] ), &jobvt );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // todo: extract the real info from gesvd
    *info = 0;

    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gesvd");

    if (lwork == -1) {
        // ScaLAPACK work request is the minimum.  We can allocate, minimum is 0
        *work = 0;
        *rwork = 0;
        return;
    }

    // Matrix sizes
    int64_t min_mn = std::min( m, n );
    int64_t Am = m;
    int64_t An = n;
    int64_t Um = m;
    int64_t Un = min_mn;
    int64_t VTm = min_mn;
    int64_t VTn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    slate::Matrix<scalar_t> U;
    if (jobu == lapack::Job::Vec) {
        Cblacs_gridinfo( desc_ctxt( descU ), &nprow, &npcol, &myprow, &mypcol );
        U = slate::Matrix<scalar_t>::fromScaLAPACK(
            desc_m( descU ), desc_n( descU ), U_data, desc_lld( descU ),
            desc_mb( descU ), desc_nb( descU ),
            grid_order, nprow, npcol, MPI_COMM_WORLD );
        U = slate_scalapack_submatrix( Um, Un, U, iu, ju, descU );
    }

    slate::Matrix<scalar_t> VT;
    if (jobvt == lapack::Job::Vec) {
        Cblacs_gridinfo( desc_ctxt( descVT ), &nprow, &npcol, &myprow, &mypcol );
        VT = slate::Matrix<scalar_t>::fromScaLAPACK(
            desc_m( descVT ), desc_n( descVT ), VT_data, desc_lld( descVT ),
            desc_mb( descVT ), desc_nb( descVT ),
            grid_order, nprow, npcol, MPI_COMM_WORLD );
        VT = slate_scalapack_submatrix( VTm, VTn, VT, ivt, jvt, descVT );
    }

    std::vector< blas::real_type<scalar_t> > Sigma_( n );

    slate::svd( A, Sigma_, U, VT, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    std::copy( Sigma_.begin(), Sigma_.end(), Sigma );
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_psgesvd BLAS_FORTRAN_NAME( psgesvd, PSGESVD )
void SCALAPACK_psgesvd(
    const char* jobu_str, const char* jobvt_str, blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* Sigma,
    float* U_data, blas_int const* iu, blas_int const* ju, blas_int const* descU,
    float* VT_data, blas_int const* ivt, blas_int const* jvt, blas_int const* descVT,
    float* work, blas_int const* lwork,
    blas_int* info )
{
    float dummy;
    slate_pgesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *ia, *ja, descA,
        Sigma,
        U_data, *iu, *ju, descU,
        VT_data, *ivt, *jvt, descVT,
        work, *lwork, &dummy, info );
}

#define SCALAPACK_pdgesvd BLAS_FORTRAN_NAME( pdgesvd, PDGESVD )
void SCALAPACK_pdgesvd(
    const char* jobu_str, const char* jobvt_str, blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* Sigma,
    double* U_data, blas_int const* iu, blas_int const* ju, blas_int const* descU,
    double* VT_data, blas_int const* ivt, blas_int const* jvt, blas_int const* descVT,
    double* work, blas_int const* lwork,
    blas_int* info )
{
    double dummy;
    slate_pgesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *ia, *ja, descA,
        Sigma,
        U_data, *iu, *ju, descU,
        VT_data, *ivt, *jvt, descVT,
        work, *lwork, &dummy, info );
}

#define SCALAPACK_pcgesvd BLAS_FORTRAN_NAME( pcgesvd, PCGESVD )
void SCALAPACK_pcgesvd(
    const char* jobu_str, const char* jobvt_str, blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* Sigma,
    std::complex<float>* U_data, blas_int const* iu, blas_int const* ju, blas_int const* descU,
    std::complex<float>* VT_data, blas_int const* ivt, blas_int const* jvt, blas_int const* descVT,
    std::complex<float>* work, blas_int const* lwork,
    float* rwork,
    blas_int* info )
{
    slate_pgesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *ia, *ja, descA,
        Sigma,
        U_data, *iu, *ju, descU,
        VT_data, *ivt, *jvt, descVT,
        work, *lwork, rwork, info );
}

#define SCALAPACK_pzgesvd BLAS_FORTRAN_NAME( pzgesvd, PZGESVD )
void SCALAPACK_pzgesvd(
    const char* jobu_str, const char* jobvt_str, blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* Sigma,
    std::complex<double>* U_data, blas_int const* iu, blas_int const* ju, blas_int const* descU,
    std::complex<double>* VT_data, blas_int const* ivt, blas_int const* jvt, blas_int const* descVT,
    std::complex<double>* work, blas_int const* lwork,
    double* rwork,
    blas_int* info )
{
    slate_pgesvd(
        jobu_str, jobvt_str, *m, *n,
        A_data, *ia, *ja, descA,
        Sigma,
        U_data, *iu, *ju, descU,
        VT_data, *ivt, *jvt, descVT,
        work, *lwork, rwork, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
