// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
void slate_pgesvd(const char* jobustr, const char* jobvtstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* s, scalar_t* u, int iu, int ju, int* descu, scalar_t* vt, int ivt, int jvt, int* descvt, scalar_t* work, int lwork, blas::real_type<scalar_t>* rwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSGESVD(const char* jobustr, const char* jobvtstr, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* s, float* u, int* iu, int* ju, int* descu, float* vt, int* ivt, int* jvt, int* descvt, float* work, int* lwork, int* info)
{
    float dummy;
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, &dummy, info);
}

extern "C" void psgesvd(const char* jobustr, const char* jobvtstr, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* s, float* u, int* iu, int* ju, int* descu, float* vt, int* ivt, int* jvt, int* descvt, float* work, int* lwork, int* info)
{
    float dummy;
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, &dummy, info);
}

extern "C" void psgesvd_(const char* jobustr, const char* jobvtstr, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* s, float* u, int* iu, int* ju, int* descu, float* vt, int* ivt, int* jvt, int* descvt, float* work, int* lwork, int* info)
{
    float dummy;
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, &dummy, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDGESVD(const char* jobustr, const char* jobvtstr, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* s, double* u, int* iu, int* ju, int* descu, double* vt, int* ivt, int* jvt, int* descvt, double* work, int* lwork, int* info)
{
    double dummy;
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, &dummy, info);
}

extern "C" void pdgesvd(const char* jobustr, const char* jobvtstr, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* s, double* u, int* iu, int* ju, int* descu, double* vt, int* ivt, int* jvt, int* descvt, double* work, int* lwork, int* info)
{
    double dummy;
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, &dummy, info);
}

extern "C" void pdgesvd_(const char* jobustr, const char* jobvtstr, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* s, double* u, int* iu, int* ju, int* descu, double* vt, int* ivt, int* jvt, int* descvt, double* work, int* lwork, int* info)
{
    double dummy;
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, &dummy, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCGESVD(const char* jobustr, const char* jobvtstr, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* s, std::complex<float>* u, int* iu, int* ju, int* descu, std::complex<float>* vt, int* ivt, int* jvt, int* descvt, std::complex<float>* work, int* lwork, float* rwork, int* info)
{
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, rwork, info);
}

extern "C" void pcgesvd(const char* jobustr, const char* jobvtstr, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* s, std::complex<float>* u, int* iu, int* ju, int* descu, std::complex<float>* vt, int* ivt, int* jvt, int* descvt, std::complex<float>* work, int* lwork, float* rwork, int* info)
{
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, rwork, info);
}

extern "C" void pcgesvd_(const char* jobustr, const char* jobvtstr, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* s, std::complex<float>* u, int* iu, int* ju, int* descu, std::complex<float>* vt, int* ivt, int* jvt, int* descvt, std::complex<float>* work, int* lwork, float* rwork, int* info)
{
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, rwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZGESVD(const char* jobustr, const char* jobvtstr, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* s, std::complex<float>* u, int* iu, int* ju, int* descu, std::complex<float>* vt, int* ivt, int* jvt, int* descvt, std::complex<float>* work, int* lwork, float* rwork, int* info)
{
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, rwork, info);
}

extern "C" void pzgesvd_(const char* jobustr, const char* jobvtstr, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* s, std::complex<float>* u, int* iu, int* ju, int* descu, std::complex<float>* vt, int* ivt, int* jvt, int* descvt, std::complex<float>* work, int* lwork, float* rwork, int* info)
{
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, rwork, info);
}

extern "C" void pzgesvd(const char* jobustr, const char* jobvtstr, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* s, std::complex<float>* u, int* iu, int* ju, int* descu, std::complex<float>* vt, int* ivt, int* jvt, int* descvt, std::complex<float>* work, int* lwork, float* rwork, int* info)
{
    slate_pgesvd(jobustr, jobvtstr, *m, *n, a, *ia, *ja, desca, s, u, *iu, *ju, descu, vt, *ivt, *jvt, descvt, work, *lwork, rwork, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgesvd(const char* jobustr, const char* jobvtstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* s, scalar_t* u, int iu, int ju, int* descu, scalar_t* vt, int ivt, int jvt, int* descvt, scalar_t* work, int lwork, blas::real_type<scalar_t>* rwork, int* info)
{
    Job jobu{};
    Job jobvt{};
    from_string( std::string( 1, jobustr[0] ), &jobu );
    from_string( std::string( 1, jobvtstr[0] ), &jobvt );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // todo: extract the real info from gesvd
    *info = 0;

    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gesvd");

    if (lwork == -1) {
        // ScaLAPACK work request is the minimum.  We can allocate, minimum is 0
        *work = 0;
        *rwork = 0;
        return;
    }

    // Matrix sizes
    int64_t min_mn = std::min(m, n);
    int64_t Am = m;
    int64_t An = n;
    int64_t Um = m;
    int64_t Un = min_mn;
    int64_t VTm = min_mn;
    int64_t VTn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    slate::Matrix<scalar_t> U;
    if (jobu == lapack::Job::Vec) {
        Cblacs_gridinfo(desc_CTXT(descu), &nprow, &npcol, &myprow, &mypcol);
        U = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descu), desc_N(descu), u, desc_LLD(descu), desc_MB(descu), desc_NB(descu), grid_order, nprow, npcol, MPI_COMM_WORLD);
        U = slate_scalapack_submatrix(Um, Un, U, iu, ju, descu);
    }

    slate::Matrix<scalar_t> VT;
    if (jobvt == lapack::Job::Vec) {
        Cblacs_gridinfo(desc_CTXT(descvt), &nprow, &npcol, &myprow, &mypcol);
        VT = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descvt), desc_N(descvt), vt, desc_LLD(descvt), desc_MB(descvt), desc_NB(descvt), grid_order, nprow, npcol, MPI_COMM_WORLD);
        VT = slate_scalapack_submatrix(VTm, VTn, VT, ivt, jvt, descvt);
    }

    std::vector< blas::real_type<scalar_t> > Sigma_( n );

    slate::svd( A, Sigma_, U, VT, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    std::copy(Sigma_.begin(), Sigma_.end(), s);
}

} // namespace scalapack_api
} // namespace slate
