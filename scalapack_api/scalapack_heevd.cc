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
void slate_pheevd(const char* jobzstr, const char* uplostr, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* w, scalar_t* z, int iz, int jz, int* descz, scalar_t* work, int lwork, blas::real_type<scalar_t>* rwork, int lrwork, int* iwork, int liwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSSYEVD(const char* jobzstr, const char* uplostr, int* n, float* a, int* ia, int* ja, int* desca, float* w, float* z, int* iz, int* jz, int* descz, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    float dummy;
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, &dummy, 1, iwork, *liwork, info);
}

extern "C" void pssyevd(const char* jobzstr, const char* uplostr, int* n, float* a, int* ia, int* ja, int* desca, float* w, float* z, int* iz, int* jz, int* descz, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    float dummy;
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, &dummy, 1, iwork, *liwork, info);
}

extern "C" void pssyevd_(const char* jobzstr, const char* uplostr, int* n, float* a, int* ia, int* ja, int* desca, float* w, float* z, int* iz, int* jz, int* descz, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    float dummy;
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, &dummy, 1, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDSYEVD(const char* jobzstr, const char* uplostr, int* n, double* a, int* ia, int* ja, int* desca, double* w, double* z, int* iz, int* jz, int* descz, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    double dummy;
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, &dummy, 1, iwork, *liwork, info);
}

extern "C" void pdsyevd(const char* jobzstr, const char* uplostr, int* n, double* a, int* ia, int* ja, int* desca, double* w, double* z, int* iz, int* jz, int* descz, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    double dummy;
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, &dummy, 1, iwork, *liwork, info);
}

extern "C" void pdsyevd_(const char* jobzstr, const char* uplostr, int* n, double* a, int* ia, int* ja, int* desca, double* w, double* z, int* iz, int* jz, int* descz, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    double dummy;
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, &dummy, 1, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCHEEVD(const char* jobzstr, const char* uplostr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* w, std::complex<float>* z, int* iz, int* jz, int* descz, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* iwork, int* liwork, int* info)
{
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}

extern "C" void pcheevd(const char* jobzstr, const char* uplostr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* w, std::complex<float>* z, int* iz, int* jz, int* descz, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* iwork, int* liwork, int* info)
{
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}

extern "C" void pcheevd_(const char* jobzstr, const char* uplostr, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* w, std::complex<float>* z, int* iz, int* jz, int* descz, std::complex<float>* work, int* lwork, float* rwork, int* lrwork, int* iwork, int* liwork, int* info)
{
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZHEEVD(const char* jobzstr, const char* uplostr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* w, std::complex<double>* z, int* iz, int* jz, int* descz, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* iwork, int* liwork, int* info)
{
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}

extern "C" void pzheevd(const char* jobzstr, const char* uplostr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* w, std::complex<double>* z, int* iz, int* jz, int* descz, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* iwork, int* liwork, int* info)
{
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}

extern "C" void pzheevd_(const char* jobzstr, const char* uplostr, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* w, std::complex<double>* z, int* iz, int* jz, int* descz, std::complex<double>* work, int* lwork, double* rwork, int* lrwork, int* iwork, int* liwork, int* info)
{
    slate_pheevd(jobzstr, uplostr, *n, a, *ia, *ja, desca, w, z, *iz, *jz, descz, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pheevd(const char* jobzstr, const char* uplostr, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* w, scalar_t* z, int iz, int jz, int* descz, scalar_t* work, int lwork, blas::real_type<scalar_t>* rwork, int lrwork, int* iwork, int liwork, int* info)
{
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    lapack::Job jobz = lapack::char2job(jobzstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    static int64_t panel_threads = slate_scalapack_set_panelthreads();
    static int64_t ib = slate_scalapack_set_ib();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // todo: extract the real info from heevd
    *info = 0;

    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "heevd");

    if (lwork == -1 || lrwork == -1 || liwork == -1) {
        *work = 0;
        *rwork = 0;
        *iwork = 0;
        return;
    }

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;
    int64_t Zm = n;
    int64_t Zn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, desc_N(desca), a, desc_LLD(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    slate::Matrix<scalar_t> Z;
    if (jobz == lapack::Job::Vec) {
        Cblacs_gridinfo(desc_CTXT(descz), &nprow, &npcol, &myprow, &mypcol);
        Z = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descz), desc_N(descz), z, desc_LLD(descz), desc_MB(descz), desc_NB(descz), grid_order, nprow, npcol, MPI_COMM_WORLD);
        Z = slate_scalapack_submatrix(Zm, Zn, Z, iz, jz, descz);
    }

    std::vector< blas::real_type<scalar_t> > Lambda_( n );

    slate::heev( A, Lambda_, Z, {
        {slate::Option::MethodEig, MethodEig::DC},
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    std::copy(Lambda_.begin(), Lambda_.end(), w);
}

} // namespace scalapack_api
} // namespace slate
