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

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_pgetri(int n, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* work, int lwork, int* iwork, int liwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSGETRI(int* n, float* a, int* ia, int* ja, int* desca, int* ipiv, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void psgetri(int* n, float* a, int* ia, int* ja, int* desca, int* ipiv, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void psgetri_(int* n, float* a, int* ia, int* ja, int* desca, int* ipiv, float* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDGETRI(int* n, double* a, int* ia, int* ja, int* desca, int* ipiv, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void pdgetri(int* n, double* a, int* ia, int* ja, int* desca, int* ipiv, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void pdgetri_(int* n, double* a, int* ia, int* ja, int* desca, int* ipiv, double* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCGETRI(int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void pcgetri(int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void pcgetri_(int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZGETRI(int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void pzgetri(int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

extern "C" void pzgetri_(int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* work, int* lwork, int* iwork, int* liwork, int* info)
{
    slate_pgetri(*n, a, *ia, *ja, desca, ipiv, work, *lwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgetri(int n, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* work, int lwork, int* iwork, int liwork, int* info)
{
    // todo: figure out if the pxq grid is in row or column

    // make blas single threaded
    // todo: does this set the omp num threads correctly
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    int64_t lookahead = slate_scalapack_set_lookahead();
    int64_t panel_threads = slate_scalapack_set_panelthreads();
    int64_t ib = slate_scalapack_set_ib();

    slate::Options const opts = {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(n, n, A, ia, ja, desca);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "getri");

    // std::vector< std::vector<Pivot> >
    slate::Pivots pivots;

    // copy pivots from ScaLAPACK local ipiv array to SLATE global Pivots structure
    {
        // allocate pivots
        int64_t min_mt_nt = std::min(A.mt(), A.nt());
        pivots.resize(min_mt_nt);
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);
        }

        // transfer local ipiv to local part of pivots
        int isrcproc0 = 0;
        int nb = desc_MB(desca); // ScaLAPACK style fixed nb
        int64_t l_numrows = scalapack_numroc(n, nb, myprow, isrcproc0, nprow);  // local number of rows
        // l_rindx local row index (Scalapack 1-index)
        // for each local ipiv entry, find corresponding local-pivot information and swap-pivot information
        for (int l_ipiv_rindx=1; l_ipiv_rindx<=l_numrows; ++l_ipiv_rindx) {
            // for local ipiv index, convert to global indexing
            int64_t g_ipiv_rindx = scalapack_indxl2g(&l_ipiv_rindx, &nb, &myprow, &isrcproc0, &nprow);
            // assuming uniform nb from scalapack (note 1-indexing), find global tile, offset
            int64_t g_ipiv_tile_indx = (g_ipiv_rindx - 1) / nb;
            int64_t g_ipiv_tile_offset = (g_ipiv_rindx -1) % nb;
            // get the reference to this specific pivot
            Pivot& pivref = pivots[g_ipiv_tile_indx][g_ipiv_tile_offset];
            // get swap-pivot information pivots(tile-index, offset)
            // note, slate indexes pivot-tiles from this-point-forward, so subtract earlier tiles.
            int64_t tileIndexSwap = ((ipiv[l_ipiv_rindx - 1] - 1) / nb) - g_ipiv_tile_indx;
            int64_t elementOffsetSwap = (ipiv[l_ipiv_rindx - 1] - 1) % nb;
            // in the local pivot object, assign swap information
            pivref = Pivot(tileIndexSwap, elementOffsetSwap);
            // using lld = long long int;
            // if (verbose)
            //     printf("[%d,%d] getrs ipiv[%lld=%lld]=%lld  ->  pivots[%lld][%lld]=(%lld,%lld)\n",
            //            myprow, mypcol,
            //            (lld)l_ipiv_rindx, (lld)g_ipiv_rindx, (lld)ipiv[l_ipiv_rindx - 1],
            //            (lld)g_ipiv_tile_indx, (lld)g_ipiv_tile_offset, (lld)tileIndexSwap, (lld)elementOffsetSwap);
            // fflush(0);
        }

        // broadcast local pivot information to all processes
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            MPI_Bcast(pivots.at(k).data(),
                      sizeof(Pivot)*pivots.at(k).size(),
                      MPI_BYTE, A.tileRank(k, k), A.mpiComm());
        }
    }

    slate::getri(A, pivots, opts);

    slate_set_num_blas_threads(saved_num_blas_threads);

    // todo: extract the real info from getri
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
