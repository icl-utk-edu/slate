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
void slate_pgetrf(int m, int n, scalar_t* a, int ia, int ja, int* desca, int* ipiv, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSGETRF(int* m, int* n, float* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void psgetrf(int* m, int* n, float* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void psgetrf_(int* m, int* n, float* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDGETRF(int* m, int* n, double* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void pdgetrf(int* m, int* n, double* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void pdgetrf_(int* m, int* n, double* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCGETRF(int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void pcgetrf(int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void pcgetrf_(int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZGETRF(int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void pzgetrf(int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

extern "C" void pzgetrf_(int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, int* info)
{
    slate_pgetrf(*m, *n, a, *ia, *ja, desca, ipiv, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgetrf(int m, int n, scalar_t* a, int ia, int ja, int* desca, int* ipiv, int* info)
{
    // todo: figure out if the pxq grid is in row or column

    // make blas single threaded
    // todo: does this set the omp num threads correctly
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    int64_t lookahead = 1;
    int64_t panel_threads = slate_scalapack_set_panelthreads();
    int64_t ib = slate_scalapack_set_ib();

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myrow, &mycol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    if (verbose && myrow == 0 && mycol == 0)
        logprintf("%s\n", "getrf");

    slate::getrf(A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    // Extract pivots from SLATE's global Pivots structure into ScaLAPACK local ipiv array
    {
        int64_t p_count = 0;
        int nb = desc_MB(desca);


        // NOTE: this is not the most efficient way, instead use local tile index directly to avoid looping over tiles
        // int64_t A_nt = A.nt();
        // int64_t A_mt = A.mt();
        // for (int tm = 0; tm < A_mt; ++tm) {
        //     for (int tn = 0; tn < A_nt; ++tn) {
        //         if (A.tileIsLocal(tm, tn)) {
        //             for (auto p_iter = pivots[tm].begin(); p_iter != pivots[tm].end(); ++p_iter) {
        //                 ipiv[p_count++] = p_iter->tileIndex() * nb + p_iter->elementOffset() + 1;
        //             }
        //             break;
        //         }
        //     }
        // }

        int ZERO = 0;
        int l_numrows = scalapack_numroc(An, nb, myrow, ZERO, nprow);  // local number of rows
        int l_rindx = 1;// local row index (Scalapack 1-index)
        // find the global tile indices of the local tiles
        while (l_rindx <= l_numrows) {
            int64_t g_rindx = scalapack_indxl2g(&l_rindx, &nb, &myrow, &ZERO, &nprow);
            int64_t g_tile_indx = g_rindx / nb; //assuming uniform tile size from Scalapack
            // extract this tile pivots
            for (auto p_iter = pivots[g_tile_indx].begin();
                 p_iter != pivots[g_tile_indx].end();
                 ++p_iter) {
                ipiv[p_count++] = p_iter->tileIndex() * nb + p_iter->elementOffset() + 1;
            }
            //next tile
            l_rindx += nb;
        }
    }

    slate_set_num_blas_threads(saved_num_blas_threads);

    // todo: extract the real info from getrf
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
