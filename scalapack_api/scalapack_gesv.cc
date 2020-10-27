// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int* np_row, int* np_col, int*  my_row, int*  my_col);

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_pgesv(int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* b, int ib, int jb, int* descb, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSGESV(int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, int* ipiv, float* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void psgesv(int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, int* ipiv, float* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void psgesv_(int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, int* ipiv, float* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDGESV(int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pdgesv(int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pdgesv_(int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCGESV(int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pcgesv(int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pcgesv_(int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZGESV(int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pzgesv(int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pzgesv_(int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgesv(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgesv(int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* b, int ib, int jb, int* descb, int* info)
{
    // todo: figure out if the pxq grid is in row or column

    // make blas single threaded
    // todo: does this set the omp num threads correctly
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    int64_t lookahead = 1;
    int64_t panel_threads = slate_scalapack_set_panelthreads();
    int64_t inner_blocking = slate_scalapack_set_ib();

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;
    int64_t Bm = n;
    int64_t Bn = nrhs;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myrow, &mycol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descb), &nprow, &npcol, &myrow, &mycol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descb), desc_N(descb), b, desc_LLD(descb), desc_MB(descb), nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    if (verbose && myrow == 0 && mycol == 0)
        logprintf("%s\n", "gesv");

    slate::gesv(A, pivots, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, inner_blocking}
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
                 ++p_iter)
                ipiv[p_count++] = p_iter->tileIndex() * nb + p_iter->elementOffset() + 1;
            //next tile
            l_rindx += nb;
        }
    }

    slate_set_num_blas_threads(saved_num_blas_threads);

    // todo: extract the real info
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
