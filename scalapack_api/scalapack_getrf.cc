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
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    static int64_t panel_threads = slate_scalapack_set_panelthreads();
    static int64_t ib = slate_scalapack_set_ib();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "getrf");

    slate::getrf(A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    // Extract pivots from SLATE's global Pivots structure into ScaLAPACK local ipiv array
    {
        int isrcproc0 = 0;
        int nb = desc_MB(desca); // ScaLAPACK style fixed nb
        int64_t l_numrows = scalapack_numroc(An, nb, myprow, isrcproc0, nprow);
        // l_ipiv_rindx local ipiv row index (Scalapack 1-index)
        // for each local ipiv entry, find corresponding local-pivot and swap-pivot
        for (int l_ipiv_rindx=1; l_ipiv_rindx <= l_numrows; ++l_ipiv_rindx) {
            // for ipiv index, convert to global indexing
            int64_t g_ipiv_rindx = scalapack_indxl2g(&l_ipiv_rindx, &nb, &myprow, &isrcproc0, &nprow);
            // assuming uniform nb from scalapack (note 1-indexing)
            // figure out pivots(tile-index, offset)
            int64_t g_ipiv_tile_indx = (g_ipiv_rindx - 1) / nb;
            int64_t g_ipiv_tile_offset = (g_ipiv_rindx -1 ) % nb;
            // get the reference to pivot corresponding to current ipiv
            Pivot pivot = pivots[g_ipiv_tile_indx][g_ipiv_tile_offset];
            // get swap information from pivot
            int64_t tileIndexSwap = pivot.tileIndex();
            int64_t elementOffsetSwap = pivot.elementOffset();
            // scalapack 1-index
            // pivots reference local submatrix; so shift by g_ipiv_tile_indx
            ipiv[l_ipiv_rindx-1] = ((tileIndexSwap+g_ipiv_tile_indx) * nb) + (elementOffsetSwap + 1);
        }
    }

    // todo: extract the real info from getrf
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
