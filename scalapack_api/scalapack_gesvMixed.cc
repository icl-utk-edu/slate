// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
void slate_pgesvMixed(int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* b, int ib, int jb, int* descb, scalar_t* x, int ix, int jx, int* descx, int* iter, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

// -----------------------------------------------------------------------------

extern "C" void PDSGESV(int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, double* x, int* ix, int* jx, int* descx, int* iter, int* info)
{
    slate_pgesvMixed(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, x, *ix, *jx, descx, iter, info);
}

extern "C" void pdsgesv(int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, double* x, int* ix, int* jx, int* descx, int* iter, int* info)
{
    slate_pgesvMixed(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, x, *ix, *jx, descx, iter, info);
}

extern "C" void pdsgesv_(int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, double* x, int* ix, int* jx, int* descx, int* iter, int* info)
{
    slate_pgesvMixed(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, x, *ix, *jx, descx, iter, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZCGESV(int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* x, int* ix, int* jx, int* descx, int* iter, int* info)
{
    slate_pgesvMixed(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, x, *ix, *jx, descx, iter, info);
}

extern "C" void pzcgesv(int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* x, int* ix, int* jx, int* descx, int* iter, int* info)
{
    slate_pgesvMixed(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, x, *ix, *jx, descx, iter, info);
}

extern "C" void pzcgesv_(int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* x, int* ix, int* jx, int* descx, int* iter, int* info)
{
    slate_pgesvMixed(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, x, *ix, *jx, descx, iter, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgesvMixed(int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* b, int ib, int jb, int* descb, scalar_t* x, int ix, int jx, int* descx, int* iter, int* info)
{
    using real_t = blas::real_type<scalar_t>;

    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    static int64_t panel_threads = slate_scalapack_set_panelthreads();
    static int64_t inner_blocking = slate_scalapack_set_ib();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;
    int64_t Bm = n;
    int64_t Bn = nrhs;
    int64_t Xm = n;
    int64_t Xn = nrhs;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descb), &nprow, &npcol, &myprow, &mypcol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descb), desc_N(descb), b, desc_LLD(descb), desc_MB(descb), desc_NB(descb), grid_order, nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    Cblacs_gridinfo(desc_CTXT(descx), &nprow, &npcol, &myprow, &mypcol);
    auto X = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descx), desc_N(descx), x, desc_LLD(descx), desc_MB(descx), desc_NB(descb), grid_order, nprow, npcol, MPI_COMM_WORLD);
    X = slate_scalapack_submatrix(Xm, Xn, X, ix, jx, descx);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gesvMixed");

    if (std::is_same<real_t, double>::value) {
        slate::gesvMixed(A, pivots, B, X, *iter, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target},
                {slate::Option::MaxPanelThreads, panel_threads},
                {slate::Option::InnerBlocking, inner_blocking}
            });
    }

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

    // todo: extract the real info
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
