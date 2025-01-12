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
void slate_pgetrf(
    blas_int m, blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas_int* ipiv,
    blas_int* info )
{
    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "getrf");

    slate::getrf( A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    // Extract pivots from SLATE's global Pivots structure into ScaLAPACK local ipiv array
    {
        blas_int isrcproc0 = 0;
        blas_int nb = desc_mb( descA ); // ScaLAPACK style fixed nb
        int64_t l_numrows = scalapack_numroc( An, nb, myprow, isrcproc0, nprow );
        // l_ipiv_rindx local ipiv row index (Scalapack 1-index)
        // for each local ipiv entry, find corresponding local-pivot and swap-pivot
        for (blas_int l_ipiv_rindx=1; l_ipiv_rindx <= l_numrows; ++l_ipiv_rindx) {
            // for ipiv index, convert to global indexing
            int64_t g_ipiv_rindx = scalapack_indxl2g( &l_ipiv_rindx, &nb, &myprow, &isrcproc0, &nprow );
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

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_psgetrf BLAS_FORTRAN_NAME( psgetrf, PSGETRF )
void SCALAPACK_psgetrf(
    blas_int const* m, blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    blas_int* info )
{
    slate_pgetrf(
        *m, *n,
        A_data, *ia, *ja, descA,
        ipiv, info );
}

#define SCALAPACK_pdgetrf BLAS_FORTRAN_NAME( pdgetrf, PDGETRF )
void SCALAPACK_pdgetrf(
    blas_int const* m, blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    blas_int* info )
{
    slate_pgetrf(
        *m, *n,
        A_data, *ia, *ja, descA,
        ipiv, info );
}

#define SCALAPACK_pcgetrf BLAS_FORTRAN_NAME( pcgetrf, PCGETRF )
void SCALAPACK_pcgetrf(
    blas_int const* m, blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    blas_int* info )
{
    slate_pgetrf(
        *m, *n,
        A_data, *ia, *ja, descA,
        ipiv, info );
}

#define SCALAPACK_pzgetrf BLAS_FORTRAN_NAME( pzgetrf, PZGETRF )
void SCALAPACK_pzgetrf(
    blas_int const* m, blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    blas_int* info )
{
    slate_pgetrf(
        *m, *n,
        A_data, *ia, *ja, descA,
        ipiv, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
