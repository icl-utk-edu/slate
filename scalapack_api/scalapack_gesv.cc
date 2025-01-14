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
void slate_pgesv(
    blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas_int* ipiv,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB,
    blas_int* info )
{
    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t inner_blocking = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;
    int64_t Bm = n;
    int64_t Bn = nrhs;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    Cblacs_gridinfo( desc_ctxt( descB ), &nprow, &npcol, &myprow, &mypcol );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descB ), desc_n( descB ), B_data, desc_lld( descB ),
        desc_mb( descB ), desc_nb( descB ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    B = slate_scalapack_submatrix( Bm, Bn, B, ib, jb, descB );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gesv");

    slate::gesv( A, pivots, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, inner_blocking}
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

    // todo: extract the real info
    *info = 0;
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_psgesv BLAS_FORTRAN_NAME( psgesv, PSGESV )
void SCALAPACK_psgesv(
    blas_int const* n, blas_int* nrhs,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgesv( *n, *nrhs,
                 A_data, *ia, *ja, descA, ipiv,
                 B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pdgesv BLAS_FORTRAN_NAME( pdgesv, PDGESV )
void SCALAPACK_pdgesv(
    blas_int const* n, blas_int* nrhs,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgesv( *n, *nrhs,
                 A_data, *ia, *ja, descA, ipiv,
                 B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pcgesv BLAS_FORTRAN_NAME( pcgesv, PCGESV )
void SCALAPACK_pcgesv(
    blas_int const* n, blas_int* nrhs,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgesv( *n, *nrhs,
                 A_data, *ia, *ja, descA, ipiv,
                 B_data, *ib, *jb, descB, info );
}

#define SCALAPACK_pzgesv BLAS_FORTRAN_NAME( pzgesv, PZGESV )
void SCALAPACK_pzgesv(
    blas_int const* n, blas_int* nrhs,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgesv( *n, *nrhs,
                 A_data, *ia, *ja, descA, ipiv,
                 B_data, *ib, *jb, descB, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
