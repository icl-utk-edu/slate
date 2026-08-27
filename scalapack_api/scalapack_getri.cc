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
void slate_pgetri(
    blas_int n,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas_int* ipiv,
    scalar_t* work, blas_int lwork,
    blas_int* iwork, blas_int liwork,
    blas_int* info )
{
    slate::Target target = TargetConfig::value( );
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t ib = IBConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    slate::Options const opts = {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( n, n, A, ia, ja, descA );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "getri");

    // std::vector< std::vector<Pivot> >
    slate::Pivots pivots;

    // copy pivots from ScaLAPACK local ipiv array to SLATE global Pivots structure
    {
        // allocate pivots
        int64_t min_mt_nt = std::min( A.mt(), A.nt() );
        pivots.resize( min_mt_nt );
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            int64_t diag_len = std::min( A.tileMb( k ), A.tileNb( k ) );
            pivots.at( k ).resize( diag_len );
        }

        // transfer local ipiv to local part of pivots
        blas_int isrcproc0 = 0;
        blas_int nb = desc_mb( descA ); // ScaLAPACK style fixed nb
        int64_t l_numrows = scalapack_numroc( n, nb, myprow, isrcproc0, nprow );  // local number of rows
        // l_rindx local row index (Scalapack 1-index)
        // for each local ipiv entry, find corresponding local-pivot information and swap-pivot information
        for (blas_int l_ipiv_rindx=1; l_ipiv_rindx <= l_numrows; ++l_ipiv_rindx) {
            // for local ipiv index, convert to global indexing
            int64_t g_ipiv_rindx = scalapack_indxl2g( &l_ipiv_rindx, &nb, &myprow, &isrcproc0, &nprow );
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
            pivref = Pivot( tileIndexSwap, elementOffsetSwap );
            // if (verbose) {
            //     printf("[%d,%d] getrs ipiv[%lld=%lld]=%lld  ->  pivots[%lld][%lld]=(%lld,%lld)\n",
            //            myprow, mypcol,
            //            llong( l_ipiv_rindx ), llong( g_ipiv_rindx ),
            //            llong( ipiv[l_ipiv_rindx - 1] ),
            //            llong( g_ipiv_tile_indx ), llong( g_ipiv_tile_offset ),
            //            llong( tileIndexSwap ), llong( elementOffsetSwap ));
            // }
            // fflush( 0 );
        }

        // broadcast local pivot information to all processes
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            MPI_Bcast(pivots.at(k).data(),
                      sizeof(Pivot)*pivots.at(k).size(),
                      MPI_BYTE, A.tileRank( k, k ), A.mpiComm() );
        }
    }

    slate::getri( A, pivots, opts);

    // todo: extract the real info from getri
    *info = 0;
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_psgetri BLAS_FORTRAN_NAME( psgetri, PSGETRI )
void SCALAPACK_psgetri(
    blas_int const* n,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    float* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_pgetri(
        *n,
        A_data, *ia, *ja, descA,
        ipiv, work, *lwork, iwork, *liwork, info );
}

#define SCALAPACK_pdgetri BLAS_FORTRAN_NAME( pdgetri, PDGETRI )
void SCALAPACK_pdgetri(
    blas_int const* n,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    double* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_pgetri(
        *n,
        A_data, *ia, *ja, descA,
        ipiv, work, *lwork, iwork, *liwork, info );
}

#define SCALAPACK_pcgetri BLAS_FORTRAN_NAME( pcgetri, PCGETRI )
void SCALAPACK_pcgetri(
    blas_int const* n,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    std::complex<float>* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_pgetri(
        *n,
        A_data, *ia, *ja, descA,
        ipiv, work, *lwork, iwork, *liwork, info );
}

#define SCALAPACK_pzgetri BLAS_FORTRAN_NAME( pzgetri, PZGETRI )
void SCALAPACK_pzgetri(
    blas_int const* n,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    std::complex<double>* work, blas_int const* lwork,
    blas_int* iwork, blas_int const* liwork,
    blas_int* info )
{
    slate_pgetri(
        *n,
        A_data, *ia, *ja, descA,
        ipiv, work, *lwork, iwork, *liwork, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
