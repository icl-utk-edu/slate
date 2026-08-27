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
void slate_pgetrs(
    const char* trans_str, blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    blas_int* ipiv,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB,
    blas_int* info )
{
    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    Op trans{};
    from_string( std::string( 1, trans_str[0] ), &trans );

    // Matrix sizes
    int64_t Am = n;
    int64_t An = n;
    int64_t Bm = n;
    int64_t Bn = nrhs;

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
        logprintf("%s\n", "getrs");

    // std::vector< std::vector<Pivot> >
    slate::Pivots pivots;

    // copy pivots from ScaLAPACK local ipiv array to SLATE global Pivots structure
    {
        // allocate pivots
        int64_t min_mt_nt = std::min( A.mt(), A.nt() );
        pivots.resize( min_mt_nt );
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            int64_t diag_len = std::min( A.tileMb(k), A.tileNb(k) );
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
            MPI_Bcast( pivots.at( k ).data(),
                       sizeof(Pivot)*pivots.at( k ).size(),
                       MPI_BYTE, A.tileRank( k, k ), A.mpiComm() );
        }
    }

    // apply operators to the matrix
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose( A );
    else if (trans == slate::Op::ConjTrans)
        opA = conj_transpose( A );

    // call the SLATE getrs routine
    slate::getrs( opA, pivots, B, opts);

    // todo: extract the real info from getrs
    *info = 0;
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_psgetrs BLAS_FORTRAN_NAME( psgetrs, PSGETRS )
void SCALAPACK_psgetrs(
    const char* trans, blas_int const* n, blas_int* nrhs,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgetrs(
        trans, *n, *nrhs,
        A_data, *ia, *ja, descA,
        ipiv,
        B_data, *ib, *jb, descB,
        info );
}

#define SCALAPACK_pdgetrs BLAS_FORTRAN_NAME( pdgetrs, PDGETRS )
void SCALAPACK_pdgetrs(
    const char* trans, blas_int const* n, blas_int* nrhs,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgetrs(
        trans, *n, *nrhs,
        A_data, *ia, *ja, descA,
        ipiv,
        B_data, *ib, *jb, descB,
        info );
}

#define SCALAPACK_pcgetrs BLAS_FORTRAN_NAME( pcgetrs, PCGETRS )
void SCALAPACK_pcgetrs(
    const char* trans, blas_int const* n, blas_int* nrhs,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgetrs(
        trans, *n, *nrhs,
        A_data, *ia, *ja, descA,
        ipiv,
        B_data, *ib, *jb, descB,
        info );
}

#define SCALAPACK_pzgetrs BLAS_FORTRAN_NAME( pzgetrs, PZGETRS )
void SCALAPACK_pzgetrs(
    const char* trans, blas_int const* n, blas_int* nrhs,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    blas_int* ipiv,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    blas_int* info )
{
    slate_pgetrs(
        trans, *n, *nrhs,
        A_data, *ia, *ja, descA,
        ipiv,
        B_data, *ib, *jb, descB,
        info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
