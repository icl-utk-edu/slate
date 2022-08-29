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
void slate_pgetrs(const char* transstr, int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* b, int ib, int jb, int* descb, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" void PSGETRS(const char* trans, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, int* ipiv, float* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void psgetrs(const char* trans, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, int* ipiv, float* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void psgetrs_(const char* trans, int* n, int* nrhs, float* a, int* ia, int* ja, int* desca, int* ipiv, float* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------

extern "C" void PDGETRS(const char* trans, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pdgetrs(const char* trans, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pdgetrs_(const char* trans, int* n, int* nrhs, double* a, int* ia, int* ja, int* desca, int* ipiv, double* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------

extern "C" void PCGETRS(const char* trans, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pcgetrs(const char* trans, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pcgetrs_(const char* trans, int* n, int* nrhs, std::complex<float>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<float>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------

extern "C" void PZGETRS(const char* trans, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pzgetrs(const char* trans, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

extern "C" void pzgetrs_(const char* trans, int* n, int* nrhs, std::complex<double>* a, int* ia, int* ja, int* desca, int* ipiv, std::complex<double>* b, int* ib, int* jb, int* descb, int* info)
{
    slate_pgetrs(trans, *n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, info);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void slate_pgetrs(const char* transstr, int n, int nrhs, scalar_t* a, int ia, int ja, int* desca, int* ipiv, scalar_t* b, int ib, int jb, int* descb, int* info)
{
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    static int64_t lookahead = slate_scalapack_set_lookahead();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    };

    // Matrix sizes
    blas::Op trans = blas::char2op(transstr[0]);
    int64_t Am = n;
    int64_t An = n;
    int64_t Bm = n;
    int64_t Bn = nrhs;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myprow, &mypcol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), desc_NB(desca), grid_order, nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descb), &nprow, &npcol, &myprow, &mypcol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descb), desc_N(descb), b, desc_LLD(descb), desc_MB(descb), desc_NB(descb), grid_order, nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "getrs");

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
        for (int l_ipiv_rindx=1; l_ipiv_rindx <= l_numrows; ++l_ipiv_rindx) {
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
            // if (verbose) {
            //     printf("[%d,%d] getrs ipiv[%lld=%lld]=%lld  ->  pivots[%lld][%lld]=(%lld,%lld)\n",
            //            myprow, mypcol,
            //            llong( l_ipiv_rindx ), llong( g_ipiv_rindx ),
            //            llong( ipiv[l_ipiv_rindx - 1] ),
            //            llong( g_ipiv_tile_indx ), llong( g_ipiv_tile_offset ),
            //            llong( tileIndexSwap ), llong( elementOffsetSwap ));
            // }
            // fflush(0);
        }

        // broadcast local pivot information to all processes
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            MPI_Bcast(pivots.at(k).data(),
                      sizeof(Pivot)*pivots.at(k).size(),
                      MPI_BYTE, A.tileRank(k, k), A.mpiComm());
        }
    }

    // apply operators to the matrix
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        opA = conjTranspose(A);

    // call the SLATE getrs routine
    slate::getrs(opA, pivots, B, opts);

    // todo: extract the real info from getrs
    *info = 0;
}

} // namespace scalapack_api
} // namespace slate
