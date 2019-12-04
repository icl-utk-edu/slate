//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/slate.hh"
#include "scalapack_slate.hh"
#include <complex>

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
    slate_pgesvMixed(*n, *nrhs, a, *ia, *ja, desca, ipiv, b, *ib, *jb, descb, x, *ix, *jx, descx, iter, info);;
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
    int64_t Xm = n;
    int64_t Xn = nrhs;
    slate::Pivots pivots;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myrow, &mycol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(desc_CTXT(descb), &nprow, &npcol, &myrow, &mycol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descb), desc_N(descb), b, desc_LLD(descb), desc_MB(descb), nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    Cblacs_gridinfo(desc_CTXT(descx), &nprow, &npcol, &myrow, &mycol);
    auto X = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(descx), desc_N(descx), x, desc_LLD(descx), desc_MB(descx), nprow, npcol, MPI_COMM_WORLD);
    X = slate_scalapack_submatrix(Xm, Xn, X, ix, jx, descx);

    if (verbose && myrow == 0 && mycol == 0)
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
        int64_t p_count = 0;
        int nb = desc_MB(desca);

        // NOTE: this is not the most efficient way, instead use local tile index directly to avoid looping over tiles
        // const int64_t A_nt = A.nt();
        // const int64_t A_mt = A.mt();
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
