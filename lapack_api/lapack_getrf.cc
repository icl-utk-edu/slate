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

#include "slate.hh"
#include "lapack_slate.hh"
#include "blas_fortran.hh"
#include <complex>

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------

// Local function
template< typename scalar_t >
void slate_getrf(const int m, const int n, scalar_t* a, const int lda, int* ipiv, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgetrf BLAS_FORTRAN_NAME( slate_sgetrf, SLATE_SGETRF )
#define slate_dgetrf BLAS_FORTRAN_NAME( slate_dgetrf, SLATE_DGETRF )
#define slate_cgetrf BLAS_FORTRAN_NAME( slate_cgetrf, SLATE_CGETRF )
#define slate_zgetrf BLAS_FORTRAN_NAME( slate_zgetrf, SLATE_ZGETRF )

extern "C" void slate_sgetrf(const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info)
{
    return slate_getrf(*m, *n, a, *lda, ipiv, info);
}

extern "C" void slate_dgetrf(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info)
{
    return slate_getrf(*m, *n, a, *lda, ipiv, info);
}

extern "C" void slate_cgetrf(const int* m, const int* n, std::complex<float>* a, const int* lda, int* ipiv, int* info)
{
    return slate_getrf(*m, *n, a, *lda, ipiv, info);
}

extern "C" void slate_zgetrf(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv, int* info)
{
    return slate_getrf(*m, *n, a, *lda, ipiv, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_getrf(const int m, const int n, scalar_t* a, const int lda, int* ipiv, int* info)
{
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized) MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    // todo: does this set the omp num threads correctly in all circumstances
    int saved_num_blas_threads = slate_lapack_set_num_blas_threads(1);

    int64_t lookahead = 1;
    int64_t panel_threads = slate_lapack_set_panelthreads();
    int64_t ib = slate_lapack_set_ib();
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int verbose = slate_lapack_set_verbose();
    static int64_t nb = slate_lapack_set_nb(target);

    // sizes of data
    int64_t Am = m;
    int64_t An = n;
    slate::Pivots pivots;

    // create SLATE matrices from the Lapack layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);

    if (verbose) logprintf("%s\n", "getrf");
    slate::getrf(A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    // Extract pivots from SLATE's Pivots structure into LAPACK ipiv array
    {
        int64_t p_count = 0;
        for (auto t_iter = pivots.begin(); t_iter != pivots.end(); ++t_iter) {
            for (auto p_iter = t_iter->begin(); p_iter != t_iter->end(); ++p_iter) {
                ipiv[p_count++] = p_iter->tileIndex() * nb + p_iter->elementOffset();
            }
        }
    }

    slate_lapack_set_num_blas_threads(saved_num_blas_threads);

    // todo get a real value for info
    *info = 0;
}

} // namespace lapack_api
} // namespace slate
