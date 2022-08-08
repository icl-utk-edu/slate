//------------------------------------------------------------------------------
// Copyright (c) 2017-2022, University of Tennessee
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

#include "lapack_slate.hh"

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------
// Local function
template <typename scalar_t>
void slate_getri(const int n, scalar_t* a, const int lda, int* ipiv, scalar_t* work, const int lwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgetri BLAS_FORTRAN_NAME( slate_sgetri, SLATE_SGETRI )
#define slate_dgetri BLAS_FORTRAN_NAME( slate_dgetri, SLATE_DGETRI )
#define slate_cgetri BLAS_FORTRAN_NAME( slate_cgetri, SLATE_CGETRI )
#define slate_zgetri BLAS_FORTRAN_NAME( slate_zgetri, SLATE_ZGETRI )

extern "C" void slate_sgetri(const int* n, float* a, const int* lda, int* ipiv, float* work, int* lwork, int* info)
{
    slate_getri(*n, a, *lda, ipiv, work, *lwork, info);
}

extern "C" void slate_dgetri(const int* n, double* a, const int* lda, int* ipiv, double* work, int* lwork, int* info)
{
    slate_getri(*n, a, *lda, ipiv, work, *lwork, info);
}

extern "C" void slate_cgetri(const int* n, std::complex<float>* a, const int* lda, int* ipiv, std::complex<float>* work, int* lwork, int* info)
{
    slate_getri(*n, a, *lda, ipiv, work, *lwork, info);
}

extern "C" void slate_zgetri(const int* n, std::complex<double>* a, const int* lda, int* ipiv, std::complex<double>* work, int* lwork, int* info)
{
    slate_getri(*n, a, *lda, ipiv, work, *lwork, info);
}

// -----------------------------------------------------------------------------
// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_getri(const int n, scalar_t* a, const int lda, int* ipiv, scalar_t* work, const int lwork, int* info)
{
    using real_t = blas::real_type<scalar_t>;

    // Respond to workspace query with a minimal value (1); workspace
    // is allocated within the SLATE routine.
    if (lwork == -1) {
        work[0] = (real_t)1.0;
        *info = 0;
        return;
    }

    // Start timing
    int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // Check and initialize MPI, else SLATE calls to MPI will fail
    // Since this is an lapack wrapper there will be only one MPI process
    int initialized=0, provided=0;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();

    // sizes
    static int64_t nb = slate_lapack_set_nb(target);

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(n, n, a, lda, nb, p, q, MPI_COMM_WORLD);

    // extract pivots from LAPACK ipiv to SLATES pivot structure
    slate::Pivots pivots; // std::vector< std::vector<Pivot> >
    {
        // allocate pivots
        const int64_t min_mt_nt = std::min(A.mt(), A.nt());
        pivots.resize(min_mt_nt);
        for (int64_t k = 0; k < min_mt_nt; ++k) {
            const int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);
        }
        // transfer ipiv to pivots
        int64_t p_count = 0;
        int64_t t_iter_add = 0;
        for (auto t_iter = pivots.begin(); t_iter != pivots.end(); ++t_iter) {
            for (auto p_iter = t_iter->begin(); p_iter != t_iter->end(); ++p_iter) {
                int64_t tileIndex = (ipiv[p_count] - 1 - t_iter_add) / nb;
                int64_t elementOffset = (ipiv[p_count] - 1 - t_iter_add) % nb;
                *p_iter = Pivot(tileIndex, elementOffset);
                p_count++;
            }
            t_iter_add += nb;
        }
    }

    // inverse
    slate::getri(A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get a real value for info
    *info = 0;

    if (verbose)
        std::cout << "slate_lapack_api: "
                  << slate_lapack_scalar_t_to_char(a) << "getri("
                  <<  n << "," <<  (void*)a << "," <<  lda << "," << (void*)ipiv << ","
                  << (void*)work << "," << lwork << "," << *info << ") "
                  << (omp_get_wtime()-timestart) << " sec "
                  << "nb:" << nb
                  << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
