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
#include "lapack_slate.hh"
#include "blas_fortran.hh"
#include "lapack_fortran.h"
#include <complex>

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------

// Local function
template< typename scalar_t >
void slate_getrs(const char* transstr, const int n, const int nrhs, scalar_t* a, const int lda, int* ipiv, scalar_t* b, const int ldb, int* info);

using llong = long long;

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgetrs BLAS_FORTRAN_NAME( slate_sgetrs, SLATE_SGETRS )
#define slate_dgetrs BLAS_FORTRAN_NAME( slate_dgetrs, SLATE_DGETRS )
#define slate_cgetrs BLAS_FORTRAN_NAME( slate_cgetrs, SLATE_CGETRS )
#define slate_zgetrs BLAS_FORTRAN_NAME( slate_zgetrs, SLATE_ZGETRS )

extern "C" void slate_sgetrs(const char* trans, const int* n, const int* nrhs, float* a, const int* lda, int* ipiv, float* b, const int* ldb, int* info)
{
    slate_getrs(trans, *n, *nrhs, a, *lda, ipiv, b, *ldb, info);
}

extern "C" void slate_dgetrs(const char* trans, const int* n, const int* nrhs, double* a, const int* lda, int* ipiv, double* b, const int* ldb, int* info)
{
    slate_getrs(trans, *n, *nrhs, a, *lda, ipiv, b, *ldb, info);
}

extern "C" void slate_cgetrs(const char* trans, const int* n, const int* nrhs, std::complex<float>* a, const int* lda, int* ipiv, std::complex<float>* b, const int* ldb, int* info)
{
    slate_getrs(trans, *n, *nrhs, a, *lda, ipiv, b, *ldb, info);
}

extern "C" void slate_zgetrs(const char* trans, const int* n, const int* nrhs, std::complex<double>* a, const int* lda, int* ipiv, std::complex<double>* b, const int* ldb, int* info)
{
    slate_getrs(trans, *n, *nrhs, a, *lda, ipiv, b, *ldb, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_getrs(const char* transstr, const int n, const int nrhs, scalar_t* a, const int lda, int* ipiv, scalar_t* b, const int ldb, int* info)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // Check and initialize MPI, else SLATE calls to MPI will fail
    int initialized, provided;
    assert(MPI_Initialized(&initialized) == MPI_SUCCESS);
    if (! initialized) assert(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS);

    // todo: does this set the omp num threads correctly in all circumstances
    int saved_num_blas_threads = slate_lapack_set_num_blas_threads(1);

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();

    // sizes
    blas::Op trans = blas::char2op(transstr[0]);
    int64_t Am = n, An = n;
    int64_t Bm = n, Bn = nrhs;
    static int64_t nb = slate_lapack_set_nb(target);

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);

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

    // apply operator to A
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        opA = conjTranspose(A);

    // solve
    slate::getrs(opA, pivots, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    slate_lapack_set_num_blas_threads(saved_num_blas_threads);

    // todo:  get a real value for info
    *info = 0;

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "getrs(" <<  transstr[0] << "," << n << "," <<  nrhs << "," << (void*)a << "," <<  lda << "," << (void*)ipiv << "," << (void*)b << "," << ldb << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
