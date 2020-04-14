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
void slate_pgels(const char* transstr, int m, int n, int nrhs, scalar_t* a, int lda, scalar_t* b, int ldb, scalar_t* work, int lwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgels BLAS_FORTRAN_NAME( slate_sgels, SLATE_SGELS )
#define slate_dgels BLAS_FORTRAN_NAME( slate_dgels, SLATE_DGELS )
#define slate_cgels BLAS_FORTRAN_NAME( slate_cgels, SLATE_CGELS )
#define slate_zgels BLAS_FORTRAN_NAME( slate_zgels, SLATE_ZGELS )

extern "C" void slate_sgels(const char* trans, int* m, int* n, int* nrhs, float* a, int* lda, float* b, int* ldb, float* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

extern "C" void slate_dgels(const char* trans, int* m, int* n, int* nrhs, double* a, int* lda, double* b, int* ldb, double* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

extern "C" void slate_cgels(const char* trans, int* m, int* n, int* nrhs, std::complex<float>* a, int* lda, std::complex<float>* b, int* ldb, std::complex<float>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

extern "C" void slate_zgels(const char* trans, int* m, int* n, int* nrhs, std::complex<double>* a, int* lda, std::complex<double>* b, int* ldb, std::complex<double>* work, int* lwork, int* info)
{
    slate_pgels(trans, *m, *n, *nrhs, a, *lda, b, *ldb, work, *lwork, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_pgels(const char* transstr, int m, int n, int nrhs, scalar_t* a, int lda, scalar_t* b, int ldb, scalar_t* work, int lwork, int* info)
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
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // Need a dummy MPI_Init for SLATE to proceed
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized) MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);

    // todo: does this set the omp num threads correctly in all circumstances
    int saved_num_blas_threads = slate_lapack_set_num_blas_threads(1);

    int64_t p = 1;
    int64_t q = 1;
    int64_t lookahead = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);
    static int64_t panel_threads = slate_lapack_set_panelthreads();
    static int64_t inner_blocking = slate_lapack_set_ib();

    // sizes
    blas::Op trans = blas::char2op(transstr[0]);
    // A is m-by-n, BX is max(m, n)-by-nrhs.
    // If op == NoTrans, op(A) is m-by-n, B is m-by-nrhs
    // otherwise,        op(A) is n-by-m, B is n-by-nrhs.
    int64_t Am = (trans == slate::Op::NoTrans ? m : n);
    int64_t An = (trans == slate::Op::NoTrans ? n : m);
    int64_t Bm = (trans == slate::Op::NoTrans ? m : n);
    int64_t Bn = nrhs;

    // create SLATE matrices from the LAPACK layouts
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);

    // Apply transpose
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        opA = conjTranspose(A);

    slate::TriangularFactors<scalar_t> T;

    slate::gels(opA, T, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, inner_blocking}
    });

    slate_lapack_set_num_blas_threads(saved_num_blas_threads);

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "gels(" << transstr[0] << "," <<  m << "," <<  n << "," << nrhs << "," <<  (void*)a << "," <<  lda << "," << (void*)b << "," << ldb << "," << (void*)work << "," << lwork << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

    // todo: extract the real info
    *info = 0;
}

} // namespace lapack_api
} // namespace slate
