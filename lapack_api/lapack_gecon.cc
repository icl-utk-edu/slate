//------------------------------------------------------------------------------
// Copyright (c) 2017-2023, University of Tennessee
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
void slate_gecon(const char* normstr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t> Anorm, blas::real_type<scalar_t>* rcond, scalar_t* work, int* iwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgecon BLAS_FORTRAN_NAME( slate_sgecon, SLATE_SGECON )
#define slate_dgecon BLAS_FORTRAN_NAME( slate_dgecon, SLATE_DGECON )
#define slate_cgecon BLAS_FORTRAN_NAME( slate_cgecon, SLATE_CGECON )
#define slate_zgecon BLAS_FORTRAN_NAME( slate_zgecon, SLATE_ZGECON )

extern "C" void slate_sgecon(const char* normstr, const int* n, float* a, const int* lda, float* Anorm, float* rcond, float* work, int* iwork, int* info)
{
    slate_gecon(normstr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}
extern "C" void slate_dgecon(const char* normstr, const int* n, double* a, const int* lda, double* Anorm, double* rcond, double* work, int* iwork, int* info)
{
    slate_gecon(normstr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}
extern "C" void slate_cgecon(const char* normstr, const int* n, std::complex<float>* a, const int* lda, float* Anorm, float* rcond, std::complex<float>* work, int* iwork, int* info)
{
    slate_gecon(normstr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}
extern "C" void slate_zgecon(const char* normstr, const int* n, std::complex<double>* a, const int* lda, double* Anorm, double* rcond, std::complex<double>* work, int* iwork, int* info)
{
    slate_gecon(normstr, *n, a, *lda, *Anorm, rcond, work, iwork, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_gecon(const char* normstr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t> Anorm, blas::real_type<scalar_t>* rcond, scalar_t* work, int* iwork, int* info)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // Check and initialize MPI, else SLATE calls to MPI will fail
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();

    // sizes
    lapack::Norm norm = lapack::char2norm(normstr[0]);
    static int64_t nb = slate_lapack_set_nb(target);

    // create SLATE matrix from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(n, n, a, lda, nb, p, q, MPI_COMM_WORLD);

    // solve
    *rcond = slate::gecondest( norm, A, Anorm, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    // todo:  get a real value for info
    *info = 0;

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "gecon(" <<  normstr[0] << "," << n << "," << (void*)a << "," <<  lda << "," << Anorm << "," << (void*)rcond << "," << (void*)work << "," << (void*)iwork << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
