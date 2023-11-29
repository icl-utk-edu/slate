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
void slate_heevd(const char* jobzstr, const char* uplostr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t>* w, scalar_t* work, const int lwork, blas::real_type<scalar_t>* rwork, int lrwork, int* iwork, int liwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_ssyevd BLAS_FORTRAN_NAME( slate_ssyevd, SLATE_SSYEVD )
#define slate_dsyevd BLAS_FORTRAN_NAME( slate_dsyevd, SLATE_DSYEVD )
#define slate_cheevd BLAS_FORTRAN_NAME( slate_cheevd, SLATE_CHEEVD )
#define slate_zheevd BLAS_FORTRAN_NAME( slate_zheevd, SLATE_ZHEEVD )

extern "C" void slate_ssyevd(const char* jobzstr, const char* uplostr, const int* n, float* a, const int* lda, float* w, float* work, const int* lwork, int* iwork, const int* liwork, int* info)
{
    float dummy;
    slate_heevd(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, &dummy, 1, iwork, *liwork, info);
}
extern "C" void slate_dsyevd(const char* jobzstr, const char* uplostr, const int* n, double* a, const int* lda, double* w, double* work, const int* lwork, int* iwork, const int* liwork, int* info)
{
    double dummy;
    slate_heevd(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, &dummy, 1, iwork, *liwork, info);
}
extern "C" void slate_cheevd(const char* jobzstr, const char* uplostr, const int* n, std::complex<float>* a, const int* lda, float* w, std::complex<float>* work, const int* lwork, float* rwork, const int* lrwork, int* iwork, const int* liwork, int* info)
{
    slate_heevd(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}
extern "C" void slate_zheevd(const char* jobzstr, const char* uplostr, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, const int* lrwork, int* iwork, const int* liwork, int* info)
{
    slate_heevd(jobzstr, uplostr, *n, a, *lda, w, work, *lwork, rwork, *lrwork, iwork, *liwork, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_heevd(const char* jobzstr, const char* uplostr, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t>* w, scalar_t* work, const int lwork, blas::real_type<scalar_t>* rwork, int lrwork, int* iwork, int liwork, int* info)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // sizes
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);

    // TODO check args more carefully
    *info = 0;

    if (lwork == -1 || lrwork == -1 || liwork == -1) {
        work[0] = n * n;
        rwork[0] = 0;
        iwork[0] = 0;
    }
    else {
        // Check and initialize MPI, else SLATE calls to MPI will fail
        int initialized, provided;
        MPI_Initialized(&initialized);
        if (! initialized)
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

        int64_t lookahead = 1;
        int64_t p = 1;
        int64_t q = 1;
        blas::Uplo uplo = blas::char2uplo(uplostr[0]);

        // create SLATE matrix from the LAPACK data
        auto A = slate::Matrix<scalar_t>::fromLAPACK( n, n, a, lda, nb, p, q, MPI_COMM_WORLD );
        slate::HermitianMatrix<scalar_t> AH( uplo, A );
        std::vector< blas::real_type<scalar_t> > Lambda_( n );

        slate::Matrix<scalar_t> Z;
        switch (jobzstr[0]) {
        case 'V':
            if (lwork >= n * n) {
                Z = slate::Matrix<scalar_t>::fromLAPACK( n, n, work, n, nb, p, q, MPI_COMM_WORLD );
            }
            else {
                Z = slate::Matrix<scalar_t>( n, n, nb, p, q, MPI_COMM_WORLD );
                Z.insertLocalTiles(target);
            }
            break;
        case 'N':
            // Leave Z empty
            break;
        default:
            *info = 1;
        }

        if (*info == 0) {
            // solve
            slate::heev( AH, Lambda_, Z, {
                {slate::Option::MethodEig, MethodEig::DC},
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            std::copy(Lambda_.begin(), Lambda_.end(), w);

            if (jobzstr[0] == 'V') {
                slate::copy( Z, A, {
                    {slate::Option::Target, target}
                });
            }
        }
    }

    if (verbose) {
        const char* routine_name = is_complex<scalar_t>::value ? "heevd" : "syevd";
        std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << routine_name << "(" <<  jobzstr[0] << "," << uplostr[0] << "," << n << "," << (void*)a << "," <<  lda << "," << (void*)w << (void*)work << "," << lwork << ",";
        if constexpr (! is_complex<scalar_t>::value) {
            std::cout << (void*)rwork << "," << lrwork << ",";
        }
        std::cout << (void*)iwork << "," << liwork << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
    }
}

} // namespace lapack_api
} // namespace slate
