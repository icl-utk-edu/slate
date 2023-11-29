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
void slate_gesvd(const char* jobustr, const char* jobvtstr, const int m, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t>* s, scalar_t* u, const int ldu, scalar_t* vt, const int ldvt, scalar_t* work, const int lwork, int* info);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_sgesvd BLAS_FORTRAN_NAME( slate_sgesvd, SLATE_SGESVD )
#define slate_dgesvd BLAS_FORTRAN_NAME( slate_dgesvd, SLATE_DGESVD )
#define slate_cgesvd BLAS_FORTRAN_NAME( slate_cgesvd, SLATE_CGESVD )
#define slate_zgesvd BLAS_FORTRAN_NAME( slate_zgesvd, SLATE_ZGESVD )

extern "C" void slate_sgesvd(const char* jobustr, const char* jobvtstr, const int* m, const int* n, float* a, const int* lda, float* s, float* u, const int* ldu, float* vt, const int* ldvt, float* work, const int* lwork, int* info)
{
    slate_gesvd(jobustr, jobvtstr, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt, work, *lwork, info);
}
extern "C" void slate_dgesvd(const char* jobustr, const char* jobvtstr, const int* m, const int* n, double* a, const int* lda, double* s, double* u, const int* ldu, double* vt, const int* ldvt, double* work, const int* lwork, int* info)
{
    slate_gesvd(jobustr, jobvtstr, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt, work, *lwork, info);
}
extern "C" void slate_cgesvd(const char* jobustr, const char* jobvtstr, const int* m, const int* n, std::complex<float>* a, const int* lda, float* s, std::complex<float>* u, const int* ldu, std::complex<float>* vt, const int* ldvt, std::complex<float>* work, const int* lwork, float* rwork, int* info)
{
    slate_gesvd(jobustr, jobvtstr, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt, work, *lwork, info);
}
extern "C" void slate_zgesvd(const char* jobustr, const char* jobvtstr, const int* m, const int* n, std::complex<double>* a, const int* lda, double* s, std::complex<double>* u, const int* ldu, std::complex<double>* vt, const int* ldvt, std::complex<double>* work, const int* lwork, double* rwork, int* info)
{
    slate_gesvd(jobustr, jobvtstr, *m, *n, a, *lda, s, u, *ldu, vt, *ldvt, work, *lwork, info);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template <typename scalar_t>
void slate_gesvd(const char* jobustr, const char* jobvtstr, const int m, const int n, scalar_t* a, const int lda, blas::real_type<scalar_t>* s, scalar_t* u, const int ldu, scalar_t* vt, const int ldvt, scalar_t* work, const int lwork, int* info)
{
    // Start timing
    static int verbose = slate_lapack_set_verbose();
    double timestart = 0.0;
    if (verbose) timestart = omp_get_wtime();

    // sizes
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);
    int64_t min_mn = std::min( m, n );

    // TODO check args more carefully
    *info = 0;

    if (lwork == -1) {
        if (jobustr[0] == 'O') {
            work[0] = m * min_mn;
        }
        else if (jobvtstr[0] == 'O') {
            work[0] = min_mn * n;
        }
        else {
            work[0] = 0;
        }
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

        // create SLATE matrix from the LAPACK data
        auto A = slate::Matrix<scalar_t>::fromLAPACK(m, n, a, lda, nb, p, q, MPI_COMM_WORLD);
        std::vector< blas::real_type<scalar_t> > Sigma_( min_mn );

        slate::Matrix<scalar_t> U;
        switch (jobustr[0]) {
        case 'A':
            U = slate::Matrix<scalar_t>::fromLAPACK(m, m, u, ldu, nb, p, q, MPI_COMM_WORLD);
            break;
        case 'S':
            U = slate::Matrix<scalar_t>::fromLAPACK(m, min_mn, u, ldu, nb, p, q, MPI_COMM_WORLD);
            break;
        case 'O':
            if (lwork >= m * min_mn) {
                U = slate::Matrix<scalar_t>::fromLAPACK(m, min_mn, work, m, nb, p, q, MPI_COMM_WORLD);
            }
            else {
                U = slate::Matrix<scalar_t>(m, min_mn, nb, p, q, MPI_COMM_WORLD);
                U.insertLocalTiles(target);
            }
            break;
        case 'N':
            // Leave U empty
            break;
        default:
            *info = 1;
        }

        slate::Matrix<scalar_t> VT;
        switch (jobvtstr[0]) {
        case 'A':
            VT = slate::Matrix<scalar_t>::fromLAPACK(n, n, vt, ldvt, nb, p, q, MPI_COMM_WORLD);
            break;
        case 'S':
            VT = slate::Matrix<scalar_t>::fromLAPACK(min_mn, n, vt, ldvt, nb, p, q, MPI_COMM_WORLD);
            break;
        case 'O':
            if (lwork >= min_mn * n) {
                VT = slate::Matrix<scalar_t>::fromLAPACK(min_mn, n, work, m, nb, p, q, MPI_COMM_WORLD);
            }
            else {
                VT = slate::Matrix<scalar_t>(min_mn, n, nb, p, q, MPI_COMM_WORLD);
                VT.insertLocalTiles(target);
            }
            break;
        case 'N':
            // Leave VT empty
            break;
        default:
            *info = 2;
        }

        if (*info == 0) {
            // solve
            slate::svd( A, Sigma_, U, VT, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            std::copy(Sigma_.begin(), Sigma_.end(), s);

            if (jobustr[0] == 'O') {
                auto A_slice = A.slice( 0, m-1, 0, min_mn-1 );
                slate::copy( U, A_slice, {
                    {slate::Option::Target, target}
                });
            }
            if (jobvtstr[0] == 'O') {
                auto A_slice = A.slice( 0, n-1, 0, min_mn-1 );
                slate::copy( VT, A_slice, {
                    {slate::Option::Target, target}
                });
            }
        }
    }

    if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "gesvd(" <<  jobustr[0] << "," << jobvtstr[0] << "," << m << "," << n << "," << (void*)a << "," <<  lda << "," << (void*)s << "," << (void*)u << "," << ldu << "," << (void*)vt << "," << ldvt << "," << (void*)work << "," << lwork << "," << *info << ") " << (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";
}

} // namespace lapack_api
} // namespace slate
