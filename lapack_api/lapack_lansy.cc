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
#include "lapack_fortran.h"
#include <complex>

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------

// Local function
template< typename scalar_t >
blas::real_type<scalar_t> slate_lansy(const char* normstr, const char* uplostr, int n, scalar_t* a, int lda, blas::real_type<scalar_t>* work);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_slansy LAPACK_GLOBAL( slate_slansy, SLATE_SLANSY )
#define slate_dlansy LAPACK_GLOBAL( slate_dlansy, SLATE_DLANSY )
#define slate_clansy LAPACK_GLOBAL( slate_clansy, SLATE_CLANSY )
#define slate_zlansy LAPACK_GLOBAL( slate_zlansy, SLATE_ZLANSY )

extern "C" float slate_slansy(const char* norm, const char* uplo, int* n, float* a, int* lda, float* work)
{
    return slate_lansy(norm, uplo, *n, a, *lda, work);
}

extern "C" double slate_dlansy(const char* norm, const char* uplo, int* n, double* a, int* lda, double* work)
{
    return slate_lansy(norm, uplo, *n, a, *lda, work);
}

extern "C" float slate_clansy(const char* norm, const char* uplo, int* n, std::complex<float>* a, int* lda, float* work)
{
    return slate_lansy(norm, uplo, *n, a, *lda, work);
}

extern "C" double slate_zlansy(const char* norm, const char* uplo, int* n, std::complex<double>* a, int* lda, double* work)
{
    return slate_lansy(norm, uplo, *n, a, *lda, work);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
blas::real_type<scalar_t> slate_lansy(const char* normstr, const char* uplostr, int n, scalar_t* a, int lda, blas::real_type<scalar_t>* work)
{
    // start timing
    // static int verbose = slate_lapack_set_verbose();
    // double timestart = 0.0;
    // if (verbose) timestart = omp_get_wtime();

    // need a dummy MPI_Init for SLATE to proceed
    int initialized, provided;
    MPI_Initialized(&initialized);
    if (! initialized) MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);

    // todo: does this set the omp num threads correctly in all circumstances
    int saved_num_blas_threads = slate_lapack_set_num_blas_threads(1);

    lapack::Norm norm = lapack::char2norm(normstr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target = slate_lapack_set_target();
    static int64_t nb = slate_lapack_set_nb(target);

    // sizes of matrices
    int64_t An = n;

    // create SLATE matrix from the Lapack layouts
    auto A = slate::SymmetricMatrix<scalar_t>::fromLAPACK(uplo, An, a, lda, nb, p, q, MPI_COMM_WORLD);

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm(norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    slate_lapack_set_num_blas_threads(saved_num_blas_threads);

    // if (verbose) std::cout << "slate_lapack_api: " << slate_lapack_scalar_t_to_char(a) << "lansy(" << normstr[0] << "," << uplostr[0] <<  "," << n << "," <<  (void*)a << "," <<  lda << "," <<  (void*)work << ") " <<  (omp_get_wtime()-timestart) << " sec " << "nb:" << nb << " max_threads:" << omp_get_max_threads() << "\n";

    return A_norm;
}

} // namespace lapack_api
} // namespace slate
