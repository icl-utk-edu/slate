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
#include "slate_cuda.hh"
#include "blas_fortran.hh"
#include <complex>

#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads(int nt);
inline int slate_set_num_blas_threads(const int nt) { return MKL_Set_Num_Threads(nt); }
#else
inline int slate_set_num_blas_threads(const int nt) { return 1; }
#endif

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------

// Local function
template< typename scalar_t >
void slate_trmm(const char* sidestr, const char* uplostr, const char* transastr, const char* diagstr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_strmm BLAS_FORTRAN_NAME( slate_strmm, SLATE_STRMM )
#define slate_dtrmm BLAS_FORTRAN_NAME( slate_dtrmm, SLATE_DTRMM )
#define slate_ctrmm BLAS_FORTRAN_NAME( slate_ctrmm, SLATE_CTRMM )
#define slate_ztrmm BLAS_FORTRAN_NAME( slate_ztrmm, SLATE_ZTRMM )

extern "C" void slate_strmm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const float* alpha, float* a, const int* lda, float* b, const int* ldb)
{
    slate_trmm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

extern "C" void slate_dtrmm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const double* alpha, double* a, const int* lda, double* b, const int* ldb)
{
    slate_trmm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

extern "C" void slate_ctrmm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const std::complex<float>* alpha, std::complex<float>* a, const int* lda, std::complex<float>* b, const int* ldb)
{
    slate_trmm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

extern "C" void slate_ztrmm(const char* side, const char* uplo, const char* transa, const char* diag, const int* m, const int* n, const std::complex<double>* alpha, std::complex<double>* a, const int* lda, std::complex<double>* b, const int* ldb)
{
    slate_trmm(side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_trmm(const char* sidestr, const char* uplostr, const char* transastr, const char* diagstr, const int m, const int n, const scalar_t alpha, scalar_t* a, const int lda, scalar_t* b, const int ldb)
{
    // Check and initialize MPI, else SLATE calls to MPI will fail
    int initialized, provided;
    assert(MPI_Initialized(&initialized) == MPI_SUCCESS);
    if (! initialized) assert(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS);

    // todo: does this set the omp num threads correctly in all circumstances
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    blas::Side side = blas::char2side(sidestr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transastr[0]);
    blas::Diag diag = blas::char2diag(diagstr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target;
    static int64_t target_is_set = 0;
    static int64_t nb = 0;

    // set target if not already done
    if (! target_is_set) {
        int cudadevcount = 0;
        target = slate::Target::HostTask;
        if (std::getenv("SLATE_TARGET")) {
            char targetchar = (char)(toupper(std::getenv("SLATE_TARGET")[4]));
            if (targetchar=='T') target = slate::Target::HostTask;
            else if (targetchar=='N') target = slate::Target::HostNest;
            else if (targetchar=='B') target = slate::Target::HostBatch;
            else if (targetchar=='C') target = slate::Target::Devices;
        }
        else if (cudaGetDeviceCount(&cudadevcount)==cudaSuccess && cudadevcount>0)
            target = slate::Target::Devices;
        target_is_set = 1;
    }

    // set nb if not already done
    if (nb == 0) {
        nb = 256;
        if (std::getenv("SLATE_NB"))
            nb = strtol(std::getenv("SLATE_NB"), NULL, 0);
        else if (target==slate::Target::HostTask)
            nb = 512;
        else if (target==slate::Target::Devices)
            nb = 1024;
    }

    // setup so op(B) is m-by-n
    int64_t An  = (side == blas::Side::Left ? m : n);
    int64_t Bm  = m;
    int64_t Bn  = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::TriangularMatrix<scalar_t>::fromLAPACK(uplo, diag, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, b, ldb, nb, p, q, MPI_COMM_WORLD);

    if (transA == Op::Trans)
        A = transpose(A);
    else if (transA == Op::ConjTrans)
        A = conj_transpose(A);

    // typedef long long lld;
    // printf("SLATE TRMM m %lld n %lld nb %lld \n", (lld)m, (lld)n, (lld)nb);
    slate::trmm(side, alpha, A, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    slate_set_num_blas_threads(saved_num_blas_threads);
}

} // namespace lapack_api
} // namespace slate
