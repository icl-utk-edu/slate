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
#include <complex>


#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads(int nt);
inline int slate_set_num_blas_threads(const int nt) { return MKL_Set_Num_Threads(nt); }
#else
inline int slate_set_num_blas_threads(const int nt) { return -1; }
#endif

namespace slate {
namespace scalapack_compat {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);

// Declarations
template< typename scalar_t >
void slate_psyr2k(const char* uplostr, const char* transstr, int n, int k, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb, scalar_t beta, scalar_t* c, int ic, int jc, int* descc);

enum slate_scalapack_desc { DTYPE_=0, CTXT_=1, M_=2, N_=3, IMB_=4, INB_=5, MB_=6, NB_=7, RSRC_=8, CSRC_=9, LLD_=10 };

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_psyr2k

extern "C" void PDSYR2K(const char* uplo, const char* trans, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pdsyr2k(const char* uplo, const char* trans, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pdsyr2k_(const char* uplo, const char* trans, int* n, int* k, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb, double* beta, double* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PSSYR2K(const char* uplo, const char* trans, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pssyr2k(const char* uplo, const char* trans, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pssyr2k_(const char* uplo, const char* trans, int* n, int* k, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb, float* beta, float* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PCSYR2K(const char* uplo, const char* trans, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pcsyr2k(const char* uplo, const char* trans, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pcsyr2k_(const char* uplo, const char* trans, int* n, int* k, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb, std::complex<float>* beta, std::complex<float>* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

extern "C" void PZSYR2K(const char* uplo, const char* trans, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pzsyr2k(const char* uplo, const char* trans, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

extern "C" void pzsyr2k_(const char* uplo, const char* trans, int* n, int* k, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb, std::complex<double>* beta, std::complex<double>* c, int* ic, int* jc, int* descc)
{
    slate_psyr2k(uplo, trans, *n, *k, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb, *beta, c, *ic, *jc, descc);
}

// -----------------------------------------------------------------------------

template< typename scalar_t >
static slate::Matrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::Matrix<scalar_t>& A, int ia, int ja, int* desca)
{
    assert((ia-1)%desca[MB_]==0);
    assert((ja-1)%desca[NB_]==0);
    int64_t i1 = (ia-1)/desca[MB_];
    int64_t i2 = (ia-1+Am)/desca[MB_]-1;
    int64_t j1 = (ja-1)/desca[NB_];
    int64_t j2 = (ja-1+An)/desca[NB_]-1;
    return A.sub(i1, i2, j1, j2);
}

template< typename scalar_t >
static slate::SymmetricMatrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::SymmetricMatrix<scalar_t>& A, int ia, int ja, int* desca)
{
    assert((ia-1)%desca[MB_]==0);
    assert((ja-1)%desca[NB_]==0);
    int64_t i1 = (ia-1)/desca[MB_];
    int64_t i2 = (ia-1+Am)/desca[MB_]-1;
    // int64_t j1 = (ja-1)/desca[NB_];
    // int64_t j2 = (ja-1+An)/desca[NB_]-1;
    return A.sub(i1, i2);
}

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_psyr2k(const char* uplostr, const char* transstr, int n, int k, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb, scalar_t beta, scalar_t* c, int ic, int jc, int* descc)
{
    // todo: figure out if the pxq grid is in row or column
    printf("syr2k");

    int saved_num_blas_threads = slate_set_num_blas_threads(1);
    int saved_num_omp_threads;
    // todo: does this set the omp num threads correctly
    #pragma omp parallel
    { saved_num_omp_threads = omp_get_num_threads(); }
    omp_set_num_threads(std::max({saved_num_blas_threads, saved_num_omp_threads}));

    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op trans = blas::char2op(transstr[0]);
    slate::Target target = slate::Target::Devices;

    // setup so op(A) and op(B) are n-by-k
    int64_t Am = (trans == blas::Op::NoTrans ? n : k);
    int64_t An = (trans == blas::Op::NoTrans ? k : n);
    int64_t Bm = Am;
    int64_t Bn = An;
    // int64_t Cm = n;
    int64_t Cn = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(desca[CTXT_], &nprow, &npcol, &myrow, &mycol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desca[M_], desca[N_], a, desca[LLD_], desca[MB_], nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(descb[CTXT_], &nprow, &npcol, &myrow, &mycol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(descb[M_], descb[N_], b, descb[LLD_], descb[MB_], nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    Cblacs_gridinfo(descc[CTXT_], &nprow, &npcol, &myrow, &mycol);
    // todo fix this allocation of C so it respects ic jc
    // auto C = slate::Matrix<scalar_t>::fromScaLAPACK(descc[M_], descc[N_], c, descc[LLD_], descc[MB_], nprow, npcol, MPI_COMM_WORLD);
    // auto Csub = slate_scalapack_submatrix(Cm, Cn, C, ic, jc, descc);
    // auto CS = slate::SymmetricMatrix<scalar_t>(uplo, Csub);
    auto CS = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(uplo, Cn, c, descc[LLD_], descc[MB_], nprow, npcol, MPI_COMM_WORLD);

    if (trans == blas::Op::Trans) {
        A = transpose(A);
        B = transpose(B);
    }
    else if (trans == blas::Op::ConjTrans) {
        A = conj_transpose(A);
        B = conj_transpose(B);
    }
    assert(A.mt() == CS.mt());
    assert(B.mt() == CS.mt());
    assert(A.nt() == B.nt());

    int64_t lookahead = 1;
    slate::syr2k(alpha, A, B, beta, CS, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    omp_set_num_threads(saved_num_omp_threads);
    slate_set_num_blas_threads(saved_num_blas_threads);
}

} // namespace scalapack_compat
} // namespace slate
