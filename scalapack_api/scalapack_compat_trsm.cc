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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
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
template<typename scalar_t>
void slate_ptrsm(const char* sidestr, const char* uplostr, const char* transastr, const char* diagstr, int m, int n, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb);

enum slate_scalapack_desc { DTYPE_=0, CTXT_=1, M_=2, N_=3, IMB_=4, INB_=5, MB_=6, NB_=7, RSRC_=8, CSRC_=9, LLD_=10 };

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_ptrsm

extern "C" void PDTRSM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pdtrsm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pdtrsm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, double* alpha, double* a, int* ia, int* ja, int* desca, double* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

// -----------------------------------------------------------------------------

extern "C" void PSTRSM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pstrsm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pstrsm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, float* alpha, float* a, int* ia, int* ja, int* desca, float* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

// -----------------------------------------------------------------------------

extern "C" void PCTRSM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pctrsm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pctrsm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<float>* alpha, std::complex<float>* a, int* ia, int* ja, int* desca, std::complex<float>* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

// -----------------------------------------------------------------------------

extern "C" void PZTRSM(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pztrsm(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
}

extern "C" void pztrsm_(const char* side, const char* uplo, const char* transa, const char* diag, int* m, int* n, std::complex<double>* alpha, std::complex<double>* a, int* ia, int* ja, int* desca, std::complex<double>* b, int* ib, int* jb, int* descb)
{
    slate_ptrsm(side, uplo, transa, diag, *m, *n, *alpha, a, *ia, *ja, desca, b, *ib, *jb, descb);
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
static slate::TriangularMatrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::TriangularMatrix<scalar_t>& A, int ia, int ja, int* desca)
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
void slate_ptrsm(const char* sidestr, const char* uplostr, const char* transastr, const char* diagstr, int m, int n, scalar_t alpha, scalar_t* a, int ia, int ja, int* desca, scalar_t* b, int ib, int jb, int* descb)
{
    // todo: figure out if the pxq grid is in row or column
    printf("trsm");;

    int saved_num_blas_threads = slate_set_num_blas_threads(1);
    int saved_num_omp_threads;
    // todo: does this set the omp num threads correctly
    #pragma omp parallel
    { saved_num_omp_threads = omp_get_num_threads(); }
    omp_set_num_threads(std::max({saved_num_blas_threads, saved_num_omp_threads}));

    blas::Side side = blas::char2side(sidestr[0]);
    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transastr[0]);
    blas::Diag diag = blas::char2diag(diagstr[0]);
    int64_t lookahead = 1;
    slate::Target target = slate::Target::Devices;

    // setup so trans(B) is m-by-n
    int64_t An  = (side == blas::Side::Left ? m : n);
    int64_t Am  = An;
    int64_t Bm  = m;
    int64_t Bn  = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    // todo fix A's allocation
    Cblacs_gridinfo(desca[CTXT_], &nprow, &npcol, &myrow, &mycol);
    auto A = slate::TriangularMatrix<scalar_t>::fromScaLAPACK(uplo, An, a, desca[LLD_], desca[MB_], nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    Cblacs_gridinfo(descb[CTXT_], &nprow, &npcol, &myrow, &mycol);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(descb[M_], descb[N_], b, descb[LLD_], descb[MB_], nprow, npcol, MPI_COMM_WORLD);
    B = slate_scalapack_submatrix(Bm, Bn, B, ib, jb, descb);

    if (transA == Op::Trans)
        A = transpose(A);
    else if (transA == Op::ConjTrans)
        A = conj_transpose(A);

    slate::trsm(side, diag, alpha, A, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    omp_set_num_threads(saved_num_omp_threads);
    slate_set_num_blas_threads(saved_num_blas_threads);
}

} // namespace scalapack_compat
} // namespace slate
