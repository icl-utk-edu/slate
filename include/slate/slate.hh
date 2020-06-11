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

#ifndef SLATE_HH
#define SLATE_HH

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/TriangularMatrix.hh"

#include "slate/BandMatrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "slate/HermitianBandMatrix.hh"

#include "slate/types.hh"

//------------------------------------------------------------------------------
/// @namespace slate
/// SLATE's top-level namespace.
///
namespace slate {

// Version is updated by make_release.py; DO NOT EDIT.
// Version 0000.00.00
#define SLATE_VERSION 00000000

int version();
const char* id();

//------------------------------------------------------------------------------
// Auxiliary

//-----------------------------------------
// copy()
template <typename src_matrix_type, typename dst_matrix_type>
void copy(
    src_matrix_type& A,
    dst_matrix_type& B,
    Options const& opts = Options());

//-----------------------------------------
// set()
template <typename scalar_t>
void set(
    scalar_t offdiag_value,
    scalar_t diag_value,
    Matrix<scalar_t>& A,
    Options const& opts = Options());

template <typename scalar_t>
void set(
    scalar_t value,
    Matrix<scalar_t>& A,
    Options const& opts = Options())
{
    set(value, value, A, opts);
}

//------------------------------------------------------------------------------
// Level 3 BLAS and LAPACK auxiliary

//-----------------------------------------
// geadd()
template <typename scalar_t>
void geadd(
    scalar_t alpha, Matrix<scalar_t>& A,
    scalar_t beta,  Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// gbmm()
template <typename scalar_t>
void gbmm(
    scalar_t alpha, BandMatrix<scalar_t>& A,
                        Matrix<scalar_t>& B,
    scalar_t beta,      Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// gemm()
template <typename scalar_t>
void gemm(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// gemmA()
template <typename scalar_t>
void gemmA(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// hbmm()
template <typename scalar_t>
void hbmm(
    Side side,
    scalar_t alpha, HermitianBandMatrix<scalar_t>& A,
                                 Matrix<scalar_t>& B,
    scalar_t beta,               Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// hemm()
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-symmetric matrices to hemm;
// disabled for complex
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hemm(side, alpha, AH, B, beta, C, opts);
}

//-----------------------------------------
// symm()
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-Hermitian matrices to symm;
// disabled for complex
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> AS(A);
    symm(side, alpha, AS, B, beta, C, opts);
}

//-----------------------------------------
// trmm()
template <typename scalar_t>
void trmm(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// tbsm()
template <typename scalar_t>
void tbsm(
    Side side,
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A, Pivots& pivots,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options());

template <typename scalar_t>
void tbsm(
    Side side,
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// trsm()
template <typename scalar_t>
void trsm(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// trtri()
template <typename scalar_t>
void trtri(
    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// trtrm()
template <typename scalar_t>
void trtrm(
    TriangularMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// herk()
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
    blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-symmetric matrices to herk;
// disabled for complex
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
    blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> CH(C);
    herk(alpha, A, beta, CH, opts);
}

//-----------------------------------------
// syrk()
template <typename scalar_t>
void syrk(
    scalar_t alpha,          Matrix<scalar_t>& A,
    scalar_t beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-Hermitian matrices to syrk;
// disabled for complex
template <typename scalar_t>
void syrk(
    scalar_t alpha,          Matrix<scalar_t>& A,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> CS(C);
    syrk(alpha, A, beta, CS, opts);
}

//-----------------------------------------
// her2k()
template <typename scalar_t>
void her2k(
    scalar_t alpha,                          Matrix<scalar_t>& A,
                                             Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-symmetric matrices to her2k;
// disabled for complex
template <typename scalar_t>
void her2k(
    scalar_t alpha,                           Matrix<scalar_t>& A,
                                              Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> CH(C);
    her2k(alpha, A, B, beta, CH, opts);
}

//-----------------------------------------
// syr2k()
template <typename scalar_t>
void syr2k(
    scalar_t alpha,          Matrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options());

// forward real-Hermitian matrices to syr2k;
// disabled for complex
template <typename scalar_t>
void syr2k(
    scalar_t alpha,          Matrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> CS(C);
    syr2k(alpha, A, B, beta, CS, opts);
}

//------------------------------------------------------------------------------
// Norms

//-----------------------------------------
// norm()
template <typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(
    Norm norm,
    matrix_type& A,
    Options const& opts = Options());

//-----------------------------------------
// colNorms()
// all cols max norm
template <typename matrix_type>
void colNorms(
    Norm norm,
    matrix_type& A,
    blas::real_type<typename matrix_type::value_type>* values,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// Linear systems

//-----------------------------------------
// LU

//-----------------------------------------
// gbsv()
template <typename scalar_t>
void gbsv(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// gesv()
template <typename scalar_t>
void gesv(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// gesv_nopiv()
template <typename scalar_t>
void gesv_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// gesvMixed()
template <typename scalar_t>
void gesvMixed(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

template <typename scalar_hi, typename scalar_lo>
void gesvMixed(
    Matrix<scalar_hi>& A, Pivots& pivots,
    Matrix<scalar_hi>& B,
    Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

//-----------------------------------------
// gbtrf()
template <typename scalar_t>
void gbtrf(
    BandMatrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
// getrf()
template <typename scalar_t>
void getrf(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

//-----------------------------------------
// getrf_nopiv()
template <typename scalar_t>
void getrf_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// gbtrs()
template <typename scalar_t>
void gbtrs(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// getrs()
template <typename scalar_t>
void getrs(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// getrs_nopiv()
template <typename scalar_t>
void getrs_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// getri()
// In-place
template <typename scalar_t>
void getri(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options());

// Out-of-place
template <typename scalar_t>
void getri(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// Cholesky

//-----------------------------------------
// pbsv()
template <typename scalar_t>
void pbsv(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// posv()
template <typename scalar_t>
void posv(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

// forward real-symmetric matrices to potrf;
// disabled for complex
template <typename scalar_t>
void posv(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    posv(AH, B, opts);
}

//-----------------------------------------
// posvMixed()
template <typename scalar_t>
void posvMixed(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
             Matrix<scalar_t>& X,
    int& iter,
    Options const& opts = Options());

template <typename scalar_hi, typename scalar_lo>
void posvMixed(
    HermitianMatrix<scalar_hi>& A,
             Matrix<scalar_hi>& B,
             Matrix<scalar_hi>& X,
    int& iter,
    Options const& opts = Options());

// todo: forward real-symmetric matrices to posvMixed?

//-----------------------------------------
// pbtrf()
template <typename scalar_t>
void pbtrf(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// potrf()
template <typename scalar_t>
void potrf(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options());

// forward real-symmetric matrices to potrf;
// disabled for complex
template <typename scalar_t>
void potrf(
    SymmetricMatrix<scalar_t>& A,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    potrf(AH, opts);
}

//-----------------------------------------
// pbtrs()
template <typename scalar_t>
void pbtrs(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// potrs()
template <typename scalar_t>
void potrs(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

// forward real-symmetric matrices to potrs;
// disabled for complex
template <typename scalar_t>
void potrs(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    potrs(AH, B, opts);
}

//-----------------------------------------
// potri()
template <typename scalar_t>
void potri(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options());

// todo:
// forward real-symmetric matrices to potrs;
// disabled for complex

//-----------------------------------------
// Symmetric indefinite -- block Aasen's

//-----------------------------------------
// hesv()
template <typename scalar_t>
void hesv(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// sysv()
// forward real-symmetric matrices to hesv;
// disabled for complex
template <typename scalar_t>
void sysv(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hesv(AH, pivots, T, pivots2, H, B, opts);
}

//-----------------------------------------
// hetrf()
template <typename scalar_t>
void hetrf(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options());

//-----------------------------------------
// sytrf()
// forward real-symmetric matrices to hetrf;
// disabled for complex
template <typename scalar_t>
void sytrf(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hetrf(AH, pivots, T, pivots2, H, opts);
}

//-----------------------------------------
// hetrs()
template <typename scalar_t>
void hetrs(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// sytrs()
// forward real-symmetric matrices to hetrs;
// disabled for complex
template <typename scalar_t>
void sytrs(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hetrs(AH, pivots, T, pivots2, B, opts);
}

//------------------------------------------------------------------------------
// QR

//-----------------------------------------
// auxiliary type for T factors
template <typename scalar_t>
using TriangularFactors = std::vector< Matrix<scalar_t> >;

//-----------------------------------------
// Least squares

//-----------------------------------------
// gels()
template <typename scalar_t>
void gels(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& BX,
    Options const& opts = Options());

//-----------------------------------------
// QR

//-----------------------------------------
// geqrf()
template <typename scalar_t>
void geqrf(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
// unmqr()
template <typename scalar_t>
void unmqr(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//-----------------------------------------
// LQ

//-----------------------------------------
// gelqf()
template <typename scalar_t>
void gelqf(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
// unmlq()
template <typename scalar_t>
void unmlq(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// SVD

//-----------------------------------------
// gesvd()
template <typename scalar_t>
void gesvd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& S,
    Options const& opts = Options());

//-----------------------------------------
// ge2tb()
template <typename scalar_t>
void ge2tb(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& TU,
    TriangularFactors<scalar_t>& TV,
    Options const& opts = Options());

//-----------------------------------------
// Bulge Chasing: TriangularBand to Bi-diagonal
// tb2bd()
template <typename scalar_t>
void tb2bd(
    TriangularBandMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// Bi-diagonal SVD
// bdsqr()
template <typename scalar_t>
void bdsqr(
    Job jobu, Job jobvt,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts = Options());

//------------------------------------------------------------------------------
// Eigenvalue decomposition

//-----------------------------------------
// Symmetric/hermitian

//-----------------------------------------
// heev()
template <typename scalar_t>
void heev(
    Job jobz,
    HermitianMatrix<scalar_t>& A, std::vector< blas::real_type<scalar_t> >& W,
             Matrix<scalar_t>& Z,
    Options const& opts = Options());

//-----------------------------------------
// syev()
// forward real-symmetric matrices to heev;
// disabled for complex
template <typename scalar_t>
void syev(
    Job jobz,
    SymmetricMatrix<scalar_t>& A, std::vector< blas::real_type<scalar_t> >& W,
             Matrix<scalar_t>& Z,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    heev(jobz, AH, W, Z, opts);
}

//-----------------------------------------
// Generalized symmetric/hermitian

//-----------------------------------------
// hegv()
template <typename scalar_t>
void hegv(
    int64_t itype, Job jobz,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B, std::vector< blas::real_type<scalar_t> >& W,
             Matrix<scalar_t>& V,
    Options const& opts = Options());

//-----------------------------------------
// sygv()
// forward real-symmetric matrices to hegv;
// disabled for complex
template <typename scalar_t>
void sygv(
    int64_t itype, Job jobz,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B, std::vector< blas::real_type<scalar_t> >& W,
             Matrix<scalar_t>& V,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    HermitianMatrix<scalar_t> BH(B);
    hegv(itype, jobz, AH, BH, W, V, opts);
}

//-----------------------------------------
// hegst()
template <typename scalar_t>
void hegst(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    Options const& opts = Options());

//-----------------------------------------
// sygst()
// forward real-symmetric matrices to hegst;
// disabled for complex
template <typename scalar_t>
void sygst(
    int64_t itype,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    HermitianMatrix<scalar_t> BH(B);
    hegst(itype, AH, BH, opts);
}

//-----------------------------------------
// he2hb()
template <typename scalar_t>
void he2hb(
    HermitianMatrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts = Options());

//-----------------------------------------
// unmtr_he2hb()
template <typename scalar_t>
void unmtr_he2hb(
    Side side, Op op, HermitianMatrix< scalar_t >& A,
    TriangularFactors< scalar_t > T,
    Matrix< scalar_t >& B,
    Options const& opts = Options());

//-----------------------------------------
// hb2st()
template <typename scalar_t>
void hb2st(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts = Options());

//-----------------------------------------
// sterf()
template <typename scalar_t>
void sterf(
    std::vector< scalar_t >& D,
    std::vector< scalar_t >& E,
    Options const& opts = Options());

//-----------------------------------------
// steqr2()
template <typename scalar_t>
void steqr2(
    Job jobz,
    std::vector< blas::real_type<scalar_t> >& D,
    std::vector< blas::real_type<scalar_t> >& E,
    Matrix<scalar_t>& Z,
    Options const& opts = Options());

} // namespace slate

//-----------------------------------------
// Simplified C++ API
#include "simplified_api.hh"

#endif // SLATE_HH
