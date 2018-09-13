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

#include "slate_Matrix.hh"
#include "slate_BandMatrix.hh"
#include "slate_HermitianMatrix.hh"
#include "slate_SymmetricMatrix.hh"
#include "slate_TriangularMatrix.hh"
#include "slate_types.hh"

namespace slate {

// -----------------------------------------------------------------------------
// Level 3 BLAS

//-----------------------------------------
// gemm()
template <typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

//-----------------------------------------
// hemm()
template <Target target, typename scalar_t>
void hemm(blas::Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <typename scalar_t>
void hemm(blas::Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

// forward real-symmetric matrices to hemm;
// disabled for complex
template <typename scalar_t>
void hemm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    hemm(side, alpha, AH, B, beta, C, opts);
}

//-----------------------------------------
// herk()
template <typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

// forward real-symmetric matrices to herk;
// disabled for complex
template <typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> CH(C);
    herk(alpha, A, beta, CH, opts);
}

//-----------------------------------------
// her2k()
template <typename scalar_t>
void her2k(scalar_t alpha,                 Matrix<scalar_t>& A,
                                           Matrix<scalar_t>& B,
           blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>& C,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void her2k(scalar_t alpha,                 Matrix<scalar_t>& A,
                                           Matrix<scalar_t>& B,
           blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>& C,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

// forward real-symmetric matrices to her2k;
// disabled for complex
template <typename scalar_t>
void her2k(scalar_t alpha,                  Matrix<scalar_t>& A,
                                            Matrix<scalar_t>& B,
           blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>& C,
           const std::map<Option, Value>& opts = std::map<Option, Value>(),
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> CH(C);
    her2k(alpha, A, B, beta, CH, opts);
}

//-----------------------------------------
// symm()
template <typename scalar_t>
void symm(blas::Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void symm(blas::Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

// forward real-Hermitian matrices to symm;
// disabled for complex
template <typename scalar_t>
void symm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> AS(A);
    symm(side, alpha, AS, B, beta, C, opts);
}

//-----------------------------------------
// syrk()
template <typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

// forward real-Hermitian matrices to syrk;
// disabled for complex
template <typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  HermitianMatrix<scalar_t>& C,
          const std::map<Option, Value>& opts = std::map<Option, Value>(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> CS(C);
    syrk(alpha, A, beta, CS, opts);
}

//-----------------------------------------
// syr2k()
template <typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>& C,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>& C,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

// forward real-Hermitian matrices to syr2k;
// disabled for complex
template <typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  HermitianMatrix<scalar_t>& C,
           const std::map<Option, Value>& opts = std::map<Option, Value>(),
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    SymmetricMatrix<scalar_t> CS(C);
    syr2k(alpha, A, B, beta, CS, opts);
}

//-----------------------------------------
// trmm()
template <Target target, typename scalar_t>
void trmm(blas::Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <typename scalar_t>
void trmm(blas::Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

//-----------------------------------------
// trsm()
template <typename scalar_t>
void trsm(blas::Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void trsm(blas::Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

//------------------------------------------------------------------------------
// Norms

//-----------------------------------------
// norm()
template <typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(Norm norm, matrix_type& A,
     const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename matrix_type>
blas::real_type<typename matrix_type::value_type>
norm(Norm norm, matrix_type& A,
     const std::map<Option, Value>& opts = std::map<Option, Value>());

// -----------------------------------------------------------------------------
// Factorizations, etc.

//-----------------------------------------
// gbtrf
template <typename scalar_t>
void gbtrf(BandMatrix<scalar_t>& A, Pivots& pivots,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void gbtrf(BandMatrix<scalar_t>& A, Pivots& pivots,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

//-----------------------------------------
// gesv
template <typename scalar_t>
void gesv(Matrix<scalar_t>& A, Pivots& pivots,
          Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void gesv(Matrix<scalar_t>& A, Pivots& pivots,
          Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts = std::map<Option, Value>());

//-----------------------------------------
// getrf
template <typename scalar_t>
void getrf(Matrix<scalar_t>& A, Pivots& pivots,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void getrf(Matrix<scalar_t>& A, Pivots& pivots,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

//-----------------------------------------
// getrs
template <typename scalar_t>
void getrs(Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void getrs(Matrix<scalar_t>& A, Pivots& pivots,
           Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

//-----------------------------------------
// potrf
template <typename scalar_t>
void potrf(HermitianMatrix<scalar_t>& A,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void potrf(HermitianMatrix<scalar_t>& A,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

// forward real-symmetric matrices to potrf;
// disabled for complex
template <typename scalar_t>
void potrf(SymmetricMatrix<scalar_t>& A,
           const std::map<Option, Value>& opts = std::map<Option, Value>(),
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    HermitianMatrix<scalar_t> AH(A);
    potrf(AH);
}

//-----------------------------------------
// potrs
template <typename scalar_t>
void potrs(HermitianMatrix<scalar_t>& A, Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void potrs(HermitianMatrix<scalar_t>& A, Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

//-----------------------------------------
// posv
template <typename scalar_t>
void posv(HermitianMatrix<scalar_t>& A, Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

template <Target target, typename scalar_t>
void posv(HermitianMatrix<scalar_t>& A, Matrix<scalar_t>& B,
           const std::map<Option, Value>& opts = std::map<Option, Value>());

} // namespace slate

#endif // SLATE_HH
