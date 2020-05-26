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

#ifndef SIMPLIFIED_API_HH
#define SIMPLIFIED_API_HH

//------------------------------------------------------------------------------
//
namespace slate {

//------------------------------------------------------------------------------
// Level 3 BLAS

//-----------------------------------------
// multiply()

//-----------------------------------------
// gbmm
template <typename scalar_t>
void multiply(scalar_t alpha, BandMatrix<scalar_t>& A,
                                  Matrix<scalar_t>& B,
              scalar_t beta,      Matrix<scalar_t>& C,
              const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    gbmm(alpha, A, B, beta, C, opts);
}

//-----------------------------------------
// gemm
template <typename scalar_t>
void multiply(scalar_t alpha, Matrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
              scalar_t beta,  Matrix<scalar_t>& C,
              const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    gemm(alpha, A, B, beta, C, opts);
}

//-----------------------------------------
// hbmm
template <typename scalar_t>
void multiply(Side side,
              scalar_t alpha, HermitianBandMatrix<scalar_t>& A,
                                           Matrix<scalar_t>& B,
              scalar_t beta,               Matrix<scalar_t>& C,
              const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    hbmm(side, alpha, A, B, beta, C, opts);
}

//-----------------------------------------
// hemm
template <typename scalar_t>
void multiply(Side side,
              scalar_t alpha, HermitianMatrix<scalar_t>& A,
                                       Matrix<scalar_t>& B,
              scalar_t beta,           Matrix<scalar_t>& C,
              const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    hemm(side, alpha, A, B, beta, C, opts);
}

//-----------------------------------------
// symm
template <typename scalar_t>
void multiply(Side side,
              scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                                       Matrix<scalar_t>& B,
              scalar_t beta,           Matrix<scalar_t>& C,
              const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    symm(side, alpha, A, B, beta, C, opts);
}

//-----------------------------------------
// trmm
template <typename scalar_t>
void multiply(Side side,
              scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                        Matrix<scalar_t>& B,
              const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    trmm(side, alpha, A, B, opts);
}

//-----------------------------------------
// rankkUpdate()

//-----------------------------------------
// herk
template <typename scalar_t>
void rankkUpdate(blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
                 blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
                const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    herk(alpha, A, beta, C, opts);
}

//-----------------------------------------
// syrk()
template <typename scalar_t>
void rankkUpdate(scalar_t alpha,           Matrix<scalar_t>& A,
                 scalar_t beta,   SymmetricMatrix<scalar_t>& C,
                const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    syrk(alpha, A, beta, C, opts);
}

//-----------------------------------------
// rank2kUpdate()

//-----------------------------------------
// her2k
template <typename scalar_t>
void rank2kUpdate(scalar_t alpha,                           Matrix<scalar_t>& A,
                                                            Matrix<scalar_t>& B,
                  blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
                const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    her2k(alpha, A, B, beta, C, opts);
}

//-----------------------------------------
// syr2k
template <typename scalar_t>
void rank2kUpdate(scalar_t alpha,           Matrix<scalar_t>& A,
                                            Matrix<scalar_t>& B,
                  scalar_t beta,   SymmetricMatrix<scalar_t>& C,
                const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    syr2k(alpha, A, B, beta, C, opts);
}

//-----------------------------------------
// triangularSolve()

//-----------------------------------------
// tbsm
template <typename scalar_t>
void triangularSolve(Side side,
                     scalar_t alpha, TriangularBandMatrix<scalar_t>& A,
                                                   Matrix<scalar_t>& B,
                const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    tbsm(side, alpha, A, B, opts);
}

//-----------------------------------------
// tbsm with pivoting
template <typename scalar_t>
void triangularSolve(Side side,
                     scalar_t alpha, TriangularBandMatrix<scalar_t>& A,
                     Pivots& pivots,               Matrix<scalar_t>& B,
                const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    tbsm(side, alpha, A, pivots, B, opts);
}

//-----------------------------------------
// trsm
template <typename scalar_t>
void triangularSolve(Side side,
                     scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                               Matrix<scalar_t>& B,
                const std::map<Option, Value>& opts = std::map<Option, Value>())
{
    trsm(side, alpha, A, B, opts);
}

//------------------------------------------------------------------------------
// Linear systems

gesv	posv	sysv	{ lu, chol, indefinite } Solve( A, B )
getrf	potrf	sytrf	{ lu, chol, indefinite } Factor( A, {pivots,...} )
getrs	potrs	sytrs	{ lu, chol, indefinite } SolveUsingFactor( A, B, {pivots,...} )?  AfterFactor?  WithFactor?
getri	potri	sytri	{ lu, chol, indefinite } InverseUsingFactor( A, {pivots,...} )
gecon	pocon	sycon	{ lu, chol, indefinite } CondUsingFactor( A, {pivots,...} )

} // namespace slate

#endif // SIMPLIFIED_API_HH
