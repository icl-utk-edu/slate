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

///-----------------------------------------------------------------------------
/// \file
///
#ifndef SLATE_INTERNAL_HH
#define SLATE_INTERNAL_HH

#include "slate_types.hh"

#include "slate_cuda.hh"
#include "slate_cublas.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

#include "slate_Matrix.hh"
#include "slate_HermitianMatrix.hh"
#include "slate_SymmetricMatrix.hh"
#include "slate_TriangularMatrix.hh"

///-----------------------------------------------------------------------------
#define THROW_IF(cond, error) \
    if (cond) \
        throw TrueConditionException( \
            #cond, error, __FILE__, __func__, __LINE__);

#define THROW_IF_NOT(cond, error) \
    if (!(cond)) \
        throw FalseConditionException( \
            #cond, error, __FILE__, __func__, __LINE__);

#define MPI_CALL(call) \
{ \
    int retval = (call); \
    if (retval != MPI_SUCCESS) \
        throw MpiException(#call, retval, __FILE__, __func__, __LINE__); \
}

#define CUDA_CALL(call) \
{ \
    cudaError_t error = (call); \
    if (error != cudaSuccess) \
        throw CudaException(#call, error, __FILE__, __func__, __LINE__); \
}

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
inline CBLAS_TRANSPOSE cblas_trans_const(Op op)
{
    switch (op) {
        case Op::NoTrans:   return CblasNoTrans;
        case Op::Trans:     return CblasTrans;
        case Op::ConjTrans: return CblasConjTrans;
        default: assert( false );
    }
}

///-----------------------------------------------------------------------------
inline cublasOperation_t cublas_op_const(Op op)
{
    switch (op) {
        case Op::NoTrans:   return CUBLAS_OP_N;
        case Op::Trans:     return CUBLAS_OP_T;
        case Op::ConjTrans: return CUBLAS_OP_C;
        default: assert(false);
    }
}

//------------------------------------------------------------------------------
// BLAS and LAPACK routines that update portions of a matrix on each node,
// as steps in a larger parallel factorization or operation.
// E.g., this gemm multiplies one block column by one block row to update the
// trailing matrix. These operations can be mapped to batch BLAS.

//------------------------------------------------------------------------------
// Level 3 BLAS

//-----------------------------------------
// gemm()
template <Target target=Target::HostTask, typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0);

//-----------------------------------------
// hemm()
template <Target target=Target::HostTask, typename scalar_t>
void hemm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0);

// forward real-symmetric matrices to hemm;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void hemm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0,
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    hemm<target>(side, alpha, std::move(A),
                 beta, HermitianMatrix<scalar_t>(C), priority);
}

//-----------------------------------------
// herk()
template <Target target=Target::HostTask, typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>&& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>&& C,
          int priority=0);

// forward real-symmetric matrices to herk;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>&& A,
          blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>&& C,
          int priority=0,
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    herk<target>(alpha, std::move(A),
                 beta, HermitianMatrix<scalar_t>(C), priority);
}

//-----------------------------------------
// her2k()
template <Target target=Target::HostTask, typename scalar_t>
void her2k(scalar_t alpha,                  Matrix< scalar_t >&& A,
                                            Matrix< scalar_t >&& B,
           blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>&& C,
           int priority=0);

// forward real-symmetric matrices to her2k;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void her2k(scalar_t alpha,                  Matrix<scalar_t>&& A,
                                            Matrix<scalar_t>&& B,
           blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>&& C,
           int priority=0,
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    her2k<target>(alpha, std::move(A),
                  beta, HermitianMatrix<scalar_t>(C), priority);
}

//-----------------------------------------
// symm()
template <Target target=Target::HostTask, typename scalar_t>
void symm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0);

// forward real-Hermitian matrices to symm;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void symm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0,
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    symm<target>(side, alpha, std::move(A),
                 beta, SymmetricMatrix<scalar_t>(C), priority);
}

//-----------------------------------------
// syrk()
template <Target target=Target::HostTask, typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>&& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>&& C,
          int priority=0);

// forward real-Hermitian matrices to syrk;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>&& A,
          scalar_t beta,  HermitianMatrix<scalar_t>&& C,
          int priority=0,
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    syrk<target>(alpha, std::move(A),
                 beta, SymmetricMatrix<scalar_t>(C), priority);
}

//-----------------------------------------
// syr2k()
template <Target target=Target::HostTask, typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>&& C,
           int priority=0);

// forward real-Hermitian matrices to syr2k;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  HermitianMatrix<scalar_t>&& C,
           int priority=0,
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    syr2k<target>(alpha, std::move(A), std::move(B),
                  beta, SymmetricMatrix<scalar_t>(C), priority);
}

//-----------------------------------------
// trmm()
template <Target target=Target::HostTask, typename scalar_t>
void trmm(Side side, Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority=0);

//-----------------------------------------
// trsm()
template <Target target=Target::HostTask, typename scalar_t>
void trsm(Side side, Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority=0);

//------------------------------------------------------------------------------
// Norms

//-----------------------------------------
// genorm()
template <Target target=Target::HostTask, typename scalar_t>
void genorm(Norm norm, Matrix<scalar_t>&& A,
            blas::real_type<scalar_t>* values,
            int priority=0);

//-----------------------------------------
// synorm()
template <Target target=Target::HostTask, typename scalar_t>
void synorm(Norm norm, SymmetricMatrix<scalar_t>&& A,
            blas::real_type<scalar_t>* values,
            int priority=0);

//-----------------------------------------
// trnorm()
template <Target target=Target::HostTask, typename scalar_t>
void trnorm(Norm norm, Diag diag, TrapezoidMatrix<scalar_t>&& A,
            blas::real_type<scalar_t>* values,
            int priority=0);

//------------------------------------------------------------------------------
// Factorizations

//-----------------------------------------
// potrf()
template <Target target=Target::HostTask, typename scalar_t>
void potrf(HermitianMatrix<scalar_t>&& A,
           int priority=0);

// forward real-symmetric matrices to potrf;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void potrf(SymmetricMatrix<scalar_t>&& A,
           int priority=0,
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    potrf<target>(SymmetricMatrix<scalar_t>(A), priority);
}

} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_HH
