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

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_INTERNAL_HH
#define SLATE_INTERNAL_HH

#include "slate/types.hh"

#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/HermitianBandMatrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "slate/BandMatrix.hh"
#include "internal/internal_batch.hh"

//------------------------------------------------------------------------------
#define THROW_IF(cond, error) \
    if (cond) \
        throw TrueConditionException( \
            #cond, error, __FILE__, __func__, __LINE__);

#define THROW_IF_NOT(cond, error) \
    if (! (cond)) \
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

//------------------------------------------------------------------------------
/// @namespace slate::internal
/// Namespace used for SLATE internal implementation.
/// It is intended that application code would not call any internal SLATE
/// functions.
namespace internal {

/// @namespace slate::internal::specialization
/// Namespace used for target implementations.
/// This differentiates, for example:
/// - internal::specialization::gemm, which is target implementation of the
///   slate::gemm PBLAS, from
/// - internal::gemm, which is one step (one block outer-product) of
///   internal::specialization::gemm.
namespace specialization {
    // here just for documentation
}

//------------------------------------------------------------------------------
inline CBLAS_TRANSPOSE cblas_trans_const(Op op)
{
    switch (op) {
        case Op::NoTrans:   return CblasNoTrans;
        case Op::Trans:     return CblasTrans;
        case Op::ConjTrans: return CblasConjTrans;
        default: assert( false );
    }
}

//------------------------------------------------------------------------------
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
inline cublasFillMode_t cublas_uplo_const(Uplo uplo)
{
    switch (uplo) {
        case Uplo::Lower: return CUBLAS_FILL_MODE_LOWER;
        case Uplo::Upper: return CUBLAS_FILL_MODE_UPPER;
        default: assert(false);
    }
}

//------------------------------------------------------------------------------
inline cublasSideMode_t cublas_side_const(Side side)
{
    switch (side) {
        case Side::Left:  return CUBLAS_SIDE_LEFT;
        case Side::Right: return CUBLAS_SIDE_RIGHT;
        default: assert(false);
    }
}

//------------------------------------------------------------------------------
inline cublasDiagType_t cublas_diag_const(Diag diag)
{
    switch (diag) {
        case Diag::NonUnit: return CUBLAS_DIAG_NON_UNIT;
        case Diag::Unit:    return CUBLAS_DIAG_UNIT;
        default: assert(false);
    }
}

//------------------------------------------------------------------------------
// Auxiliary class to store and communicate the pivot information internally
// in the panel factorization routine.
template <typename scalar_t>
class AuxPivot {
public:
    AuxPivot()
    {}

    AuxPivot(int64_t tile_index,
             int64_t element_offset,
             int64_t local_tile_index,
             scalar_t value,
             int rank)
        : tile_index_(tile_index),
          element_offset_(element_offset),
          local_tile_index_(local_tile_index),
          value_(value),
          rank_(rank)
    {}

    int64_t tileIndex() { return tile_index_; }
    int64_t elementOffset() { return element_offset_; }
    int64_t localTileIndex() { return local_tile_index_; }
    scalar_t value() { return value_; }
    int rank() { return rank_; }

private:
    int64_t tile_index_;       ///< tile index in the panel submatrix
    int64_t element_offset_;   ///< pivot offset in the tile
    int64_t local_tile_index_; ///< tile index in the local list
    scalar_t value_;           ///< value of the pivot element
    int rank_;                 ///< rank of the pivot owner
};

//------------------------------------------------------------------------------
// BLAS and LAPACK routines that update portions of a matrix on each node,
// as steps in a larger parallel factorization or operation.
// E.g., this gemm multiplies one block column by one block row to update the
// trailing matrix. These operations can be mapped to batch BLAS.

//------------------------------------------------------------------------------
// Auxiliary

//-----------------------------------------
// copy()
template <Target target=Target::HostTask,
          typename src_scalar_t, typename dst_scalar_t>
void copy(Matrix<src_scalar_t>&& A,
          Matrix<dst_scalar_t>&& B,
          int priority=0);

template <Target target=Target::HostTask,
          typename src_scalar_t, typename dst_scalar_t>
void copy(BaseTrapezoidMatrix<src_scalar_t>&& A,
          BaseTrapezoidMatrix<dst_scalar_t>&& B,
          int priority=0);

//-----------------------------------------
// set()
template <Target target=Target::HostTask, typename scalar_t>
void set(scalar_t alpha, scalar_t beta,
         Matrix<scalar_t>&& A,
         int priority=0);

template <Target target=Target::HostTask, typename scalar_t>
void copytb2bd(TriangularBandMatrix<scalar_t>& A,
               std::vector< blas::real_type<scalar_t> >& D,
               std::vector< blas::real_type<scalar_t> >& E);

template <Target target=Target::HostTask, typename scalar_t>
void copyhb2st(HermitianBandMatrix<scalar_t>& A,
               std::vector< blas::real_type<scalar_t> >& D,
               std::vector< blas::real_type<scalar_t> >& E);

//------------------------------------------------------------------------------
// Level 3 BLAS and LAPACK auxiliary

//-----------------------------------------
// gemm()
template <Target target=Target::HostTask, typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          Layout layout, int priority=0, int64_t batch_arrays_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void gemmA(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  Matrix<scalar_t>&& C,
           Layout layout, int priority=0);

template <Target target=Target::Devices, typename scalar_t>
void gemmPrep(scalar_t alpha, Matrix<scalar_t>&& A,
                              Matrix<scalar_t>&& B,
              scalar_t beta,  Matrix<scalar_t>&& C,
              GemmBatchArrays<scalar_t>* batchArrays,
              Layout layout, bool prefetched=false, int priority=0);

template <Target target=Target::Devices, typename scalar_t>
void gemmExec(scalar_t alpha, Matrix<scalar_t>&& A,
                              Matrix<scalar_t>&& B,
              scalar_t beta,  Matrix<scalar_t>&& C,
              GemmBatchArrays<scalar_t>* batchArrays,
              Layout layout, int priority=0);

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
void trmm(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority=0);

//-----------------------------------------
// trsm()
template <Target target=Target::HostTask, typename scalar_t>
void trsm(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority=0, Layout layout=Layout::ColMajor,
          int64_t batch_arrays_index=0);

//-----------------------------------------
// trtri()
template <Target target=Target::HostTask, typename scalar_t>
void trtri(TriangularMatrix<scalar_t>&& A,
           int priority=0);

//-----------------------------------------
// trtrm()
template <Target target=Target::HostTask, typename scalar_t>
void trtrm(TriangularMatrix<scalar_t>&& A,
           int priority=0);

//------------------------------------------------------------------------------
// LAPACK auxiliary
template <Target target=Target::HostTask, typename scalar_t>
void permuteRows(
    Direction direction,
    Matrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority=0, int tag=0);

template <Target target=Target::HostTask, typename scalar_t>
void permuteRowsCols(
    Direction direction,
    HermitianMatrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    int priority=0, int tag=0);

//------------------------------------------------------------------------------
// Other BLAS-like
template <Target target=Target::HostTask, typename scalar_t>
void geadd(scalar_t alpha, Matrix<scalar_t>&& A,
           scalar_t beta, Matrix<scalar_t>&& B,
           int priority=0);

//------------------------------------------------------------------------------
// Band reduction
template <Target target, typename scalar_t>
void gebr1(Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v1,
           std::vector<scalar_t>& v2,
           int priority=0);

template <Target target, typename scalar_t>
void gebr2(std::vector<scalar_t> const& v1,
           Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v2,
           int priority=0);

template <Target target, typename scalar_t>
void gebr3(std::vector<scalar_t> const& v1,
           Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v2,
           int priority=0);

//------------------------------------------------------------------------------
// Tridiagonal band reduction
template <Target target, typename scalar_t>
void hebr1(HermitianMatrix<scalar_t>&& A,
           std::vector<scalar_t>& v,
           int priority=0);

template <Target target, typename scalar_t>
void hebr2(std::vector<scalar_t>& v1,
           Matrix<scalar_t>&& A,
           std::vector<scalar_t>& v2,
           int priority=0);

template <Target target, typename scalar_t>
void hebr3(std::vector<scalar_t>& v,
           HermitianMatrix<scalar_t>&& A,
           int priority=0);

//------------------------------------------------------------------------------
// Norms
template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, Matrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, HermitianMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, SymmetricMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, TrapezoidMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, BandMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, HermitianBandMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0);

//------------------------------------------------------------------------------
// Factorizations

//-----------------------------------------
// getrf()
// todo: Make the signatures of getrf and geqrf uniform.
//       Probably best to do A, pivot, ib, diag_let, ib, ...
//       Possibly compute diag_len in internal.
template <Target target=Target::HostTask, typename scalar_t>
void getrf(Matrix<scalar_t>&& A, int64_t diag_len, int64_t ib,
           std::vector<Pivot>& pivot,
           int max_panel_threads, int priority=0);

//-----------------------------------------
// getrf_nopiv()
template <Target target=Target::HostTask, typename scalar_t>
void getrf_nopiv(Matrix<scalar_t>&& A, int64_t diag_len, int64_t ib,
           int max_panel_threads, int priority=0);

//-----------------------------------------
// geqrf()
template <Target target=Target::HostTask, typename scalar_t>
void geqrf(Matrix<scalar_t>&& A, Matrix<scalar_t>&& T, int64_t ib,
           int max_panel_threads, int priority=0);

//-----------------------------------------
// ttqrt()
template <Target target=Target::HostTask, typename scalar_t>
void ttqrt(Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T);

// ttlqt()
template <Target target=Target::HostTask, typename scalar_t>
void ttlqt(Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T);

//-----------------------------------------
// ttmqr()
template <Target target=Target::HostTask, typename scalar_t>
void ttmqr(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           int tag=0);

// ttmlq()
template <Target target=Target::HostTask, typename scalar_t>
void ttmlq(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           int tag=0);

// hettmqr()
template <Target target=Target::HostTask, typename scalar_t>
void hettmqr(Op op,
             Matrix<scalar_t>&& A,
             Matrix<scalar_t>&& T,
             HermitianMatrix<scalar_t>&& C,
             int tag=0);

//-----------------------------------------
// unmqr()
template <Target target=Target::HostTask, typename scalar_t>
void unmqr(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           Matrix<scalar_t>&& W);

// unmlq()
template <Target target=Target::HostTask, typename scalar_t>
void unmlq(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           Matrix<scalar_t>&& W);

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

//-----------------------------------------
// hegst()
template <Target target=Target::HostTask, typename scalar_t>
void hegst(int64_t itype, HermitianMatrix<scalar_t>&& A,
                          HermitianMatrix<scalar_t>&& B);

} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_HH
