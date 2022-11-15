// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_INTERNAL_HH
#define SLATE_INTERNAL_HH

#include "slate/types.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/HermitianBandMatrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "slate/BandMatrix.hh"
#include "lapack.hh"

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

    AuxPivot(int64_t tile_index,
             int64_t element_offset,
             int64_t local_tile_index,
             int64_t local_offset,
             scalar_t value,
             int rank)
        : tile_index_(tile_index),
          element_offset_(element_offset),
          local_tile_index_(local_tile_index),
          local_offset_(local_offset),
          value_(value),
          rank_(rank)
    {}

    int64_t tileIndex() { return tile_index_; }
    int64_t elementOffset() { return element_offset_; }
    int64_t localTileIndex() { return local_tile_index_; }
    int64_t localOffset() { return local_offset_; }
    scalar_t value() { return value_; }
    int rank() { return rank_; }

    void set_tileIndex(int64_t i) { tile_index_ = i; }
    void set_elementOffset(int64_t i) { element_offset_ = i; }
    void set_localTileIndex(int64_t i) { local_tile_index_ = i; }
    void set_localOffset(int64_t i) { local_offset_ = i; }
    void set_value(scalar_t v) { value_ = v; }
    void set_rank(int r) { rank_ = r; }

private:
    int64_t tile_index_;       ///< tile index in the panel submatrix
    int64_t element_offset_;   ///< pivot offset in the tile
    int64_t local_tile_index_; ///< tile index in the local list
    int64_t local_offset_;     ///< pivot offset in the local list
    scalar_t value_;           ///< value of the pivot element
    int rank_;                 ///< MPI rank of the pivot owner
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
          int priority=0, int queue_index=0);

template <Target target=Target::HostTask,
          typename src_scalar_t, typename dst_scalar_t>
void copy(BaseTrapezoidMatrix<src_scalar_t>&& A,
          BaseTrapezoidMatrix<dst_scalar_t>&& B,
          int priority=0, int queue_index=0);

//-----------------------------------------
// scale()
template <Target target=Target::HostTask, typename scalar_t>
void scale(blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           Matrix<scalar_t>&& A,
           int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void scale(blas::real_type<scalar_t> numer, blas::real_type<scalar_t> denom,
           BaseTrapezoidMatrix<scalar_t>&& A,
           int priority=0, int queue_index=0);

//-----------------------------------------
// scale_row_col
template <Target target=Target::HostTask, typename scalar_t, typename scalar_t2>
void scale_row_col(
    Equed equed,
    std::vector< scalar_t2 > const& R,
    std::vector< scalar_t2 > const& C,
    Matrix<scalar_t>&& A );

//-----------------------------------------
// set()
template <Target target=Target::HostTask, typename scalar_t>
void set(scalar_t offdiag_value, scalar_t diag_value,
         Matrix<scalar_t>&& A,
         int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void set(scalar_t offdiag_value, scalar_t diag_value,
         BaseTrapezoidMatrix<scalar_t>&& A,
         int priority=0, int queue_index=0);

//-----------------------------------------
// copytb2bd, copyhb2st
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
          Layout layout, int priority=0, int64_t queue_index=0,
          Options const& opts = Options());

//-----------------------------------------
// gemmA()
template <Target target=Target::HostTask, typename scalar_t>
void gemmA(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  Matrix<scalar_t>&& C,
           Layout layout, int priority=0, int64_t queue_index=0,
           Options const& opts = Options());

//-----------------------------------------
// hemm()
template <Target target=Target::HostTask, typename scalar_t>
void hemm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0,
          Options const& opts = Options());

// forward real-symmetric matrices to hemm;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void hemm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0,
          Options const& opts = Options(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    hemm<target>(side, alpha, std::move(A),
                 beta, HermitianMatrix<scalar_t>(C), priority, opts);
}

//-----------------------------------------
// hemmA()
template <Target target=Target::HostTask, typename scalar_t>
void hemmA(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0);

//-----------------------------------------
// herk()
template <Target target=Target::HostTask, typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>&& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>&& C,
          int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
          Options const& opts = Options());

// forward real-symmetric matrices to herk;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>&& A,
          blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>&& C,
          int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
          Options const& opts = Options(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    herk<target>(alpha, std::move(A),
                 beta, HermitianMatrix<scalar_t>(C),
                 priority, queue_index, layout, opts);
}

//-----------------------------------------
// her2k()
template <Target target=Target::HostTask, typename scalar_t>
void her2k(scalar_t alpha,                 Matrix<scalar_t>&& A,
                                           Matrix<scalar_t>&& B,
           blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>&& C,
           int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
           Options const& opts = Options());

// forward real-symmetric matrices to her2k;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void her2k(scalar_t alpha,                  Matrix<scalar_t>&& A,
                                            Matrix<scalar_t>&& B,
           blas::real_type<scalar_t> beta,  SymmetricMatrix<scalar_t>&& C,
           int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
           Options const& opts = Options(),
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    her2k<target>(alpha, std::move(A),
                  beta, HermitianMatrix<scalar_t>(C),
                  priority, queue_index, layout, opts);
}

//-----------------------------------------
// symm()
template <Target target=Target::HostTask, typename scalar_t>
void symm(Side side,
          scalar_t alpha, SymmetricMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0,
          Options const& opts = Options());

// forward real-Hermitian matrices to symm;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void symm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority=0,
          Options const& opts = Options(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    symm<target>(side, alpha, std::move(A),
                 beta, SymmetricMatrix<scalar_t>(C), priority, opts);
}

//-----------------------------------------
// syrk()
template <Target target=Target::HostTask, typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>&& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>&& C,
          int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
          Options const& opts = Options());

// forward real-Hermitian matrices to syrk;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>&& A,
          scalar_t beta,  HermitianMatrix<scalar_t>&& C,
          int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
          Options const& opts = Options(),
          enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    syrk<target>(alpha, std::move(A),
                 beta, SymmetricMatrix<scalar_t>(C),
                 priority, queue_index, layout);
}

//-----------------------------------------
// syr2k()
template <Target target=Target::HostTask, typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>&& C,
           int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
           Options const& opts = Options());

// forward real-Hermitian matrices to syr2k;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  HermitianMatrix<scalar_t>&& C,
           int priority=0, int queue_index=0, Layout layout=Layout::ColMajor,
           Options const& opts = Options(),
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    syr2k<target>(alpha, std::move(A), std::move(B),
                  beta, SymmetricMatrix<scalar_t>(C),
                  priority, queue_index, layout, opts);
}

//-----------------------------------------
// trmm()
template <Target target=Target::HostTask, typename scalar_t>
void trmm(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority=0, int64_t queue_index=0);

//-----------------------------------------
// trsm()
template <Target target=Target::HostTask, typename scalar_t>
void trsm(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority=0, Layout layout=Layout::ColMajor,
          int64_t queue_index=0, Options const& opts = Options());

//-----------------------------------------
// trsmA()
template <Target target=Target::HostTask, typename scalar_t>
void trsmA(Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>&& A,
                                    Matrix<scalar_t>&& B,
          int priority=0, Layout layout=Layout::ColMajor,
          int64_t queue_index=0);

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
    Layout layout, int priority=0, int tag=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void permuteRowsCols(
    Direction direction,
    HermitianMatrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    int priority=0, int tag=0);

//------------------------------------------------------------------------------
// Other BLAS-like
template <Target target=Target::HostTask, typename scalar_t>
void add(scalar_t alpha, Matrix<scalar_t>&& A,
         scalar_t beta,  Matrix<scalar_t>&& B,
         int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void add(scalar_t alpha, BaseTrapezoidMatrix<scalar_t>&& A,
         scalar_t beta,  BaseTrapezoidMatrix<scalar_t>&& B,
         int priority=0, int queue_index=0);

//------------------------------------------------------------------------------
// Bidiagonal band reduction
template <Target target, typename scalar_t>
void gebr1(Matrix<scalar_t>&& A,
           int64_t n1, scalar_t* v1,
           int64_t n2, scalar_t* v2,
           int priority=0);

template <Target target, typename scalar_t>
void gebr2(int64_t n1, scalar_t* v1,
           Matrix<scalar_t>&& A,
           int64_t n2, scalar_t* v2,
           int priority=0);

template <Target target, typename scalar_t>
void gebr3(int64_t n1, scalar_t* v1,
           Matrix<scalar_t>&& A,
           int64_t n2, scalar_t* v2,
           int priority=0);

//------------------------------------------------------------------------------
// Tridiagonal band reduction
template <Target target, typename scalar_t>
void hebr1(int64_t n, scalar_t* v,
           HermitianMatrix<scalar_t>&& A,
           int priority=0);

template <Target target, typename scalar_t>
void hebr2(int64_t n1, scalar_t* v1,
           int64_t n2, scalar_t* v2,
           Matrix<scalar_t>&& A,
           int priority=0);

template <Target target, typename scalar_t>
void hebr3(int64_t n, scalar_t* v,
           HermitianMatrix<scalar_t>&& A,
           int priority=0);

//------------------------------------------------------------------------------
// Norms
template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, Matrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, HermitianMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, SymmetricMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, TrapezoidMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, BandMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0, int queue_index=0);

template <Target target=Target::HostTask, typename scalar_t>
void norm(Norm in_norm, NormScope scope, HermitianBandMatrix<scalar_t>&& A,
          blas::real_type<scalar_t>* values,
          int priority=0, int queue_index=0);

//------------------------------------------------------------------------------
// Factorizations

//-----------------------------------------
// getrf()
// todo: Make the signatures of getrf and geqrf uniform.
//       Probably best to do A, pivot, ib, diag_let, ib, ...
//       Possibly compute diag_len in internal.
template <Target target=Target::HostTask, typename scalar_t>
void getrf_panel(
    Matrix<scalar_t>&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    blas::real_type<scalar_t> remote_pivot_threshold,
    int max_panel_threads, int priority=0, int tag=0);

//-----------------------------------------
// getrf_nopiv()
template <Target target=Target::HostTask, typename scalar_t>
void getrf_nopiv(Matrix<scalar_t>&& A,
                 int64_t ib, int priority=0);

//-----------------------------------------
// getrf_tntpiv()
template <Target target=Target::HostTask, typename scalar_t>
void getrf_tntpiv_panel(
    Matrix<scalar_t>&& A, Matrix<scalar_t>&& Awork,
    std::vector< scalar_t* > dwork_array,
    size_t work_size,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority=0);

//-----------------------------------------
// geqrf()
template <Target target=Target::HostTask, typename scalar_t>
void geqrf(Matrix<scalar_t>&& A, Matrix<scalar_t>&& T,
           std::vector< scalar_t* > dwork_array, size_t work_size,
           int64_t ib, int max_panel_threads, int priority=0);

//-----------------------------------------
// For backwards compatibility of
// ge2tb, gelqf, he2hb
// geqrf()
template <Target target=Target::HostTask, typename scalar_t>
void geqrf(Matrix<scalar_t>&& A, Matrix<scalar_t>&& T,
           int64_t ib, int max_panel_threads, int priority=0)
{
    std::vector< scalar_t* > dwork_array(1);
    dwork_array[0] = nullptr;
    geqrf( std::move(A), std::move(T),
           dwork_array, 0, ib, max_panel_threads, priority);
}

//-----------------------------------------
// he2hb_hemm()
template <Target target=Target::HostTask, typename scalar_t>
void he2hb_hemm(HermitianMatrix<scalar_t>&& A,
            Matrix<scalar_t>&& B,
            Matrix<scalar_t>&& C,
            std::vector<int64_t>& panel_rank_rows,
            int priority=0, int64_t queue_index=0);

//-----------------------------------------
// he2hb_trmm()
template <Target target=Target::HostTask, typename scalar_t>
void he2hb_trmm(HermitianMatrix<scalar_t>&& AH,
            Matrix<scalar_t>&& A,
            Matrix<scalar_t>&& B,
            std::vector<int64_t>& panel_rank_rows,
            int priority=0, int64_t queue_index=0);


//-----------------------------------------
// he2hb_gemm()
template <Target target=Target::HostTask, typename scalar_t>
void he2hb_gemm(scalar_t alpha, Matrix<scalar_t>&& A,
                                Matrix<scalar_t>&& B,
                scalar_t beta,  Matrix<scalar_t>&& T,
                int panel_rank,
                int priority=0, int64_t queue_index=0);

//-----------------------------------------
// he2hb_her2k_offdiag_ranks()
template <Target target=Target::HostTask, typename scalar_t>
void he2hb_her2k_offdiag_ranks(
        scalar_t alpha, Matrix<scalar_t>&& A,
                        Matrix<scalar_t>&& B,
        scalar_t beta,  HermitianMatrix<scalar_t>&& C,
        std::vector<int64_t>& panel_rank_rows,
        int priority=0, int64_t queue_index=0);

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
           Matrix<scalar_t>&& W,
           int priority=0, int64_t queue_index=0);

// unmlq()
template <Target target=Target::HostTask, typename scalar_t>
void unmlq(Side side, Op op,
           Matrix<scalar_t>&& A,
           Matrix<scalar_t>&& T,
           Matrix<scalar_t>&& C,
           Matrix<scalar_t>&& W);

//-----------------------------------------
// unmtr_hb2st()
template <Target target=Target::HostTask, typename scalar_t>
void unmtr_hb2st(Side side, Op op,
                 Matrix<scalar_t>& V,
                 Matrix<scalar_t>& C,
                 const std::map<Option, Value>& opts);

//-----------------------------------------
// potrf()
template <Target target=Target::HostTask, typename scalar_t>
void potrf(HermitianMatrix<scalar_t>&& A,
           int priority=0, int64_t queue_index=0,
           lapack::device_info_int* device_info=nullptr);

// forward real-symmetric matrices to potrf;
// disabled for complex
template <Target target=Target::HostTask, typename scalar_t>
void potrf(SymmetricMatrix<scalar_t>&& A,
           int priority=0, int64_t queue_index=0,
           lapack::device_info_int* device_info=nullptr,
           enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    potrf<target>(SymmetricMatrix<scalar_t>(A), priority);
}

//-----------------------------------------
// hegst()
template <Target target=Target::HostTask, typename scalar_t>
void hegst(int64_t itype, HermitianMatrix<scalar_t>&& A,
                          HermitianMatrix<scalar_t>&& B);

//------------------------------------------------------------------------------
// Norm 1 estimate
template <typename scalar_t>
void norm1est(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& V,
    Matrix<int64_t>& S,
    blas::real_type<scalar_t>* one_normest,
    int* kase,
    std::vector<int64_t>& isave,
    Options const& opts = Options());

} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_HH
