// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// W is a block cloumn. T is a upper triangular or trapezoid matrix.
/// at each iteration, check if T is trapezoid, then slice it and use
/// the upper triangular Tk0.
/// For W, multiply the tile W(i, 0) if its rank is either on upper or lower.
/// Dispatches to target implementations.
/// @ingroup he2hb_trmm_internal
///
template <Target target, typename scalar_t>
void he2hb_trmm(HermitianMatrix<scalar_t>&& A, Matrix<scalar_t>&& W,
           Matrix<scalar_t>&& T,
           std::vector<int64_t>& indices,
           uint8_t* row,
           int priority, int64_t queue_index)
{
    he2hb_trmm(internal::TargetType<target>(),
          A, W, T, indices, row, priority, queue_index);
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// W is a block cloumn. T is a upper triangular or trapezoid matrix.
/// at each iteration, check if T is trapezoid, then slice it and use
/// the upper triangular Tk0.
/// For W, multiply the tile W(i, 0) if its rank is either on upper or lower.
/// Host OpenMP task implementation.
/// @ingroup he2hb_trmm_internal
///
template <typename scalar_t>
void he2hb_trmm(internal::TargetType<Target::HostTask>,
           HermitianMatrix<scalar_t>& A,
           Matrix<scalar_t>& W,
           Matrix<scalar_t>& Tlocal,
           std::vector<int64_t>& indices,
           uint8_t* row,
           int priority, int64_t queue_index)
{
    const scalar_t one  = 1;
    int my_rank = A.mpiRank();

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int rank_lower = -1;
    int rank_upper = -1;
    auto T0    = Tlocal.sub(0, 0, 0, 0);
    // todo: check for slicing here, try to move it to he2hb
    // todo: replace W and T by A and B similar to trmm

    for (int64_t i = 0; i < W.mt(); ++i) {
        #pragma omp task depend(inout:row[i])
        {

            for (int64_t j: indices) {
                if (i >= j) { // lower
                    rank_lower = A.tileRank(i, j);
                }
                else { // upper
                    rank_upper = A.tileRank(j, i);
                }
            }
            // If I contributed to Wi, multiply by T.
            if (rank_upper == my_rank || rank_lower == my_rank) {
                // Wi = Wi * T
                //auto T0    = Tlocal.sub(0, 0, 0, 0);
                auto TVAVT0 = W.sub(i, i, 0, 0);

                int64_t mb = T0.tileMb(0);
                int64_t nb = T0.tileNb(0);
                bool trapezoid = (mb < nb);

                W.tileGetForWriting(i, 0, LayoutConvert(layout));
                if (trapezoid) {
                    auto TVAVT00 = TVAVT0(0, 0);
                    int64_t mb1 = TVAVT00.mb();
                    T0     = T0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                    TVAVT0 = TVAVT0.slice(0, mb1-1, 0, mb-1); // first mb1-by-mb part
                }

                auto Tk0 = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
                trmm(Side::Right, Diag::NonUnit,
                     one, std::move(Tk0(0, 0)), TVAVT0(0, 0));
            }
        }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// W is a block cloumn. T is a upper triangular or trapezoid matrix.
/// at each iteration, check if T is trapezoid, then slice it and use
/// the upper triangular Tk0.
/// For W, multiply the tile W(i, 0) if its rank is either on upper or lower.
/// Device implementation.
/// @ingroup he2hb_trmm_internal
///
template <typename scalar_t>
void he2hb_trmm(internal::TargetType<Target::Devices>,
           HermitianMatrix<scalar_t>& A,
           Matrix<scalar_t>& W,
           Matrix<scalar_t>& Tlocal,
           std::vector<int64_t>& indices,
           uint8_t* row,
           int priority, int64_t queue_index)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    int my_rank = A.mpiRank();

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    for (int device = 0; device < W.num_devices(); ++device) {
        #pragma omp task shared(A, W) priority(priority)
        {
            std::set<ij_tuple> W_tiles_set, T0_tiles_set;
            int rank_lower = -1;
            int rank_upper = -1;

            for (int64_t i = 0; i < W.mt(); ++i) {
                for (int64_t j: indices) {
                    if (i >= j) { // lower
                        rank_lower = A.tileRank(i, j);
                    }
                    else { // upper
                        rank_upper = A.tileRank(j, i);
                    }
                }

                if (rank_upper == my_rank || rank_lower == my_rank) {
                    if (device == W.tileDevice(i, 0)) {
                        W_tiles_set.insert({i, 0});
                        //T0_tiles_set.insert({0, 0});
                    }
                }
            }

            int64_t i_interior = W.mt();
            int64_t i_last = 0;
            int64_t mt = W.mt();
            if (W.tileMb(mt-1) != W.tileMb(0)) {
                i_interior = W.mt() - 1;
                i_last = 1;
            }

            int64_t batch_size = W_tiles_set.size();
            if (batch_size > 0) {

                auto T0    = Tlocal.sub(0, 0, 0, 0);
                T0.tileGetForReading(0, 0, device, LayoutConvert(layout));
                W.tileGetForWriting(W_tiles_set, device, LayoutConvert(layout));

                // interior
                std::vector<scalar_t*> w_array0;
                std::vector<scalar_t*> t_array0;
                w_array0.reserve( batch_size );
                t_array0.reserve( batch_size );

                // bottom-right tile
                std::vector<scalar_t*> w_array1;
                std::vector<scalar_t*> t_array1;

                int64_t ldw0 = 0;
                int64_t ldt0 = 0;
                int64_t ldw1 = 0;
                int64_t ldt1 = 0;

                int64_t mb0 = W.tileMb(0);
                int64_t nb0 = W.tileNb(0);
                int64_t mb1 = W.tileMb(W.mt()-1);
                int64_t nb1 = W.tileNb(W.mt()-1);

                int rank_lower = -1;
                int rank_upper = -1;

                for (int64_t i = 0; i < i_interior; ++i) {
                    for (int64_t j: indices) {
                        if (i >= j) { // lower
                            rank_lower = A.tileRank(i, j);
                        }
                        else { // upper
                            rank_upper = A.tileRank(j, i);
                        }
                    }
                    T0    = Tlocal.sub(0, 0, 0, 0);
                    auto TVAVT0 = W.sub(i, i, 0, 0);
                    int64_t mb = T0.tileMb(0);
                    int64_t nb = T0.tileNb(0);
                    bool trapezoid = (mb < nb);

                    if (trapezoid) {
                        auto TVAVT00 = TVAVT0(0, 0);
                        int64_t mb1 = TVAVT00.mb();
                        T0     = T0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                        TVAVT0 = TVAVT0.slice(0, mb1-1, 0, mb-1); // first mb1-by-mb part
                    }
                    auto Tk0 = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);

                    if (rank_upper == my_rank || rank_lower == my_rank) {
                        if (device == W.tileDevice(i, 0)) {
                            w_array0.push_back( TVAVT0(0, 0, device).data() );
                            //w_array0.push_back( W(i, 0, device).data() );
                            t_array0.push_back( Tk0(0, 0, device).data() );
                            ldw0 = TVAVT0(0, 0, device).stride();
                            ldt0 = T0(0, 0, device).stride();
                            mb0 = TVAVT0.tileMb(0);
                            nb0 = TVAVT0.tileNb(0);
                        }
                    }
                }

                if (i_last == 1)
                {
                    int64_t i = W.mt()-1;
                    int rank_lower = -1;
                    int rank_upper = -1;
                    for (int64_t j: indices) {
                        if (i >= j) { // lower
                            rank_lower = A.tileRank(i, j);
                        }
                        else { // upper
                            rank_upper = A.tileRank(j, i);
                        }
                    }
                    T0    = Tlocal.sub(0, 0, 0, 0);
                    auto TVAVT0 = W.sub(i, i, 0, 0);
                    int64_t mb = T0.tileMb(0);
                    int64_t nb = T0.tileNb(0);
                    bool trapezoid = (mb < nb);

                    if (trapezoid) {
                        auto TVAVT00 = TVAVT0(0, 0);
                        int64_t mb1 = TVAVT00.mb();
                        T0     = T0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                        TVAVT0 = TVAVT0.slice(0, mb1-1, 0, mb-1); // first mb1-by-mb part
                    }
                    auto Tk0 = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
                    if (rank_upper == my_rank || rank_lower == my_rank) {
                        if (device == W.tileDevice(i, 0)) {
                            w_array1.push_back( TVAVT0(0, 0, device).data() );
                            t_array1.push_back( Tk0(0, 0, device).data() );
                            ldw1 = TVAVT0(0, 0, device).stride();
                            ldt1 = Tk0(0, 0, device).stride();
                            mb1 = TVAVT0.tileMb(0);
                            nb1 = TVAVT0.tileNb(0);
                        }
                    }
                }

                {
                    trace::Block trace_block("blas::batch::he2hb_trmm");
                    blas::Queue* queue = W.compute_queue(device, queue_index);
                    assert(queue != nullptr);

                    Side sideW = Side::Right;
                    Uplo uploW = Uplo::Upper;
                    Op opW = Op::NoTrans;
                    Diag diagW = Diag::NonUnit;
                    scalar_t alpha = 1.;
                    std::vector<Side>      side_(1, sideW);
                    std::vector<Uplo>      uplo_(1, uploW);
                    std::vector<Op>         opA_(1, opW  );
                    std::vector<Diag>      diag_(1, diagW);
                    std::vector<scalar_t> alpha_(1, alpha);
                    std::vector<int64_t> info;

                    if (w_array0.size() > 0) {
                        std::vector<int64_t>    m(1,  mb0);
                        std::vector<int64_t>    n(1,  nb0);
                        std::vector<int64_t>  ldw(1, ldw0);
                        std::vector<int64_t>  ldt(1, ldt0);
                        blas::batch::trmm(
                            layout, side_, uplo_, opA_, diag_,
                            m, n,
                            alpha_, t_array0, ldt,
                            w_array0, ldw,
                            t_array0.size(), info, *queue);
                    }

                    if (w_array1.size() > 0) {
                        std::vector<int64_t>    m(1,  mb1);
                        std::vector<int64_t>    n(1,  nb1);
                        std::vector<int64_t>  ldw(1, ldw1);
                        std::vector<int64_t>  ldt(1, ldt1);
                        blas::batch::trmm(
                            layout, side_, uplo_, opA_, diag_,
                            m, n,
                            //m, ldt,
                            alpha_, t_array1, ldt,
                            w_array1, ldw,
                            t_array1.size(), info, *queue);
                    }

                    queue->sync();
                }

                if (rank_upper == my_rank || rank_lower == my_rank) {
                    //T0.tileRelease(0, 0, device);
                    for (auto i = 0; i < batch_size; ++i) {
                    //    T0.tileTick(0, 0);
                    }
                }
            }
        }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host nested OpenMP implementation.
/// @ingroup he2hb_trmm_internal
///
template <typename scalar_t>
void he2hb_trmm(internal::TargetType<Target::HostNest>,
           HermitianMatrix<scalar_t>& A,
           Matrix<scalar_t>& W,
           Matrix<scalar_t>& Tlocal,
           std::vector<int64_t>& indices,
           uint8_t* row,
           int priority, int64_t queue_index)
{
    slate_not_implemented("Target::HostNest isn't yet supported.");
}

//------------------------------------------------------------------------------
/// Triangular matrix multiply.
/// Host batched OpenMP implementation.
/// @ingroup he2hb_trmm_internal
///
template <typename scalar_t>
void he2hb_trmm(internal::TargetType<Target::HostBatch>,
           HermitianMatrix<scalar_t>& A,
           Matrix<scalar_t>& W,
           Matrix<scalar_t>& Tlocal,
           std::vector<int64_t>& indices,
           uint8_t* row,
           int priority, int64_t queue_index)
{
    slate_not_implemented("Target::HostBatch isn't yet supported.");
}


//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void he2hb_trmm<Target::HostTask, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& W,
    Matrix<float>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostTask, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& W,
    Matrix<double>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostTask, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& W,
    Matrix< std::complex<float> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostTask, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& W,
    Matrix< std::complex<double> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::Devices, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& W,
    Matrix<float>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::Devices, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& W,
    Matrix<double>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::Devices, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& W,
    Matrix< std::complex<float> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::Devices, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& W,
    Matrix< std::complex<double> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostNest, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& W,
    Matrix<float>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostNest, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& W,
    Matrix<double>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostNest, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& W,
    Matrix< std::complex<float> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostNest, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& W,
    Matrix< std::complex<double> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostBatch, float>(
    HermitianMatrix<float>&& A,
    Matrix<float>&& W,
    Matrix<float>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm<Target::HostBatch, double>(
    HermitianMatrix<double>&& A,
    Matrix<double>&& W,
    Matrix<double>&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostBatch, std::complex<float> >(
    HermitianMatrix< std::complex<float> >&& A,
    Matrix< std::complex<float> >&& W,
    Matrix< std::complex<float> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

// ----------------------------------------
template
void he2hb_trmm< Target::HostBatch, std::complex<double> >(
    HermitianMatrix< std::complex<double> >&& A,
    Matrix< std::complex<double> >&& W,
    Matrix< std::complex<double> >&& T,
    std::vector<int64_t>& indices,
    uint8_t* row,
    int priority, int64_t queue_index);

} // namespace internal
} // namespace slate
