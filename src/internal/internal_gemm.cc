// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// General matrix multiply to update trailing matrix,
/// where A is a single block column and B is a single block row.
/// Dispatches to target implementations.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conjTranspose;
/// if $op(C)$ is conjTranspose, then $op(A)$ and $op(B)$ cannot be transpose.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) to operate with.
///     Local tiles of matrix C and corresponding tiles of A & B
///        on target devices will be converted to layout.
///
/// @ingroup gemm_internal
///
template <Target target, typename scalar_t>
void gemm(scalar_t alpha, Matrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          Layout layout, int priority, int64_t queue_index,
          Options const& opts )
{
    if (C.is_complex &&
        ((C.op() == Op::Trans &&
         (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans)) ||
         (C.op() == Op::ConjTrans &&
         (A.op() == Op::Trans || B.op() == Op::Trans))))
    {
        throw std::exception();
    }

    gemm(internal::TargetType<target>(),
         alpha, A,
                B,
         beta,  C,
         layout, priority, queue_index, opts);
}

//------------------------------------------------------------------------------
/// General matrix multiply to update trailing matrix,
/// where A is a single block column and B is a single block row.
/// Host OpenMP task implementation.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(internal::TargetType<Target::HostTask>,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          Layout layout, int priority, int64_t queue_index,
          Options const& opts )
{
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'

    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    // check dimensions
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    int err = 0;
    std::string err_msg;
    std::set<ij_tuple> A_tiles_set, B_tiles_set;
    for (int64_t i = 0; i < C.mt(); ++i) {
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(i, j)) {
                A_tiles_set.insert({i, 0});
                B_tiles_set.insert({0, j});
            }
        }
    }
    A.tileGetForReading(A_tiles_set, LayoutConvert(layout));
    B.tileGetForReading(B_tiles_set, LayoutConvert(layout));

    #pragma omp taskgroup
    for (int64_t i = 0; i < C.mt(); ++i) {
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B, C, err, err_msg ) \
                    firstprivate(i, j, layout, alpha, beta, call_tile_tick) \
                    priority(priority)
                {
                    try {
                        C.tileGetForWriting(i, j, LayoutConvert(layout));
                        tile::gemm(
                            alpha, A(i, 0), B(0, j),
                            beta,  C(i, j) );
                        if (call_tile_tick) {
                            // todo: shouldn't tileRelease()?
                            A.tileTick(i, 0);
                            B.tileTick(0, j);
                        }
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                        err_msg = std::string(e.what());
                    }
                }
            }
        }
    }

    if (err)
        slate_error(err_msg+", line "+std::to_string(err));
}

//------------------------------------------------------------------------------
/// General matrix multiply to update trailing matrix,
/// where A is a single block column and B is a single block row.
/// Host nested OpenMP implementation.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(internal::TargetType<Target::HostNest>,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          Layout layout, int priority, int64_t queue_index,
          Options const& opts )
{
    // check dimensions
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

    int err = 0;
    std::string err_msg;
    int64_t C_mt = C.mt();
    int64_t C_nt = C.nt();

    #pragma omp parallel for collapse(2) schedule(dynamic, 1) slate_omp_default_none \
        shared(A, B, C, err, err_msg) firstprivate(C_nt, C_mt, layout, alpha, beta)
    for (int64_t i = 0; i < C_mt; ++i) {
        for (int64_t j = 0; j < C_nt; ++j) {
            if (C.tileIsLocal(i, j)) {
                try {
                    A.tileGetForReading(i, 0, LayoutConvert(layout));
                    B.tileGetForReading(0, j, LayoutConvert(layout));
                    C.tileGetForWriting(i, j, LayoutConvert(layout));
                    tile::gemm(
                        alpha, A(i, 0), B(0, j),
                        beta,  C(i, j) );
                    // todo: shouldn't tileRelease()?
                    A.tileTick(i, 0);
                    B.tileTick(0, j);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                    err_msg = std::string(e.what());
                }
            }
        }
    }

    if (err)
        slate_error(err_msg+", line "+std::to_string(err));
}

//------------------------------------------------------------------------------
/// General matrix multiply to update trailing matrix,
/// where A is a single block col (mt tiles by 1 tile)
/// and   B is a single block row (1 tile by nt tiles)
/// and   C is mt tiles by nt tiles.
/// Host batched implementation.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(internal::TargetType<Target::HostBatch>,
          scalar_t alpha, Matrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          Layout layout, int priority, int64_t queue_index,
          Options const& opts )
{
#ifdef BLAS_HAVE_MKL
    using blas::conj;
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // check dimensions
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

    // load off-diagonal tiles to host, if not there
    // also count tiles
    int batch_count = 0;
    std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
    for (int64_t i = 0; i < C.mt(); ++i) {
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(i, j)) {
                A_tiles_set.insert({i, 0});
                B_tiles_set.insert({0, j});
                C_tiles_set.insert({i, j});
                ++batch_count;
            }
        }
    }

    #pragma omp taskgroup
    {
        #pragma omp task slate_omp_default_none \
            shared( A, A_tiles_set ) firstprivate( layout )
        {
            A.tileGetForReading(A_tiles_set, LayoutConvert(layout));
        }
        #pragma omp task slate_omp_default_none \
            shared( B, B_tiles_set ) firstprivate( layout )
        {
            B.tileGetForReading(B_tiles_set, LayoutConvert(layout));
        }
        #pragma omp task slate_omp_default_none \
            shared( C, C_tiles_set ) firstprivate( layout )
        {
            C.tileGetForWriting(C_tiles_set, LayoutConvert(layout));
        }
    }

    if (batch_count > 0) {
        // if op(C) is NoTrans, invert opA, opB if possible
        Op opA = A.op();
        if (C.op() != Op::NoTrans) {
            if (opA == Op::NoTrans)
                opA = C.op();
            else if (A.op() == C.op() || C.is_real) {
                // A and C are both Trans or both ConjTrans;
                // Trans == ConjTrans if real
                opA = Op::NoTrans;
            }
            else
                throw std::exception();
        }

        Op opB = B.op();
        if (C.op() != Op::NoTrans) {
            if (opB == Op::NoTrans)
                opB = C.op();
            else if (opB == C.op() || C.is_real) {
                // B and C are both Trans or both ConjTrans;
                // Trans == ConjTrans if real
                opB = Op::NoTrans;
            }
            else
                throw std::exception();
        }

        if (C.op() == Op::ConjTrans) {
            alpha = conj(alpha);
            beta  = conj(beta);
        }

        // all same
        std::vector<CBLAS_TRANSPOSE> opA_array(batch_count,
                                               cblas_trans_const(opA));
        // all same
        std::vector<CBLAS_TRANSPOSE> opB_array(batch_count,
                                               cblas_trans_const(opB));
        std::vector<int> m_array(batch_count);
        std::vector<int> n_array(batch_count);
        std::vector<int> k_array(batch_count);
        std::vector<scalar_t> alpha_array(batch_count, alpha);  // all same
        std::vector<scalar_t>  beta_array(batch_count,  beta);  // all same
        std::vector<const scalar_t*> a_array(batch_count);
        std::vector<const scalar_t*> b_array(batch_count);
        std::vector<scalar_t*> c_array(batch_count);
        std::vector<int> lda_array(batch_count);
        std::vector<int> ldb_array(batch_count);
        std::vector<int> ldc_array(batch_count);
        std::vector<int> group_size(batch_count, 1);  // all same

        int index = 0;
        for (int64_t i = 0; i < C.mt(); ++i) {
            for (int64_t j = 0; j < C.nt(); ++j) {
                if (C.tileIsLocal(i, j)) {
                    m_array[ index ] = C(i, j).mb();
                    n_array[ index ] = C(i, j).nb();
                    k_array[ index ] = A(i, 0).nb();  // should be all same

                    assert(A(i, 0).mb() == m_array[index]);
                    assert(B(0, j).nb() == n_array[index]);
                    assert(B(0, j).mb() == k_array[index]);

                    a_array[ index ] = A(i, 0).data();
                    b_array[ index ] = B(0, j).data();
                    c_array[ index ] = C(i, j).data();

                    lda_array[ index ] = A(i, 0).stride();
                    ldb_array[ index ] = B(0, j).stride();
                    ldc_array[ index ] = C(i, j).stride();

                    ++index;
                }
            }
        }

        if (C.op() != Op::NoTrans) {
            // swap A <=> B; swap m <=> n
            swap(opA_array, opB_array);
            swap(a_array,   b_array);
            swap(lda_array, ldb_array);
            swap(m_array,   n_array);
        }

        {
            trace::Block trace_block("cblas_gemm_batch");
            // mkl_set_num_threads_local(...);
            if (layout == Layout::ColMajor) {
                cblas_gemm_batch(
                    CblasColMajor,
                    opA_array.data(), opB_array.data(),
                    m_array.data(), n_array.data(), k_array.data(),
                    alpha_array.data(), a_array.data(), lda_array.data(),
                                        b_array.data(), ldb_array.data(),
                    beta_array.data(),  c_array.data(), ldc_array.data(),
                    batch_count, group_size.data());
            }
            else {
                cblas_gemm_batch(
                    CblasColMajor,
                    opB_array.data(), opA_array.data(),
                    n_array.data(), m_array.data(), k_array.data(),
                    alpha_array.data(), b_array.data(), ldb_array.data(),
                                        a_array.data(), lda_array.data(),
                    beta_array.data(),  c_array.data(), ldc_array.data(),
                    batch_count, group_size.data());
            }
            // mkl_set_num_threads_local(1);
        }

        for (int64_t i = 0; i < C.mt(); ++i) {
            for (int64_t j = 0; j < C.nt(); ++j) {
                if (C.tileIsLocal(i, j)) {
                    // todo: shouldn't tileRelease()?
                    A.tileTick(i, 0);
                    B.tileTick(0, j);
                }
            }
        }
    }
#else
    slate_not_implemented( "HostBatch requires Intel MKL" );
#endif
}

//------------------------------------------------------------------------------
/// General matrix multiply to update trailing matrix,
/// where A is a single block column and B is a single block row.
/// GPU device batched-BLAS implementation.
/// GPU can use either ColMajor or RowMajor.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(internal::TargetType<Target::Devices>,
          scalar_t alpha, Matrix< scalar_t >& A,
                          Matrix< scalar_t >& B,
          scalar_t beta,  Matrix< scalar_t >& C,
          Layout layout, int priority, int64_t queue_index,
          Options const& opts
          )
{
    using blas::conj;
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );
    // check dimensions
    assert(C.mt() > 0);
    assert(C.nt() > 0);
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

    assert(C.num_devices() > 0);

    int err = 0;

    #pragma omp taskgroup
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, B, C, err) priority(priority) \
            firstprivate(alpha, beta, layout, queue_index, device, tile_release_strategy)
        {
            // if op(C) is NoTrans, invert opA, opB if possible
            Op opA = A.op();
            if (C.op() != Op::NoTrans) {
                if (opA == Op::NoTrans)
                    opA = C.op();
                else if (A.op() == C.op() || C.is_real) {
                    // A and C are both Trans or both ConjTrans;
                    // Trans == ConjTrans if real
                    opA = Op::NoTrans;
                }
                else {
                    err = __LINE__;  // ConjNoTrans not supported
                }
            }

            Op opB = B.op();
            if (C.op() != Op::NoTrans) {
                if (opB == Op::NoTrans)
                    opB = C.op();
                else if (opB == C.op() || C.is_real) {
                    // B and C are both Trans or both ConjTrans;
                    // Trans == ConjTrans if real
                    opB = Op::NoTrans;
                }
                else {
                    err = __LINE__;  // ConjNoTrans not supported
                }
            }

            if (C.op() == Op::ConjTrans) {
                alpha = conj(alpha);
                beta  = conj(beta);
            }

            std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
            for (int64_t i = 0; i < C.mt(); ++i) {
                for (int64_t j = 0; j < C.nt(); ++j) {
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            A_tiles_set.insert({i, 0});
                            B_tiles_set.insert({0, j});
                            C_tiles_set.insert({i, j});
                        }
                    }
                }
            }

            #pragma omp taskgroup
            {
                #pragma omp task slate_omp_default_none \
                    shared( A, A_tiles_set ) firstprivate( layout, device )
                {
                    A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp task slate_omp_default_none \
                    shared( B, B_tiles_set ) firstprivate( layout, device )
                {
                    B.tileGetForReading(B_tiles_set, device, LayoutConvert(layout));
                }
                #pragma omp task slate_omp_default_none \
                    shared( C, C_tiles_set ) firstprivate( layout, device )
                {
                    C.tileGetForWriting(C_tiles_set, device, LayoutConvert(layout));
                }
            }

            int64_t batch_size = C_tiles_set.size();

            // interior, excluding bottom row and right column
            std::vector<scalar_t*> a_array00;
            std::vector<scalar_t*> b_array00;
            std::vector<scalar_t*> c_array00;
            a_array00.reserve( batch_size );
            b_array00.reserve( batch_size );
            c_array00.reserve( batch_size );

            int64_t lda00 = 0;
            int64_t ldb00 = 0;
            int64_t ldc00 = 0;
            int64_t mb00 = C.tileMb(0);
            int64_t nb00 = C.tileNb(0);
            int64_t kb   = A.tileNb(0);
            for (int64_t i = 0; i < C.mt()-1; ++i) {
                for (int64_t j = 0; j < C.nt()-1; ++j) {
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            a_array00.push_back( A(i, 0, device).data() );
                            b_array00.push_back( B(0, j, device).data() );
                            c_array00.push_back( C(i, j, device).data() );
                            lda00 = A(i, 0, device).stride();
                            ldb00 = B(0, j, device).stride();
                            ldc00 = C(i, j, device).stride();
                        }
                    }
                }
            }

            // bottom row
            std::vector<scalar_t*> a_array10;
            std::vector<scalar_t*> b_array10;
            std::vector<scalar_t*> c_array10;
            a_array10.reserve( batch_size );
            b_array10.reserve( batch_size );
            c_array10.reserve( batch_size );

            int64_t lda10 = 0;
            int64_t ldb10 = 0;
            int64_t ldc10 = 0;
            int64_t mb10 = C.tileMb(C.mt()-1);
            int64_t nb10 = C.tileNb(0);
            // same kb as above
            {
                int64_t i = C.mt()-1;
                for (int64_t j = 0; j < C.nt()-1; ++j) {
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            a_array10.push_back( A(i, 0, device).data() );
                            b_array10.push_back( B(0, j, device).data() );
                            c_array10.push_back( C(i, j, device).data() );
                            lda10 = A(i, 0, device).stride();
                            ldb10 = B(0, j, device).stride();
                            ldc10 = C(i, j, device).stride();
                        }
                    }
                }
            }

            // right column
            std::vector<scalar_t*> a_array01;
            std::vector<scalar_t*> b_array01;
            std::vector<scalar_t*> c_array01;
            a_array01.reserve( batch_size );
            b_array01.reserve( batch_size );
            c_array01.reserve( batch_size );

            int64_t lda01 = 0;
            int64_t ldb01 = 0;
            int64_t ldc01 = 0;
            int64_t mb01 = C.tileMb(0);
            int64_t nb01 = C.tileNb(C.nt()-1);
            // same kb as above
            {
                int64_t j = C.nt()-1;
                for (int64_t i = 0; i < C.mt()-1; ++i) {
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            a_array01.push_back( A(i, 0, device).data() );
                            b_array01.push_back( B(0, j, device).data() );
                            c_array01.push_back( C(i, j, device).data() );
                            lda01 = A(i, 0, device).stride();
                            ldb01 = B(0, j, device).stride();
                            ldc01 = C(i, j, device).stride();
                        }
                    }
                }
            }

            // bottom-right corner
            std::vector<scalar_t*> a_array11;
            std::vector<scalar_t*> b_array11;
            std::vector<scalar_t*> c_array11;

            int64_t lda11 = 0;
            int64_t ldb11 = 0;
            int64_t ldc11 = 0;
            int64_t mb11 = C.tileMb(C.mt()-1);
            int64_t nb11 = C.tileNb(C.nt()-1);
            // same kb as above
            {
                int i = C.mt()-1;
                int j = C.nt()-1;
                if (C.tileIsLocal(i, j)) {
                    if (device == C.tileDevice(i, j)) {
                        a_array11.push_back( A(i, 0, device).data() );
                        b_array11.push_back( B(0, j, device).data() );
                        c_array11.push_back( C(i, j, device).data() );
                        lda11 = A(i, 0, device).stride();
                        ldb11 = B(0, j, device).stride();
                        ldc11 = C(i, j, device).stride();
                    }
                }
            }

            if (C.op() != Op::NoTrans) {
                // swap A <=> B; swap m <=> n
                swap(opA, opB);
                swap(a_array00, b_array00);
                swap(a_array10, b_array10);
                swap(a_array01, b_array01);
                swap(a_array11, b_array11);
                swap(lda00, ldb00);
                swap(lda10, ldb10);
                swap(lda01, ldb01);
                swap(lda11, ldb11);
                swap(mb00, nb00);
                swap(mb10, nb10);
                swap(mb01, nb01);
                swap(mb11, nb11);
            }

            {
                trace::Block trace_block("blas::batch::gemm");

                std::vector<Op> opA_(1, opA);
                std::vector<Op> opB_(1, opB);
                std::vector<scalar_t> alpha_(1, alpha);
                std::vector<scalar_t> beta_(1, beta);
                std::vector<int64_t> k(1, kb);
                // info size 0 disables slow checks in batched BLAS++.
                std::vector<int64_t> info;

                blas::Queue* queue = C.compute_queue(device, queue_index);
                assert(queue != nullptr);

                if (c_array00.size() > 0) {
                    std::vector<int64_t>    m(1,  mb00);
                    std::vector<int64_t>    n(1,  nb00);
                    std::vector<int64_t> ldda(1, lda00);
                    std::vector<int64_t> lddb(1, ldb00);
                    std::vector<int64_t> lddc(1, ldc00);
                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, k,
                        alpha_, a_array00, ldda,
                                b_array00, lddb,
                        beta_,  c_array00, lddc,
                        c_array00.size(), info, *queue);
                }

                if (c_array10.size() > 0) {
                    std::vector<int64_t>    m(1,  mb10);
                    std::vector<int64_t>    n(1,  nb10);
                    std::vector<int64_t> ldda(1, lda10);
                    std::vector<int64_t> lddb(1, ldb10);
                    std::vector<int64_t> lddc(1, ldc10);
                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, k,
                        alpha_, a_array10, ldda,
                                b_array10, lddb,
                        beta_,  c_array10, lddc,
                        c_array10.size(), info, *queue);
                }

                if (c_array01.size() > 0) {
                    std::vector<int64_t>    m(1,  mb01);
                    std::vector<int64_t>    n(1,  nb01);
                    std::vector<int64_t> ldda(1, lda01);
                    std::vector<int64_t> lddb(1, ldb01);
                    std::vector<int64_t> lddc(1, ldc01);
                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, k,
                        alpha_, a_array01, ldda,
                                b_array01, lddb,
                        beta_,  c_array01, lddc,
                        c_array01.size(), info, *queue);
                }

                if (c_array11.size() > 0) {
                    std::vector<int64_t>    m(1,  mb11);
                    std::vector<int64_t>    n(1,  nb11);
                    std::vector<int64_t> ldda(1, lda11);
                    std::vector<int64_t> lddb(1, ldb11);
                    std::vector<int64_t> lddc(1, ldc11);
                    blas::batch::gemm(
                        layout, opA_, opB_,
                        m, n, k,
                        alpha_, a_array11, ldda,
                                b_array11, lddb,
                        beta_,  c_array11, lddc,
                        c_array11.size(), info, *queue);
                }

                queue->sync();
            }

            if (tile_release_strategy == TileReleaseStrategy::Internal
                || tile_release_strategy == TileReleaseStrategy::All) {
                for (int64_t i = 0; i < C.mt(); ++i) {
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        if (C.tileIsLocal(i, j)) {
                            if (device == C.tileDevice(i, j)) {
                                // erase tmp local and remote device tiles;
                                A.tileRelease(i, 0, device);
                                B.tileRelease(0, j, device);
                                // decrement life for remote tiles
                                A.tileTick(i, 0);
                                B.tileTick(0, j);
                            }
                        }
                    }
                }
            }
        }
    }

    if (err)
        slate_error(std::to_string(err));
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void gemm<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void gemm<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void gemm< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

// ----------------------------------------
template
void gemm< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

template
void gemm< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts );

} // namespace internal
} // namespace slate
