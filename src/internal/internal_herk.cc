// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Dispatches to target implementations.
/// C is Lower, NoTrans or Upper, Trans/ConjTrans.
/// In complex case, A and C cannot be Trans.
/// @ingroup herk_internal
///
template <Target target, typename scalar_t>
void herk(blas::real_type<scalar_t> alpha, Matrix<scalar_t>&& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>&& C,
          int priority, int queue_index, Layout layout, Options const& opts)
{
    if (! ((C.uplo() == Uplo::Lower)
           &&
           (C.is_real || (C.op() != Op::Trans &&
                          A.op() != Op::Trans))))
        throw std::exception();

    herk(internal::TargetType<target>(),
         alpha, A,
         beta,  C,
         priority, queue_index, layout, opts);
}

//------------------------------------------------------------------------------
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Host OpenMP task implementation.
/// Assumes A is NoTrans or ConjTrans; C is Lower, NoTrans or Upper, ConjTrans.
/// @ingroup herk_internal
///
template <typename scalar_t>
void herk(internal::TargetType<Target::HostTask>,
          blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix< scalar_t >& C,
          int priority, int queue_index, Layout layout, Options const& opts)
{
    scalar_t alpha_ = scalar_t(alpha);
    scalar_t beta_  = scalar_t(beta);

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    // CPU assumes column major
    // todo: relax this assumption, by updating Tile_blas.hh::herk()
    //       to operate in row major
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    assert(layout == Layout::ColMajor);

    // Lower, NoTrans
    int err = 0;
    #pragma omp taskgroup
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j; i < C.mt(); ++i) {  // lower
            if (C.tileIsLocal(i, j)) {
                if (i == j) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, C, err ) priority( priority ) \
                        firstprivate(j, layout, alpha, beta, call_tile_tick)
                    {
                        try {
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(j, j, LayoutConvert(layout));
                            tile::herk(
                                alpha, A(j, 0),
                                beta,  C(j, j) );

                            if (call_tile_tick) {
                                // todo: should tileRelease()?
                                A.tileTick(j, 0);
                                // todo: why the second tick?
                                A.tileTick(j, 0);
                            }
                        }
                        catch (std::exception& e) {
                            err = __LINE__;
                        }
                    }
                }
                else {
                    #pragma omp task slate_omp_default_none \
                        shared( A, C, err ) priority( priority ) \
                        firstprivate(i, j, layout, alpha_, beta_, call_tile_tick)
                    {
                        try {
                            A.tileGetForReading(i, 0, LayoutConvert(layout));
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(i, j, LayoutConvert(layout));
                            auto Aj0 = A(j, 0);
                            tile::gemm(
                                alpha_, A(i, 0), conj_transpose( Aj0 ),
                                beta_,  C(i, j) );

                            if (call_tile_tick) {
                                // todo: should tileRelease()?
                                A.tileTick(i, 0);
                                A.tileTick(j, 0);
                            }
                        }
                        catch (std::exception& e) {
                            err = __LINE__;
                        }
                    }
                }
            }
        }
    }

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Host nested OpenMP implementation.
/// Assumes A is NoTrans or ConjTrans; C is Lower, NoTrans or Upper, ConjTrans.
/// @ingroup herk_internal
///
template <typename scalar_t>
void herk(internal::TargetType<Target::HostNest>,
          blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
          int priority, int queue_index, Layout layout, Options const& opts)
{
    scalar_t alpha_ = scalar_t(alpha);
    scalar_t beta_  = scalar_t(beta);

    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::herk()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    assert(layout == Layout::ColMajor);

    // Lower, NoTrans
    int err = 0;
    #pragma omp taskgroup
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task slate_omp_default_none \
                shared( A, C, err ) \
                firstprivate(j, layout, alpha, beta)
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    tile::herk(
                        alpha, A(j, 0),
                        beta,  C(j, j) );
                    // todo: should tileRelease()?
                    A.tileTick(j, 0);
                    A.tileTick(j, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }

    int64_t C_nt = C.nt();
    int64_t C_mt = C.mt();

    // #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...) default(none)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) slate_omp_default_none \
        shared(A, C, err) firstprivate(C_nt, C_mt, layout, beta_, alpha_)
    for (int64_t j = 0; j < C_nt; ++j) {
        for (int64_t i = 0; i < C_mt; ++i) {  // full
            if (i >= j+1) {                    // strictly lower
                if (C.tileIsLocal(i, j)) {
                    try {
                        A.tileGetForReading(i, 0, LayoutConvert(layout));
                        A.tileGetForReading(j, 0, LayoutConvert(layout));
                        C.tileGetForWriting(i, j, LayoutConvert(layout));
                        auto Aj0 = A(j, 0);
                        tile::gemm(
                            alpha_, A(i, 0), conj_transpose( Aj0 ),
                            beta_,  C(i, j) );
                        // todo: should tileRelease()?
                        A.tileTick(i, 0);
                        A.tileTick(j, 0);
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Host batched implementation.
/// Assumes A is NoTrans or ConjTrans; C is Lower, NoTrans or Upper, ConjTrans.
/// @ingroup herk_internal
///
template <typename scalar_t>
void herk(internal::TargetType<Target::HostBatch>,
          blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
          int priority, int queue_index, Layout layout, Options const& opts)
{
#ifdef BLAS_HAVE_MKL
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::herk()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    assert(layout == Layout::ColMajor);

    // diagonal tiles by herk on host
    int err = 0;
    #pragma omp taskgroup
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task slate_omp_default_none \
                shared( A, C, err ) \
                firstprivate(j, layout, alpha, beta)
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    tile::herk(
                        alpha, A(j, 0),
                        beta,  C(j, j) );
                    // todo: should tileRelease()?
                    A.tileTick(j, 0);
                    A.tileTick(j, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }
    // end omp taskgroup

    // load off-diagonal tiles to host, if not there
    // also count tiles
    int batch_count = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
            if (C.tileIsLocal(i, j)) {
                // todo: omp task?
                A.tileGetForReading(i, 0, LayoutConvert(layout));
                A.tileGetForReading(j, 0, LayoutConvert(layout));
                C.tileGetForWriting(i, j, LayoutConvert(layout));
                ++batch_count;
            }
        }
    }

    if (batch_count > 0) {
        // off-diagonal tiles by batch gemm on host
        Op opA = A.op();
        if (C.op() != Op::NoTrans) {
            if (A.op() == Op::NoTrans)
                opA = C.op();
            else if (A.op() == C.op() || C.is_real) {
                // A and C are both Trans or both ConjTrans;
                // Trans == ConjTrans if real
                opA = Op::NoTrans;
            }
            else
                throw std::exception();
        }

        Op opB = (opA == Op::NoTrans ? Op::ConjTrans : Op::NoTrans);

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
        for (int64_t j = 0; j < C.nt(); ++j) {
            for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
                if (C.tileIsLocal(i, j)) {
                    m_array[index] = C(i, j).mb();
                    n_array[index] = C(i, j).nb();
                    k_array[index] = A(i, 0).nb();  // should be all same

                    assert(A(i, 0).mb() == m_array[index]);
                    assert(A(j, 0).mb() == n_array[index]);
                    assert(A(j, 0).nb() == k_array[index]);

                    a_array[index] = A(i, 0).data();
                    b_array[index] = A(j, 0).data();
                    c_array[index] = C(i, j).data();

                    lda_array[index] = A(i, 0).stride();
                    ldb_array[index] = A(j, 0).stride();
                    ldc_array[index] = C(i, j).stride();

                    ++index;
                }
            }
        }

        if (C.op() != Op::NoTrans) {
            // swap A <=> B; swap m <=> n
            swap(opA_array, opB_array);
            swap(a_array,   b_array  );
            swap(lda_array, ldb_array);
            swap(m_array,   n_array  );
        }

        {
            trace::Block trace_block("cblas_gemm_batch");
            // mkl_set_num_threads_local(...);
            cblas_gemm_batch(CblasColMajor,
                             opA_array.data(), opB_array.data(),
                             m_array.data(), n_array.data(), k_array.data(),
                             alpha_array.data(),
                             a_array.data(), lda_array.data(),
                             b_array.data(), ldb_array.data(),
                             beta_array.data(),
                             c_array.data(), ldc_array.data(),
                             batch_count, group_size.data());
            // mkl_set_num_threads_local(1);
        }

        for (int64_t j = 0; j < C.nt(); ++j) {
            for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
                if (C.tileIsLocal(i, j)) {
                    // todo: should tileRelease()?
                    A.tileTick(i, 0);
                    A.tileTick(j, 0);
                }
            }
        }
    }

    if (err)
        throw std::exception();
#else
    slate_not_implemented(
        "slate::Target::HostBatch needs Intel MKL.");
#endif
}

//------------------------------------------------------------------------------
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// GPU device batched cuBLAS implementation.
/// Assumes A is NoTrans or ConjTrans; C is Lower, NoTrans or Upper, ConjTrans.
/// @ingroup herk_internal
///
template <typename scalar_t>
void herk(internal::TargetType<Target::Devices>,
          blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
          int priority, int queue_index, Layout layout, Options const& opts)
{
    int err = 0;
    using std::swap;
    using real_t = blas::real_type<scalar_t>;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );
    assert(C.num_devices() > 0);

    // if single tile, avoid creating tasks for all devices
    #pragma omp taskgroup
    if (C.nt() == 1) {
        if (C.tileIsLocal(0, 0)) {
            #pragma omp task slate_omp_default_none \
                shared( A, C, err ) priority( priority ) \
                firstprivate(layout, queue_index, alpha, beta, tile_release_strategy)
            {
                int device = C.tileDevice(0, 0);
                A.tileGetForReading(0, 0, device, LayoutConvert(layout));
                C.tileGetForWriting(0, 0, device, LayoutConvert(layout));

                blas::Queue* queue = C.compute_queue(device, queue_index);

                auto A00 = A(0, 0, device);
                auto C00 = C(0, 0, device);

                blas::herk(
                    layout, C00.uploPhysical(), A00.op(),
                    C00.nb(), A00.nb(),
                    alpha, A00.data(), A00.stride(),
                    beta,  C00.data(), C00.stride(), *queue);

                queue->sync();

                if (tile_release_strategy == TileReleaseStrategy::Internal
                    || tile_release_strategy == TileReleaseStrategy::All) {

                    A.tileRelease(0, 0, device);
                    A.tileTick(0, 0);
                    A.tileTick(0, 0);
                }
            }
        }
    }
    else {
        // off-diagonal tiles by batch gemm on device
        // diagonal tiles by herk on device
        for (int device = 0; device < C.num_devices(); ++device) {
            #pragma omp task slate_omp_default_none \
                shared( A, C, err ) priority( priority ) \
                firstprivate(layout, queue_index, device, alpha, beta, tile_release_strategy)
            {
                try {
                    // if op(C) is NoTrans, invert opA, opB if possible
                    Op opA = A.op();
                    if (C.op() != Op::NoTrans) {
                        if (A.op() == Op::NoTrans)
                            opA = C.op();
                        else if (A.op() == C.op() || C.is_real) {
                            // A and C are both Trans or both ConjTrans;
                            // Trans == ConjTrans if real
                            opA = Op::NoTrans;
                        }
                        else
                            throw std::exception();
                    }

                    Op opB = (opA == Op::NoTrans ? Op::ConjTrans : Op::NoTrans);

                    std::set<ij_tuple> A_tiles_gemm, C_tiles_gemm;
                    std::set<ij_tuple> A_tiles_herk, C_tiles_herk;
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        for (int64_t i = j; i < C.mt(); ++i) {  // lower
                            if (C.tileIsLocal(i, j)
                                && device == C.tileDevice(i, j)) {
                                if (i == j) {
                                    A_tiles_herk.insert({j, 0});
                                    C_tiles_herk.insert({j, j});
                                }
                                else {
                                    A_tiles_gemm.insert({i, 0});
                                    A_tiles_gemm.insert({j, 0});
                                    C_tiles_gemm.insert({i, j});
                                }
                            }
                        }
                    }

                    #pragma omp taskgroup
                    {
                        #pragma omp task slate_omp_default_none \
                            shared( A, A_tiles_gemm ) \
                            firstprivate(device, layout)
                        {
                            A.tileGetForReading(A_tiles_gemm, device, LayoutConvert(layout));
                        }
                        #pragma omp task slate_omp_default_none \
                            shared( C, C_tiles_gemm ) \
                            firstprivate(device, layout)
                        {
                            C.tileGetForWriting(C_tiles_gemm, device, LayoutConvert(layout));
                        }
                    }

                    int64_t batch_size_gemm = C_tiles_gemm.size();

                    // interior
                    std::vector<scalar_t*> a_array_gemm00;
                    std::vector<scalar_t*> b_array_gemm00;
                    std::vector<scalar_t*> c_array_gemm00;
                    a_array_gemm00.reserve( batch_size_gemm );
                    b_array_gemm00.reserve( batch_size_gemm );
                    c_array_gemm00.reserve( batch_size_gemm );

                    int64_t lda00 = 0;
                    int64_t ldb00 = 0;
                    int64_t ldc00 = 0;
                    int64_t mb00 = C.tileMb(0);
                    int64_t nb00 = C.tileNb(0);
                    int64_t kb   = A.tileNb(0);
                    for (int64_t j = 0; j < C.nt()-1; ++j) {
                        // strictly lower
                        for (int64_t i = j+1; i < C.mt()-1; ++i) {
                            if (C.tileIsLocal(i, j)) {
                                if (device == C.tileDevice(i, j)) {
                                    a_array_gemm00.push_back( A(i, 0, device).data() );
                                    b_array_gemm00.push_back( A(j, 0, device).data() );
                                    c_array_gemm00.push_back( C(i, j, device).data() );
                                    lda00 = A(i, 0, device).stride();
                                    ldb00 = A(j, 0, device).stride();
                                    ldc00 = C(i, j, device).stride();
                                }
                            }
                        }
                    }

                    // bottom row
                    std::vector<scalar_t*> a_array_gemm10;
                    std::vector<scalar_t*> b_array_gemm10;
                    std::vector<scalar_t*> c_array_gemm10;
                    a_array_gemm10.reserve( batch_size_gemm );
                    b_array_gemm10.reserve( batch_size_gemm );
                    c_array_gemm10.reserve( batch_size_gemm );

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
                                    a_array_gemm10.push_back( A(i, 0, device).data() );
                                    b_array_gemm10.push_back( A(j, 0, device).data() );
                                    c_array_gemm10.push_back( C(i, j, device).data() );
                                    lda10 = A(i, 0, device).stride();
                                    ldb10 = A(j, 0, device).stride();
                                    ldc10 = C(i, j, device).stride();
                                }
                            }
                        }
                    }

                    if (C.op() != Op::NoTrans) {
                        // swap A <=> B; swap m <=> n
                        swap(opA, opB);
                        swap(a_array_gemm00, b_array_gemm00);
                        swap(a_array_gemm10, b_array_gemm10);
                        swap(lda00, ldb00);
                        swap(lda10, ldb10);
                        swap(mb00, nb00);
                        swap(mb10, nb10);
                    }

                    std::vector<Op> opA_(1, opA);
                    std::vector<Op> opB_(1, opB);
                    std::vector<int64_t> k(1,  kb);
                    std::vector<int64_t> info;

                    blas::Queue* queue = C.compute_queue(device, queue_index);

                    {
                        trace::Block trace_block("blas::batch::gemm");

                        std::vector<scalar_t> alpha_(1, scalar_t(alpha));
                        std::vector<scalar_t> beta_ (1, scalar_t(beta));

                        if (c_array_gemm00.size() > 0) {
                            std::vector<int64_t>    m(1,  mb00);
                            std::vector<int64_t>    n(1,  nb00);
                            std::vector<int64_t> ldda(1, lda00);
                            std::vector<int64_t> lddb(1, ldb00);
                            std::vector<int64_t> lddc(1, ldc00);
                            blas::batch::gemm(
                                layout, opA_, opB_,
                                m, n, k,
                                alpha_, a_array_gemm00, ldda,
                                        b_array_gemm00, lddb,
                                beta_,  c_array_gemm00, lddc,
                                c_array_gemm00.size(), info, *queue);
                        }

                        if (c_array_gemm10.size() > 0) {
                            std::vector<int64_t>    m(1,  mb10);
                            std::vector<int64_t>    n(1,  nb10);
                            std::vector<int64_t> ldda(1, lda10);
                            std::vector<int64_t> lddb(1, ldb10);
                            std::vector<int64_t> lddc(1, ldc10);
                            blas::batch::gemm(
                                layout, opA_, opB_,
                                m, n, k,
                                alpha_, a_array_gemm10, ldda,
                                        b_array_gemm10, lddb,
                                beta_,  c_array_gemm10, lddc,
                                c_array_gemm10.size(), info, *queue);
                        }
                    }

                    #pragma omp taskgroup
                    {
                        #pragma omp task slate_omp_default_none \
                            shared( A, A_tiles_herk ) \
                            firstprivate(device, layout)
                        {
                            A.tileGetForReading(A_tiles_herk, device, LayoutConvert(layout));
                        }
                        #pragma omp task slate_omp_default_none \
                            shared( C, C_tiles_herk ) \
                            firstprivate(device, layout)
                        {
                            C.tileGetForWriting(C_tiles_herk, device, LayoutConvert(layout));
                        }
                    }

                    int64_t batch_size_herk = C_tiles_herk.size();

                    // diagonal
                    std::vector<scalar_t*> a_array_herk0;
                    std::vector<scalar_t*> c_array_herk0;
                    a_array_herk0.reserve( batch_size_herk );
                    c_array_herk0.reserve( batch_size_herk );

                    int64_t lda_herk_0 = 0;
                    int64_t ldc_herk_0 = 0;
                    int64_t nb_herk_0 = C.tileNb(0);
                    for (int64_t j = 0; j < C.nt()-1; ++j) {
                        if (C.tileIsLocal(j, j)
                            && device == C.tileDevice(j, j))
                        {
                            a_array_herk0.push_back( A(j, 0, device).data() );
                            c_array_herk0.push_back( C(j, j, device).data() );
                            lda_herk_0 = A(j, 0, device).stride();
                            ldc_herk_0 = C(j, j, device).stride();
                        }
                    }

                    // bottom-right corner
                    // todo: replace batch herk with plain herk
                    std::vector<scalar_t*> a_array_herk1;
                    std::vector<scalar_t*> c_array_herk1;

                    int64_t lda_herk_1 = 0;
                    int64_t ldc_herk_1 = 0;
                    int64_t nb_herk_1 = C.tileNb(C.nt()-1);
                    {
                        int i = C.mt()-1;
                        int j = C.nt()-1;
                        if (C.tileIsLocal(i, j)) {
                            if (device == C.tileDevice(i, j)) {
                                a_array_herk1.push_back( A(j, 0, device).data() );
                                c_array_herk1.push_back( C(j, j, device).data() );
                                lda_herk_1 = A(j, 0, device).stride();
                                ldc_herk_1 = C(j, j, device).stride();
                            }
                        }
                    }

                    {
                        trace::Block trace_block("blas::batch::herk");

                        std::vector<Uplo> uplo(1, C.uploPhysical());
                        std::vector<real_t> alpha_(1, alpha);
                        std::vector<real_t> beta_ (1, beta);

                        if (c_array_herk0.size() > 0) {
                            std::vector<int64_t>    n(1,  nb_herk_0);
                            std::vector<int64_t> ldda(1, lda_herk_0);
                            std::vector<int64_t> lddc(1, ldc_herk_0);
                            blas::batch::herk(
                                layout, uplo, opA_,
                                n, k,
                                alpha_, a_array_herk0, ldda,
                                beta_,  c_array_herk0, lddc,
                                c_array_herk0.size(), info, *queue);
                        }

                        if (c_array_herk1.size() > 0) {
                            std::vector<int64_t>    n(1,  nb_herk_1);
                            std::vector<int64_t> ldda(1, lda_herk_1);
                            std::vector<int64_t> lddc(1, ldc_herk_1);
                            blas::batch::herk(
                                layout, uplo, opA_,
                                n, k,
                                alpha_, a_array_herk1, ldda,
                                beta_,  c_array_herk1, lddc,
                                c_array_herk1.size(), info, *queue);
                        }
                    }

                    queue->sync();

                    if (tile_release_strategy == TileReleaseStrategy::Internal
                        || tile_release_strategy == TileReleaseStrategy::All) {
                        // both off-diagonal batch gemm and diagonal herks are done
                        for (int64_t j = 0; j < C.nt(); ++j) {
                            for (int64_t i = j; i < C.mt(); ++i) {  // lower
                                if (C.tileIsLocal(i, j)) {
                                    if (device == C.tileDevice(i, j)) {
                                        // erase tmp local and remote device tiles;
                                        A.tileRelease(i, 0, device);
                                        A.tileRelease(j, 0, device);
                                        // decrement life for remote tiles
                                        // todo: should tileRelease()?
                                        A.tileTick(i, 0);
                                        A.tileTick(j, 0);
                                    }
                                }
                            }
                        }
                    }
                }
                catch (std::exception& e) {
                    err = __LINE__;
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
void herk<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
    float beta,  HermitianMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
    float beta,  HermitianMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
    float beta,  HermitianMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
    float beta,  HermitianMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

// ----------------------------------------
template
void herk<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

// ----------------------------------------
template
void herk< Target::HostTask, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk< Target::HostNest, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk< Target::HostBatch, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk< Target::Devices, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

// ----------------------------------------
template
void herk< Target::HostTask, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk< Target::HostNest, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk< Target::HostBatch, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void herk< Target::Devices, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

} // namespace internal
} // namespace slate
