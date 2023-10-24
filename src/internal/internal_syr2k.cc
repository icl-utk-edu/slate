// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Symmetric rank-2k update of single block column (i.e., k = nb).
/// Dispatches to target implementations.
/// C is Lower, NoTrans or Upper, Trans/ConjTrans.
/// In complex case, A, B, and C cannot be ConjTrans.
/// Requires op(A) and op(B) to be the same, either both NoTrans, or both Trans.
/// @ingroup syr2k_internal
///
template <Target target, typename scalar_t>
void syr2k(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>&& C,
           int priority, int queue_index, Layout layout, Options const& opts)
{
    if (! ((C.uplo() == Uplo::Lower)
           &&
           (C.is_real || (C.op() != Op::ConjTrans &&
                          A.op() != Op::ConjTrans))
           &&
           (A.op() == B.op())))
        throw std::exception();

    syr2k(internal::TargetType<target>(),
          alpha, A,
                 B,
          beta,  C,
          priority, queue_index, layout, opts);
}

//------------------------------------------------------------------------------
/// Symmetric rank-2k update of single block column (i.e., k = nb).
/// Host OpenMP task implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syr2k_internal
///
template <typename scalar_t>
void syr2k(internal::TargetType<Target::HostTask>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>& C,
           int priority, int queue_index, Layout layout, Options const& opts)
{
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::syr2k()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    assert(layout == Layout::ColMajor);

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    int err = 0;
    #pragma omp taskgroup
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j; i < C.mt(); ++i) {  // lower
            if (C.tileIsLocal(i, j)) {
                if (i == j) {
                    #pragma omp task slate_omp_default_none \
                        shared( A, B, C, err ) \
                        firstprivate(j, layout, alpha, beta, call_tile_tick) \
                        priority(priority)
                    {
                        try {
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            B.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(j, j, LayoutConvert(layout));
                            tile::syr2k(
                                alpha, A(j, 0), B(j, 0),
                                beta,  C(j, j) );
                            if (call_tile_tick) {
                                // todo: should tileRelease()?
                                A.tileTick( j, 0 );
                                B.tileTick( j, 0 );
                            }
                        }
                        catch (std::exception& e) {
                            err = __LINE__;
                        }
                    }
                }
                else {
                    #pragma omp task slate_omp_default_none \
                        shared( A, B, C, err ) \
                        firstprivate(i, j, layout, alpha, beta, call_tile_tick) \
                        priority(priority)
                    {
                        try {
                            const scalar_t one = 1.0;

                            A.tileGetForReading(i, 0, LayoutConvert(layout));
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            B.tileGetForReading(i, 0, LayoutConvert(layout));
                            B.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(i, j, LayoutConvert(layout));
                            auto Aj0 = A(j, 0);
                            auto Bj0 = B(j, 0);
                            tile::gemm(
                                alpha, A(i, 0), transpose( Bj0 ),
                                beta,  C(i, j) );
                            tile::gemm(
                                alpha, B(i, 0), transpose( Aj0 ),
                                one,   C(i, j) );
                            if (call_tile_tick) {
                                // todo: should tileRelease()?
                                A.tileTick( i, 0 );
                                A.tileTick( j, 0 );
                                B.tileTick( i, 0 );
                                B.tileTick( j, 0 );
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
/// Symmetric rank-2k update of single block column (i.e., k = nb).
/// Host nested OpenMP implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syr2k_internal
///
template <typename scalar_t>
void syr2k(internal::TargetType<Target::HostNest>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>& C,
           int priority, int queue_index, Layout layout, Options const& opts)
{
#if defined(SLATE_HAVE_OMPTARGET) || defined(SLATE_SKIP_HOSTNEST)
    // SYCL/OMP-target-offload can't process this section
    slate_not_implemented("Target::HostNest isn't supported in this configuration.");
#else
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::syr2k()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    assert(layout == Layout::ColMajor);

    int err = 0;
    #pragma omp taskgroup
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task slate_omp_default_none \
                shared( A, B, C, err ) \
                firstprivate(j, layout, alpha, beta)
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    B.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    tile::syr2k(
                        alpha, A(j, 0), B(j, 0),
                        beta,  C(j, j) );
                    // todo: should tileRelease()?
                    A.tileTick(j, 0);
                    B.tileTick(j, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }

    int64_t C_mt = C.mt();
    int64_t C_nt = C.nt();

//  #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...) default(none)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) slate_omp_default_none \
        shared(A, B, C, err) firstprivate(C_mt, C_nt, layout, alpha, beta)
    for (int64_t j = 0; j < C_nt; ++j) {
        for (int64_t i = 0; i < C_mt; ++i) {  // full
            if (i >= j+1) {                     // strictly lower
                if (C.tileIsLocal(i, j)) {
                    try {
                        const scalar_t one = 1.0;

                        A.tileGetForReading(i, 0, LayoutConvert(layout));
                        B.tileGetForReading(j, 0, LayoutConvert(layout));
                        C.tileGetForWriting(i, j, LayoutConvert(layout));
                        auto Aj0 = A(j, 0);
                        auto Bj0 = B(j, 0);
                        tile::gemm(
                            alpha, A(i, 0), transpose( Bj0 ),
                            beta,  C(i, j) );
                        tile::gemm(
                            alpha, B(i, 0), transpose( Aj0 ),
                            one,   C(i, j) );
                        // todo: should tileRelease()?
                        A.tileTick(i, 0);
                        A.tileTick(j, 0);
                        B.tileTick(i, 0);
                        B.tileTick(j, 0);
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
#endif // omit if SLATE_HAVE_OMPTARGET
}

//------------------------------------------------------------------------------
/// Symmetric rank-2k update of single block column (i.e., k = nb).
/// Host batched implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syr2k_internal
///
template <typename scalar_t>
void syr2k(internal::TargetType<Target::HostBatch>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>& C,
           int priority, int queue_index, Layout layout, Options const& opts)
{
#ifdef BLAS_HAVE_MKL
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::syr2k() to
    //       take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    assert(layout == Layout::ColMajor);

    // diagonal tiles by syr2k on host
    int err = 0;
    #pragma omp taskgroup
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task slate_omp_default_none \
                shared( A, B, C, err ) \
                firstprivate(j, layout, alpha, beta)
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    B.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    tile::syr2k(
                        alpha, A(j, 0), B(j, 0),
                        beta,  C(j, j) );
                    // todo: should tileRelease()?
                    A.tileTick(j, 0);
                    B.tileTick(j, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }

    // load off-diagonal tiles to host, if not there
    // also count tiles
    int batch_count = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
            if (C.tileIsLocal(i, j)) {
                // todo: omp task?
                A.tileGetForReading(i, 0, LayoutConvert(layout));
                B.tileGetForReading(j, 0, LayoutConvert(layout));
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

        Op opB = (opA == Op::NoTrans ? Op::Trans : Op::NoTrans);

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
        std::vector<const scalar_t*> ai_array(batch_count);
        std::vector<const scalar_t*> aj_array(batch_count);
        std::vector<const scalar_t*> bi_array(batch_count);
        std::vector<const scalar_t*> bj_array(batch_count);
        std::vector<scalar_t*> c_array(batch_count);
        std::vector<int> ldai_array(batch_count);
        std::vector<int> ldaj_array(batch_count);
        std::vector<int> ldbi_array(batch_count);
        std::vector<int> ldbj_array(batch_count);
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

                    ai_array[index] = A(i, 0).data();
                    aj_array[index] = A(j, 0).data();
                    bi_array[index] = B(i, 0).data();
                    bj_array[index] = B(j, 0).data();
                    c_array[index] = C(i, j).data();

                    ldai_array[index] = A(i, 0).stride();
                    ldaj_array[index] = A(j, 0).stride();
                    ldbi_array[index] = B(i, 0).stride();
                    ldbj_array[index] = B(j, 0).stride();
                    ldc_array[index] = C(i, j).stride();

                    ++index;
                }
            }
        }

        if (C.op() != Op::NoTrans) {
            // swap A <=> B; swap m <=> n
            swap(opA_array,  opB_array );
            swap(ai_array,   bj_array  );
            swap(aj_array,   bi_array  );
            swap(ldai_array, ldbj_array);
            swap(ldaj_array, ldbi_array);
            swap(m_array,    n_array   );
        }

        {
            trace::Block trace_block("cblas_gemm_batch");
                const scalar_t one = 1.0;

            // mkl_set_num_threads_local(...);
            cblas_gemm_batch(CblasColMajor,
                             opA_array.data(), opB_array.data(),
                             m_array.data(), n_array.data(), k_array.data(),
                             alpha_array.data(),
                             ai_array.data(), ldai_array.data(),
                             bj_array.data(), ldbj_array.data(),
                             beta_array.data(),
                             c_array.data(), ldc_array.data(),
                             batch_count, group_size.data());

            // ai => bi, bj => aj, set beta = 1
            std::fill( beta_array.begin(), beta_array.end(), one );
            cblas_gemm_batch(CblasColMajor,
                             opA_array.data(), opB_array.data(),
                             m_array.data(), n_array.data(), k_array.data(),
                             alpha_array.data(),
                             bi_array.data(), ldbi_array.data(),
                             aj_array.data(), ldaj_array.data(),
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
                    B.tileTick(i, 0);
                    B.tileTick(j, 0);
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
/// Symmetric rank-2k update of single block column (i.e., k = nb).
/// GPU device batched cuBLAS implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syr2k_internal
///
template <typename scalar_t>
void syr2k(internal::TargetType<Target::Devices>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  SymmetricMatrix<scalar_t>& C,
           int priority, int queue_index, Layout layout, Options const& opts)
{
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    bool call_tile_tick = tile_release_strategy == TileReleaseStrategy::Internal
                          || tile_release_strategy == TileReleaseStrategy::All;

    assert(C.num_devices() > 0);

    int err = 0;

    // if single tile, avoid creating tasks for all devices
    #pragma omp taskgroup
    if (C.nt() == 1) {
        if (C.tileIsLocal(0, 0)) {
            #pragma omp task slate_omp_default_none \
                shared( A, B, C, err ) \
                firstprivate(layout, alpha, beta, queue_index, call_tile_tick) \
                priority(priority)
            {
                int device = C.tileDevice(0, 0);
                A.tileGetForReading(0, 0, device, LayoutConvert(layout));
                B.tileGetForReading(0, 0, device, LayoutConvert(layout));
                C.tileGetForWriting(0, 0, device, LayoutConvert(layout));

                blas::Queue* queue = C.compute_queue(device, queue_index);

                auto A00 = A(0, 0, device);
                auto B00 = B(0, 0, device);
                auto C00 = C(0, 0, device);

                blas::syr2k(
                    layout, C00.uploPhysical(), A00.op(),
                    C00.nb(), A00.nb(),
                    alpha, A00.data(), A00.stride(),
                           B00.data(), B00.stride(),
                    beta,  C00.data(), C00.stride(), *queue);

                queue->sync();

                if (call_tile_tick) {
                    A.tileRelease(0, 0, device);
                    B.tileRelease(0, 0, device);
                    A.tileTick(0, 0);
                    A.tileTick(0, 0);
                    B.tileTick(0, 0);
                    B.tileTick(0, 0);
                }
            }
        }
    }
    else {
        // off-diagonal tiles by batch gemm on device
        // diagonal tiles by BLAS++ syr2k on device
        for (int device = 0; device < C.num_devices(); ++device) {
            #pragma omp task slate_omp_default_none \
                shared( A, B, C, err ) \
                firstprivate(device, layout, alpha, beta, queue_index, call_tile_tick) \
                priority(priority)
            {
                try {
                    const scalar_t one = 1.0;

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

                    Op opB = (opA == Op::NoTrans ? Op::Trans : Op::NoTrans);

                    std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        for (int64_t i = j; i < C.mt(); ++i) {  // lower
                            if (C.tileIsLocal(i, j)
                                && device == C.tileDevice(i, j)) {

                                A_tiles_set.insert({j, 0});
                                B_tiles_set.insert({j, 0});
                                C_tiles_set.insert({i, j});
                                if (i == j) {
                                }
                                else {
                                    A_tiles_set.insert({i, 0});
                                    B_tiles_set.insert({i, 0});
                                }
                            }
                        }
                    }

                    #pragma omp taskgroup
                    {
                        #pragma omp task slate_omp_default_none \
                            shared( A, A_tiles_set ) \
                            firstprivate( device, layout )
                        {
                            A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                        }
                        #pragma omp task slate_omp_default_none \
                            shared( B, B_tiles_set ) \
                            firstprivate( device, layout )
                        {
                            B.tileGetForReading(B_tiles_set, device, LayoutConvert(layout));
                        }
                        #pragma omp task slate_omp_default_none \
                            shared( C, C_tiles_set ) \
                            firstprivate( device, layout )
                        {
                            C.tileGetForWriting(C_tiles_set, device, LayoutConvert(layout));
                        }
                    }

                    int64_t batch_size = C_tiles_set.size();

                    scalar_t** a_array_host = C.array_host(device, queue_index);
                    scalar_t** b_array_host = a_array_host + batch_size;
                    scalar_t** c_array_host = b_array_host + batch_size;

                    // There are only 3 batch arrays
                    std::vector<scalar_t*> t_array_vect( 2*batch_size );
                    scalar_t** at_array_host = t_array_vect.data();
                    scalar_t** bt_array_host = at_array_host + batch_size;

                    // Use transposed A and B to broadcast correctly
                    auto AT = transpose(A);
                    auto BT = transpose(B);

                    // C comes first since we do computation for a local C
                    auto group_params = device_regions_build<true, 5, scalar_t>(
                                                            {C, A, AT, BT, B},
                                                            {c_array_host, a_array_host, at_array_host, b_array_host, bt_array_host},
                                                            device );


                    if (C.op() != Op::NoTrans) {
                        swap(opA, opB);
                    }

                    {
                        trace::Block trace_block("blas::batch::her2k");

                        std::vector<Op> opA_(1, opA);
                        std::vector<Op> opB_(1, opB);
                        std::vector<int64_t> k(1, A.tileNb(0));
                        std::vector<int64_t> info;

                        std::vector<scalar_t> alpha_(1, alpha);
                        std::vector<scalar_t> beta_(1, beta);
                        std::vector<scalar_t> one_( 1, one );
                        std::vector<Uplo> uplo(1, C.uploPhysical());

                        blas::Queue* queue = C.compute_queue(device, queue_index);

                        for (size_t g = 0; g < group_params.size(); ++g) {

                            int64_t group_count = group_params[ g ].count;

                            std::vector<int64_t>    n(1, group_params[ g ].nb);
                            std::vector<int64_t> ldda(1, group_params[ g ].ld[1]);
                            std::vector<int64_t> lddb(1, group_params[ g ].ld[3]);
                            std::vector<int64_t> lddc(1, group_params[ g ].ld[0]);
                            std::vector<scalar_t*> a_array(a_array_host, a_array_host+group_count);
                            std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);
                            std::vector<scalar_t*> c_array(c_array_host, c_array_host+group_count);

                            if (group_params[ g ].is_diagonal) {
                                blas::batch::syr2k(
                                    layout, uplo, opA_,
                                    n, k,
                                    alpha_, a_array, ldda,
                                                 b_array, lddb,
                                    beta_,      c_array, lddc,
                                    group_count, info, *queue);
                            }
                            else {
                                std::vector<int64_t>    m(1, group_params[ g ].mb);
                                std::vector<int64_t> lddat(1, group_params[ g ].ld[2]);
                                std::vector<int64_t> lddbt(1, group_params[ g ].ld[4]);
                                std::vector<scalar_t*> at_array(at_array_host, at_array_host+group_count);
                                std::vector<scalar_t*> bt_array(bt_array_host, bt_array_host+group_count);

                                if (C.op() != Op::NoTrans) {
                                    swap(m, n);
                                    swap(a_array, b_array);
                                    swap(at_array, bt_array);
                                    swap(ldda, lddb);
                                    swap(lddat, lddbt);
                                }

                                blas::batch::gemm(
                                    layout, opA_, opB_,
                                    m, n, k,
                                    alpha_, a_array, ldda,
                                             b_array, lddb,
                                    beta_,  c_array, lddc,
                                    group_count, info, *queue);

                                blas::batch::gemm(
                                    layout, opA_, opB_,
                                    m, n, k,
                                    alpha_, bt_array, lddbt,
                                                  at_array, lddat,
                                    one_,       c_array, lddc,
                                    group_count, info, *queue);
                            }
                            a_array_host += group_count;
                            at_array_host += group_count;
                            b_array_host += group_count;
                            bt_array_host += group_count;
                            c_array_host += group_count;
                        }

                        queue->sync();
                    }

                    if (call_tile_tick) {
                        for (int64_t j = 0; j < C.nt(); ++j) {
                            for (int64_t i = j; i < C.mt(); ++i) {  // lower
                                if (C.tileIsLocal(i, j)
                                    && device == C.tileDevice(i, j))
                                {
                                    // erase tmp local and remote device tiles;
                                    A.tileRelease(i, 0, device);
                                    A.tileRelease(j, 0, device);
                                    B.tileRelease(i, 0, device);
                                    B.tileRelease(j, 0, device);
                                    // decrement life for remote tiles
                                    // todo: should tileRelease()?
                                    A.tileTick(i, 0);
                                    A.tileTick(j, 0);
                                    B.tileTick(i, 0);
                                    B.tileTick(j, 0);
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
        throw std::exception();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void syr2k<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  SymmetricMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  SymmetricMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  SymmetricMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  SymmetricMatrix<float>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

// ----------------------------------------
template
void syr2k<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
                           Matrix<double>&& B,
    double beta,  SymmetricMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  SymmetricMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  SymmetricMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  SymmetricMatrix<double>&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

// ----------------------------------------
template
void syr2k< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

// ----------------------------------------
template
void syr2k< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

template
void syr2k< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout, Options const& opts);

} // namespace internal
} // namespace slate
