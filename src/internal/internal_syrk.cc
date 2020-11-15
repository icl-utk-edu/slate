// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Dispatches to target implementations.
/// C is Lower, NoTrans or Upper, Trans/ConjTrans.
/// In complex case, A and C cannot be ConjTrans.
/// @ingroup syrk_internal
///
template <Target target, typename scalar_t>
void syrk(scalar_t alpha,          Matrix<scalar_t>&& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>&& C,
          int priority)
{
    if (! ((C.uplo() == Uplo::Lower)
           &&
           (C.is_real || (C.op() != Op::ConjTrans &&
                          A.op() != Op::ConjTrans))))
        throw std::exception();

    syrk(internal::TargetType<target>(),
         alpha, A,
         beta,  C,
         priority);
}

//------------------------------------------------------------------------------
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host OpenMP task implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syrk_internal
///
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostTask>,
          scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    // CPU assumes column major
    // todo: relax this assumption, by updating Tile_blas.hh::syrk()
    //       to operate in row major
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    // Lower, NoTrans
    int err = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j; i < C.mt(); ++i) {  // lower
            if (C.tileIsLocal(i, j)) {
                if (i == j) {
                    #pragma omp task shared(A, C, err) priority(priority)
                    {
                        try {
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(j, j, LayoutConvert(layout));
                            syrk(alpha, A(j, 0),
                                 beta,  C(j, j));
                            // todo: should tileRelease()?
                            A.tileTick(j, 0);
                            // todo: why the second tick?
                            A.tileTick(j, 0);
                        }
                        catch (std::exception& e) {
                            err = __LINE__;
                        }
                    }
                }
                else {
                    #pragma omp task shared(A, C, err) priority(priority)
                    {
                        try {
                            A.tileGetForReading(i, 0, LayoutConvert(layout));
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(i, j, LayoutConvert(layout));
                            auto Aj0 = A(j, 0);
                            gemm(alpha, A(i, 0),
                                        transpose(Aj0),
                                 beta,  C(i, j));
                            // todo: should tileRelease()?
                            A.tileTick(i, 0);
                            A.tileTick(j, 0);
                        }
                        catch (std::exception& e ) {
                            err = __LINE__;
                        }
                    }
                }
            }
        }
    }

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host nested OpenMP implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syrk_internal
///
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostNest>,
          scalar_t alpha,          Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::syrk()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    // Lower, NoTrans
    int err = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C, err)
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    syrk(alpha, A(j, 0),
                         beta,  C(j, j));
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

    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = 0; i < C.mt(); ++i) {  // full
            if (i >= j+1) {                     // strictly lower
                if (C.tileIsLocal(i, j)) {
                    try {
                        A.tileGetForReading(i, 0, LayoutConvert(layout));
                        A.tileGetForReading(j, 0, LayoutConvert(layout));
                        C.tileGetForWriting(i, j, LayoutConvert(layout));
                        auto Aj0 = A(j, 0);
                        gemm(alpha, A(i, 0),
                                    transpose(Aj0),
                             beta,  C(i, j));
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

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host batched implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syrk_internal
///
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostBatch>,
          scalar_t alpha,          Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    // CPU assumes column major
    // todo: relax this assumption, by allowing Tile_blas.hh::syrk()
    //       to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    // diagonal tiles by syrk on host
    int err = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C, err)
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    syrk(alpha, A(j, 0),
                         beta,  C(j, j));
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

            #ifdef SLATE_WITH_MKL
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
            #else
                slate_not_implemented(
                    "slate::Target::HostBatch needs Intel MKL.");
            #endif
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

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// GPU device batched cuBLAS implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
/// @ingroup syrk_internal
///
template <typename scalar_t>
void syrk(internal::TargetType<Target::Devices>,
          scalar_t alpha,          Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    int err = 0;
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // assumes column major for now
    // todo: relax this assumption,
    //       by allowing Tile_blas.hh::syrk() to take layout param
    //       look at internal::gemm()
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    assert(C.num_devices() > 0);

    // if single tile, avoid creating tasks for all devices
    if (C.nt() == 1) {
        if (C.tileIsLocal(0, 0)) {
            #pragma omp task shared(A, C, err) priority(priority)
            {
                auto device = C.tileDevice(0, 0);

                A.tileGetForReading(0, 0, device, LayoutConvert(layout));
                C.tileGetForWriting(0, 0, device, LayoutConvert(layout));

                blas::Queue* queue = C.queue(device);

                auto A00 = A(0, 0, device);
                auto C00 = C(0, 0, device);

                blas::syrk(
                    layout, C00.uploPhysical(), A00.op(),
                    C00.nb(), A00.nb(),
                    alpha, A00.data(), A00.stride(),
                    beta,  C00.data(), C00.stride(), *queue);

                queue->sync();

                A.tileRelease(0, 0, device);
                A.tileTick(0, 0);
                A.tileTick(0, 0);
            }
        }
    }
    else {
        // off-diagonal tiles by batch gemm on device
        // diagonal tiles by herk on device
        for (int device = 0; device < C.num_devices(); ++device) {
            #pragma omp task shared(A, C, err) priority(priority)
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

                    Op opB = (opA == Op::NoTrans ? Op::Trans : Op::NoTrans);

                    std::set<ij_tuple> A_tiles_gemm, C_tiles_gemm;
                    std::set<ij_tuple> A_tiles_syrk, C_tiles_syrk;
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        for (int64_t i = j; i < C.mt(); ++i) {
                            if (C.tileIsLocal(i, j)) {
                                if (device == C.tileDevice(i, j)) {
                                    if (i == j) {
                                        A_tiles_syrk.insert({j, 0});
                                        C_tiles_syrk.insert({j, j});
                                    }
                                    else {
                                        A_tiles_gemm.insert({i, 0});
                                        A_tiles_gemm.insert({j, 0});
                                        C_tiles_gemm.insert({i, j});
                                    }
                                }
                            }
                        }
                    }

                    #pragma omp task default(shared)
                    {
                        A.tileGetForReading(
                            A_tiles_gemm, device, LayoutConvert(layout));
                    }
                    #pragma omp task default(shared)
                    {
                        C.tileGetForWriting(
                            C_tiles_gemm, device, LayoutConvert(layout));
                    }
                    #pragma omp taskwait

                    int64_t batch_size_gemm = C_tiles_gemm.size();

                    std::vector<scalar_t*> a_array_host_gemm_0(batch_size_gemm);
                    std::vector<scalar_t*> b_array_host_gemm_0(batch_size_gemm);
                    std::vector<scalar_t*> c_array_host_gemm_0(batch_size_gemm);

                    int64_t batch_count_gemm_0 = 0;
                    int64_t lda_gemm_0 = 0;
                    int64_t ldb_gemm_0 = 0;
                    int64_t ldc_gemm_0 = 0;

                    int64_t mb0 = C.tileMb(0);
                    int64_t nb0 = C.tileNb(0);
                    int64_t kb  = A.tileNb(0);

                    for (int64_t j = 0; j < C.nt()-1; ++j) {
                        for (int64_t i = j+1; i < C.mt()-1; ++i) {
                            if (C.tileIsLocal(i, j)) {
                                if (device == C.tileDevice(i, j)) {
                                    a_array_host_gemm_0[batch_count_gemm_0]
                                        = A(i, 0, device).data();
                                    b_array_host_gemm_0[batch_count_gemm_0]
                                        = A(j, 0, device).data();
                                    c_array_host_gemm_0[batch_count_gemm_0]
                                        = C(i, j, device).data();
                                    lda_gemm_0 = A(i, 0, device).stride();
                                    ldb_gemm_0 = A(j, 0, device).stride();
                                    ldc_gemm_0 = C(i, j, device).stride();
                                    ++batch_count_gemm_0;
                                }
                            }
                        }
                    }

                    std::vector<scalar_t*> a_array_host_gemm_1(batch_size_gemm);
                    std::vector<scalar_t*> b_array_host_gemm_1(batch_size_gemm);
                    std::vector<scalar_t*> c_array_host_gemm_1(batch_size_gemm);

                    int64_t batch_count_gemm_1 = 0;
                    int64_t lda_gemm_1 = 0;
                    int64_t ldb_gemm_1 = 0;
                    int64_t ldc_gemm_1 = 0;

                    int64_t mb1 = C.tileMb(C.mt()-1);
                    int64_t nb1 = C.tileNb(0);

                    {
                        int64_t i = C.mt()-1;
                        for (int64_t j = 0; j < C.nt()-1; ++j) {
                            if (C.tileIsLocal(i, j)) {
                                if (device == C.tileDevice(i, j)) {
                                    a_array_host_gemm_1[batch_count_gemm_1]
                                        = A(i, 0, device).data();
                                    b_array_host_gemm_1[batch_count_gemm_1]
                                        = A(j, 0, device).data();
                                    c_array_host_gemm_1[batch_count_gemm_1]
                                        = C(i, j, device).data();
                                    lda_gemm_1 = A(i, 0, device).stride();
                                    ldb_gemm_1 = A(j, 0, device).stride();
                                    ldc_gemm_1 = C(i, j, device).stride();
                                    ++batch_count_gemm_1;
                                }
                            }
                        }
                    }

                    if (C.op() != Op::NoTrans) {
                        // swap A <=> B; swap m <=> n
                        swap(opA, opB);
                        swap(a_array_host_gemm_0, b_array_host_gemm_0);
                        swap(a_array_host_gemm_1, b_array_host_gemm_1);
                        swap(lda_gemm_0, ldb_gemm_0);
                        swap(lda_gemm_1, ldb_gemm_1);
                        swap(mb0, nb0);
                        swap(mb1, nb1);
                    }

                    blas::Queue* queue = C.queue(device);
                    std::vector<Op> transA(1, opA);
                    std::vector<int64_t> k(1, kb);
                    std::vector<scalar_t> alpha_(1, scalar_t(alpha));
                    std::vector<scalar_t> beta_(1, scalar_t(beta));

                    {
                        trace::Block trace_block("blas::batch::gemm");

                        std::vector<Op> transB(1, opB);

                        if (batch_count_gemm_0 > 0) {
                            std::vector<int64_t> m(1, mb0);
                            std::vector<int64_t> n(1, nb0);
                            std::vector<int64_t> ldda(1, lda_gemm_0);
                            std::vector<int64_t> lddb(1, ldb_gemm_0);
                            std::vector<int64_t> lddc(1, ldc_gemm_0);
                            std::vector<int64_t> info(batch_count_gemm_0);
                            blas::batch::gemm(
                                layout, transA, transB,
                                m, n, k,
                                alpha_, a_array_host_gemm_0, ldda,
                                        b_array_host_gemm_0, lddb,
                                beta_,  c_array_host_gemm_0, lddc,
                                batch_count_gemm_0, info, *queue);
                        }

                        if (batch_count_gemm_1 > 0) {
                            std::vector<int64_t> m(1, mb1);
                            std::vector<int64_t> n(1, nb1);
                            std::vector<int64_t> ldda(1, lda_gemm_1);
                            std::vector<int64_t> lddb(1, ldb_gemm_1);
                            std::vector<int64_t> lddc(1, ldc_gemm_1);
                            std::vector<int64_t> info(batch_count_gemm_1);
                            blas::batch::gemm(
                                layout, transA, transB,
                                m, n, k,
                                alpha_, a_array_host_gemm_1, ldda,
                                        b_array_host_gemm_1, lddb,
                                beta_,  c_array_host_gemm_1, lddc,
                                batch_count_gemm_1, info, *queue);
                        }
                    }

                    #pragma omp task default(shared)
                    {
                        A.tileGetForReading(
                            A_tiles_syrk, device, LayoutConvert(layout));
                    }
                    #pragma omp task default(shared)
                    {
                        C.tileGetForWriting(
                            C_tiles_syrk, device, LayoutConvert(layout));
                    }
                    #pragma omp taskwait

                    int64_t batch_size_syrk = C_tiles_syrk.size();

                    std::vector<scalar_t*> a_array_host_syrk(batch_size_syrk);
                    std::vector<scalar_t*> c_array_host_syrk(batch_size_syrk);

                    int64_t batch_count_syrk = 0;
                    int64_t lda_syrk = 0;
                    int64_t ldc_syrk = 0;

                    for (auto iter  = C_tiles_syrk.begin();
                              iter != C_tiles_syrk.end();
                            ++iter) {
                        int64_t j = std::get<1>(*iter);
                        a_array_host_syrk[batch_count_syrk]
                            = A(j, 0, device).data();
                        c_array_host_syrk[batch_count_syrk]
                            = C(j, j, device).data();
                        lda_syrk = A(j, 0, device).stride();
                        ldc_syrk = C(j, j, device).stride();
                        ++batch_count_syrk;
                    }

                    {
                        trace::Block trace_block("blas::batch::syrk");

                        std::vector<Uplo> uplo(1, C.uploPhysical());

                        if (batch_count_syrk > 0) {
                            std::vector<int64_t> n(1, nb0);
                            std::vector<int64_t> ldda(1, lda_syrk);
                            std::vector<int64_t> lddc(1, ldc_syrk);
                            std::vector<int64_t> info(batch_count_syrk);
                            blas::batch::syrk(
                                layout, uplo, transA,
                                n, k,
                                alpha_, a_array_host_syrk, ldda,
                                beta_,  c_array_host_syrk, lddc,
                                batch_count_syrk, info, *queue);
                        }
                    }

                    queue->sync();

                    // both off-diagonal batch gemm and diagonal syrk are done
                    for (int64_t j = 0; j < C.nt()-1; ++j) {
                        for (int64_t i = j+1; i < C.mt(); ++i) {
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
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }
    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void syrk<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

template
void syrk<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

template
void syrk<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

template
void syrk<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

// ----------------------------------------
template
void syrk<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

template
void syrk<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

template
void syrk<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

template
void syrk<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

// ----------------------------------------
template
void syrk< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

template
void syrk< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

template
void syrk< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

template
void syrk< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

// ----------------------------------------
template
void syrk< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

template
void syrk< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

template
void syrk< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

template
void syrk< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

} // namespace internal
} // namespace slate
