// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
          int priority, int queue_index, Layout layout )
{
    if (! ((C.uplo() == Uplo::Lower)
           &&
           (C.is_real || (C.op() != Op::Trans &&
                          A.op() != Op::Trans))))
        throw std::exception();

    herk(internal::TargetType<target>(),
         alpha, A,
         beta,  C,
         priority, queue_index, layout );
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
          int priority, int queue_index, Layout layout )
{
    scalar_t alpha_ = scalar_t(alpha);
    scalar_t beta_  = scalar_t(beta);

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
                        firstprivate( j, layout, alpha, beta )
                    {
                        try {
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(j, j, LayoutConvert(layout));
                            tile::herk(
                                alpha, A(j, 0),
                                beta,  C(j, j) );
                        }
                        catch (std::exception& e) {
                            err = __LINE__;
                        }
                    }
                }
                else {
                    #pragma omp task slate_omp_default_none \
                        shared( A, C, err ) priority( priority ) \
                        firstprivate( i, j, layout, alpha_, beta_ )
                    {
                        try {
                            A.tileGetForReading(i, 0, LayoutConvert(layout));
                            A.tileGetForReading(j, 0, LayoutConvert(layout));
                            C.tileGetForWriting(i, j, LayoutConvert(layout));
                            auto Aj0 = A(j, 0);
                            tile::gemm(
                                alpha_, A(i, 0), conj_transpose( Aj0 ),
                                beta_,  C(i, j) );
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
          int priority, int queue_index, Layout layout )
{
#if defined(SLATE_HAVE_OMPTARGET) || defined(SLATE_SKIP_HOSTNEST)
    // SYCL/OMP-target-offload can't process this section
    slate_not_implemented("Target::HostNest isn't supported in this configuration.");
#else
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
                firstprivate( j, layout, alpha, beta )
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    tile::herk(
                        alpha, A(j, 0),
                        beta,  C(j, j) );
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
        shared( A, C, err ) firstprivate( C_nt, C_mt, layout, beta_, alpha_ )
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
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Host batched implementation.
/// Assumes A is NoTrans or ConjTrans; C is Lower, NoTrans or Upper, ConjTrans.
/// @ingroup herk_internal
///
template <typename scalar_t>
void herk(internal::TargetType<Target::HostBatch>,
          blas::real_type<scalar_t> alpha, Matrix<scalar_t>& A,
          blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
          int priority, int queue_index, Layout layout )
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
                shared( A, C, err ) firstprivate( j, layout, alpha, beta )
            {
                try {
                    A.tileGetForReading(j, 0, LayoutConvert(layout));
                    C.tileGetForWriting(j, j, LayoutConvert(layout));
                    tile::herk(
                        alpha, A(j, 0),
                        beta,  C(j, j) );
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
          int priority, int queue_index, Layout layout )
{
    int err = 0;
    using std::swap;
    using real_t = blas::real_type<scalar_t>;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    assert(C.num_devices() > 0);

    // if single tile, avoid creating tasks for all devices
    #pragma omp taskgroup
    if (C.nt() == 1) {
        if (C.tileIsLocal(0, 0)) {
            #pragma omp task slate_omp_default_none \
                shared( A, C, err ) priority( priority ) \
                firstprivate( layout, queue_index, alpha, beta )
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
            }
        }
    }
    else {
        // off-diagonal tiles by batch gemm on device
        // diagonal tiles by herk on device
        for (int device = 0; device < C.num_devices(); ++device) {
            #pragma omp task slate_omp_default_none \
                shared( A, C, err ) priority( priority ) \
                firstprivate( layout, queue_index, device, alpha, beta )
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

                    std::set<ij_tuple> A_tiles_set, C_tiles_set;
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        for (int64_t i = j; i < C.mt(); ++i) {  // lower
                            if (C.tileIsLocal(i, j)
                                && device == C.tileDevice(i, j)) {
                                A_tiles_set.insert({j, 0});
                                C_tiles_set.insert({i, j});
                                if (i != j) {
                                    A_tiles_set.insert({i, 0});
                                }
                            }
                        }
                    }

                    #pragma omp taskgroup
                    {
                        #pragma omp task slate_omp_default_none \
                            shared( A, A_tiles_set ) \
                            firstprivate(device, layout)
                        {
                            A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
                        }
                        #pragma omp task slate_omp_default_none \
                            shared( C, C_tiles_set ) \
                            firstprivate(device, layout)
                        {
                            C.tileGetForWriting(C_tiles_set, device, LayoutConvert(layout));
                        }
                    }

                    int64_t batch_size = C_tiles_set.size();

                    scalar_t** a_array_host = C.array_host(device, queue_index);
                    scalar_t** b_array_host = a_array_host + batch_size;
                    scalar_t** c_array_host = b_array_host + batch_size;

                    // Use transposed A to broadcast down the rows correctly
                    auto AT = conj_transpose(A);

                    // C comes first since we do computation for a local C
                    auto group_params = device_regions_build<true, 3, scalar_t>(
                            {C, A, AT},
                            {c_array_host, a_array_host, b_array_host},
                            device );

                    if (C.op() != Op::NoTrans) {
                        swap(opA, opB);
                    }

                    {
                        trace::Block trace_block("blas::batch::herk");

                        std::vector<Op> opA_(1, opA);
                        std::vector<Op> opB_(1, opB);
                        std::vector<int64_t> k(1,  A.tileNb(0));
                        std::vector<int64_t> info;

                        std::vector<real_t>   alpha_r(1, alpha);
                        std::vector<real_t>   beta_r (1, beta);
                        std::vector<scalar_t> alpha_s(1, scalar_t(alpha));
                        std::vector<scalar_t> beta_s (1, scalar_t(beta));
                        std::vector<Uplo> uplo(1, C.uploPhysical());

                        blas::Queue* queue = C.compute_queue(device, queue_index);

                        for (size_t g = 0; g < group_params.size(); ++g) {

                            int64_t group_count = group_params[ g ].count;

                            std::vector<int64_t>    n(1, group_params[ g ].nb);
                            std::vector<int64_t> ldda(1, group_params[ g ].ld[1]);
                            std::vector<int64_t> lddc(1, group_params[ g ].ld[0]);
                            std::vector<scalar_t*> a_array(a_array_host, a_array_host+group_count);
                            std::vector<scalar_t*> c_array(c_array_host, c_array_host+group_count);

                            if (group_params[ g ].is_diagonal) {
                                blas::batch::herk(
                                    layout, uplo, opA_,
                                    n, k,
                                    alpha_r, a_array, ldda,
                                    beta_r,  c_array, lddc,
                                    group_count, info, *queue);
                            }
                            else {
                                std::vector<int64_t>    m(1, group_params[ g ].mb);
                                std::vector<int64_t> lddb(1, group_params[ g ].ld[2]);

                                std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);

                                if (C.op() != Op::NoTrans) {
                                    swap(m, n);
                                    swap(a_array, b_array);
                                    swap(ldda, lddb);
                                }

                                blas::batch::gemm(
                                    layout, opA_, opB_,
                                    m, n, k,
                                    alpha_s, a_array, ldda,
                                             b_array, lddb,
                                    beta_s,  c_array, lddc,
                                    group_count, info, *queue);
                            }
                            a_array_host += group_count;
                            b_array_host += group_count;
                            c_array_host += group_count;
                        }

                        queue->sync();
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
    int priority, int queue_index, Layout layout );

template
void herk<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
    float beta,  HermitianMatrix<float>&& C,
    int priority, int queue_index, Layout layout );

template
void herk<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
    float beta,  HermitianMatrix<float>&& C,
    int priority, int queue_index, Layout layout );

template
void herk<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
    float beta,  HermitianMatrix<float>&& C,
    int priority, int queue_index, Layout layout );

// ----------------------------------------
template
void herk<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout );

template
void herk<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout );

template
void herk<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout );

template
void herk<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
    double beta,  HermitianMatrix<double>&& C,
    int priority, int queue_index, Layout layout );

// ----------------------------------------
template
void herk< Target::HostTask, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout );

template
void herk< Target::HostNest, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout );

template
void herk< Target::HostBatch, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout );

template
void herk< Target::Devices, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority, int queue_index, Layout layout );

// ----------------------------------------
template
void herk< Target::HostTask, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout );

template
void herk< Target::HostNest, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout );

template
void herk< Target::HostBatch, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout );

template
void herk< Target::Devices, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority, int queue_index, Layout layout );

} // namespace internal
} // namespace slate
