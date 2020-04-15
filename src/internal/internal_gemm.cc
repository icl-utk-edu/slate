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

#include "slate/Matrix.hh"
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
          Layout layout, int priority, int64_t batch_arrays_index)
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
         layout, priority, batch_arrays_index);
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
          Layout layout, int priority, int64_t batch_arrays_index)
{
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'

    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    // check dimensions
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

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

    for (int64_t i = 0; i < C.mt(); ++i) {
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(i, j)) {
                #pragma omp task shared(A, B, C, err, err_msg) priority(priority)
                {
                    try {
                        C.tileGetForWriting(i, j, LayoutConvert(layout));
                        gemm(alpha, A(i, 0),
                                    B(0, j),
                             beta,  C(i, j));
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
    }

    #pragma omp taskwait

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
          Layout layout, int priority, int64_t batch_arrays_index)
{
    // check dimensions
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

    int err = 0;
    std::string err_msg;
    const int64_t C_mt = C.mt();
    const int64_t C_nt = C.nt();
    //  #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) shared(A, B, C, err, err_msg)
    for (int64_t i = 0; i < C_mt; ++i) {
        for (int64_t j = 0; j < C_nt; ++j) {
            if (C.tileIsLocal(i, j)) {
                try {
                    A.tileGetForReading(i, 0, LayoutConvert(layout));
                    B.tileGetForReading(0, j, LayoutConvert(layout));
                    C.tileGetForWriting(i, j, LayoutConvert(layout));
                    gemm(alpha, A(i, 0),
                                B(0, j),
                         beta,  C(i, j));
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

    // #pragma omp taskwait

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
          Layout layout, int priority, int64_t batch_arrays_index)
{
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
    #pragma omp task default(shared)
    {
        A.tileGetForReading(A_tiles_set, LayoutConvert(layout));
    }
    #pragma omp task default(shared)
    {
        B.tileGetForReading(B_tiles_set, LayoutConvert(layout));
    }
    #pragma omp task default(shared)
    {
        C.tileGetForWriting(C_tiles_set, LayoutConvert(layout));
    }
    #pragma omp taskwait

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
            #ifdef SLATE_WITH_MKL
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
            #else
                assert(false);
            #endif
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

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
/// General matrix multiply to update trailing matrix,
/// where A is a single block column and B is a single block row.
/// GPU device batched cuBLAS implementation.
/// GPU can use either ColMajor or RowMajor.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(internal::TargetType<Target::Devices>,
          scalar_t alpha, Matrix< scalar_t >& A,
                          Matrix< scalar_t >& B,
          scalar_t beta,  Matrix< scalar_t >& C,
          Layout layout, int priority, int64_t batch_arrays_index)
{
    using blas::conj;
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // check dimensions
    assert(C.mt() > 0);
    assert(C.nt() > 0);
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(A.mt() == C.mt());
    assert(B.nt() == C.nt());

    assert(C.num_devices() > 0);

    int err = 0;
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, B, C, err) priority(priority)
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
            #pragma omp task default(shared)
            {
                A.tileGetForReading(A_tiles_set, device, LayoutConvert(layout));
            }
            #pragma omp task default(shared)
            {
                B.tileGetForReading(B_tiles_set, device, LayoutConvert(layout));
            }
            #pragma omp task default(shared)
            {
                C.tileGetForWriting(C_tiles_set, device, LayoutConvert(layout));
            }
            #pragma omp taskwait

            int64_t batch_size = C_tiles_set.size();
            scalar_t** a_array_host = C.array_host(device, batch_arrays_index);
            scalar_t** b_array_host = a_array_host + batch_size;
            scalar_t** c_array_host = b_array_host + batch_size;

            int64_t batch_count = 0;
            int64_t batch_count_00 = 0;
            int64_t lda00 = 0;
            int64_t ldb00 = 0;
            int64_t ldc00 = 0;
            int64_t mb00 = C.tileMb(0);
            int64_t nb00 = C.tileNb(0);
            int64_t kb = A.tileNb(0);   // == A.tileMb(0)
            for (int64_t i = 0; i < C.mt()-1; ++i) {
                for (int64_t j = 0; j < C.nt()-1; ++j) {
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            a_array_host[batch_count] = A(i, 0, device).data();
                            b_array_host[batch_count] = B(0, j, device).data();
                            c_array_host[batch_count] = C(i, j, device).data();
                            lda00 = A(i, 0, device).stride();
                            ldb00 = B(0, j, device).stride();
                            ldc00 = C(i, j, device).stride();
                            ++batch_count_00;
                            ++batch_count;
                        }
                    }
                }
            }
            int64_t batch_count_10 = 0;
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
                            a_array_host[batch_count] = A(i, 0, device).data();
                            b_array_host[batch_count] = B(0, j, device).data();
                            c_array_host[batch_count] = C(i, j, device).data();
                            lda10 = A(i, 0, device).stride();
                            ldb10 = B(0, j, device).stride();
                            ldc10 = C(i, j, device).stride();
                            ++batch_count_10;
                            ++batch_count;
                        }
                    }
                }
            }
            int64_t batch_count_01 = 0;
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
                            a_array_host[batch_count] = A(i, 0, device).data();
                            b_array_host[batch_count] = B(0, j, device).data();
                            c_array_host[batch_count] = C(i, j, device).data();
                            lda01 = A(i, 0, device).stride();
                            ldb01 = B(0, j, device).stride();
                            ldc01 = C(i, j, device).stride();
                            ++batch_count_01;
                            ++batch_count;
                        }
                    }
                }
            }
            int64_t batch_count_11 = 0;
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
                        a_array_host[batch_count] = A(i, 0, device).data();
                        b_array_host[batch_count] = B(0, j, device).data();
                        c_array_host[batch_count] = C(i, j, device).data();
                        lda11 = A(i, 0, device).stride();
                        ldb11 = B(0, j, device).stride();
                        ldc11 = C(i, j, device).stride();
                        ++batch_count_11;
                        ++batch_count;
                    }
                }
            }

            slate_assert(batch_count == batch_size);

            scalar_t** a_array_dev = C.array_device(device, batch_arrays_index);
            scalar_t** b_array_dev = a_array_dev + batch_size;
            scalar_t** c_array_dev = b_array_dev + batch_size;

            if (C.op() != Op::NoTrans) {
                // swap A <=> B; swap m <=> n
                swap(opA, opB);
                swap(a_array_dev, b_array_dev);
                swap(lda00, ldb00);
                swap(lda10, ldb10);
                swap(lda01, ldb01);
                swap(lda11, ldb11);
                swap(mb00, nb00);
                swap(mb10, nb10);
                swap(mb01, nb01);
                swap(mb11, nb11);
            }

            slate_cuda_call(
                cudaSetDevice(device));

            // cublas_handle uses this stream
            cudaStream_t stream = C.compute_stream(device);
            cublasHandle_t cublas_handle = C.cublas_handle(device);

            slate_cuda_call(
                cudaMemcpyAsync(C.array_device(device, batch_arrays_index),
                                C.array_host(device, batch_arrays_index),
                                sizeof(scalar_t*)*batch_count*3,
                                cudaMemcpyHostToDevice,
                                stream));

            {
                trace::Block trace_block("cublasGemmBatched");
                if (batch_count_00 > 0) {
                    if (layout == Layout::ColMajor) {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opA), cublas_op_const(opB),
                                mb00, nb00, kb,
                                &alpha, (const scalar_t**) a_array_dev, lda00,
                                        (const scalar_t**) b_array_dev, ldb00,
                                &beta,                     c_array_dev, ldc00,
                                batch_count_00));
                    }
                    else {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opB), cublas_op_const(opA),
                                nb00, mb00, kb,
                                &alpha, (const scalar_t**) b_array_dev, ldb00,
                                        (const scalar_t**) a_array_dev, lda00,
                                &beta,                     c_array_dev, ldc00,
                                batch_count_00));
                    }
                    a_array_dev += batch_count_00;
                    b_array_dev += batch_count_00;
                    c_array_dev += batch_count_00;
                }

                if (batch_count_10 > 0) {
                    if (layout == Layout::ColMajor) {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opA), cublas_op_const(opB),
                                mb10, nb10, kb,
                                &alpha, (const scalar_t**) a_array_dev, lda10,
                                        (const scalar_t**) b_array_dev, ldb10,
                                &beta,                     c_array_dev, ldc10,
                                batch_count_10));
                    }
                    else {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opB), cublas_op_const(opA),
                                nb10, mb10, kb,
                                &alpha, (const scalar_t**) b_array_dev, ldb10,
                                        (const scalar_t**) a_array_dev, lda10,
                                &beta,                     c_array_dev, ldc10,
                                batch_count_10));
                    }
                    a_array_dev += batch_count_10;
                    b_array_dev += batch_count_10;
                    c_array_dev += batch_count_10;
                }

                if (batch_count_01 > 0) {
                    if (layout == Layout::ColMajor) {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opA), cublas_op_const(opB),
                                mb01, nb01, kb,
                                &alpha, (const scalar_t**) a_array_dev, lda01,
                                        (const scalar_t**) b_array_dev, ldb01,
                                &beta,                     c_array_dev, ldc01,
                                batch_count_01));
                    }
                    else {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opB), cublas_op_const(opA),
                                nb01, mb01, kb,
                                &alpha, (const scalar_t**) b_array_dev, ldb01,
                                        (const scalar_t**) a_array_dev, lda01,
                                &beta,                     c_array_dev, ldc01,
                                batch_count_01));
                    }
                    a_array_dev += batch_count_01;
                    b_array_dev += batch_count_01;
                    c_array_dev += batch_count_01;
                }

                if (batch_count_11 > 0) {
                    if (layout == Layout::ColMajor) {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opA), cublas_op_const(opB),
                                mb11, nb11, kb,
                                &alpha, (const scalar_t**) a_array_dev, lda11,
                                        (const scalar_t**) b_array_dev, ldb11,
                                &beta,                     c_array_dev, ldc11,
                                batch_count_11));
                    }
                    else {
                        slate_cublas_call(
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opB), cublas_op_const(opA),
                                nb11, mb11, kb,
                                &alpha, (const scalar_t**) b_array_dev, ldb11,
                                        (const scalar_t**) a_array_dev, lda11,
                                &beta,                     c_array_dev, ldc11,
                                batch_count_11));
                    }
                }

                slate_cuda_call(
                    cudaStreamSynchronize(stream));
            }

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

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void gemm<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

// ----------------------------------------
template
void gemm<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

// ----------------------------------------
template
void gemm< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

// ----------------------------------------
template
void gemm< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

template
void gemm< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t batch_arrays_index);

} // namespace internal
} // namespace slate
