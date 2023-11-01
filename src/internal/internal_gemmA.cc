// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
/// General matrix multiply for a left-looking update.
/// Does computation where the A tiles are local.
/// If needed (e.g., not local), inserts C tiles and sets beta to 0.
/// On output, partial products Cik exist distributed wherever Aij is local,
/// for all i = 0 : A.mt, j = 0 : A.nt, k=.
/// Dispatches to target implementations.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conj_transpose;
/// if $op(C)$ is conj_transpose, then $op(A)$ and $op(B)$ cannot be transpose.
/// @ingroup gemm_internal
///
template <Target target, typename scalar_t>
void gemmA(scalar_t alpha, Matrix<scalar_t>&& A,
                           Matrix<scalar_t>&& B,
           scalar_t beta,  Matrix<scalar_t>&& C,
           Layout layout, int priority, int64_t queue_index,
           Options const& opts)
{
    if (C.is_complex &&
        ((C.op() == Op::Trans &&
         (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans)) ||
         (C.op() == Op::ConjTrans &&
         (A.op() == Op::Trans || B.op() == Op::Trans))))
    {
        throw std::exception();
    }

    gemmA( internal::TargetType<target>(),
          alpha, A,
                 B,
          beta,  C,
          layout, priority, queue_index, opts );
}

//------------------------------------------------------------------------------
/// General matrix multiply for a left-looking update.
/// This routine consists of two steps:  the management of the memory
/// and the actual computation.
/// Note that this routine may insert some tiles that must be erased later.
/// The original intent was to erase them when calling the ListReduce routine.
/// First step:
///   It iterates over the tiles of A and gets the needed tiles of B and C.
///   In the case where the corresponding tiles of C do not exist, it
///   acquires them. It means these tiles are created and inserted as workspace.
///   It is expected that they are erased either by the calling routine or
///   through the call to ListReduce.
/// Second step:
///   As soon as the data is ready, the routine calls gemm. However, the beta
///   is used once and only in the case where the current tile of C existed
///   prior the call to this routine. Otherwise, the beta value is replaced
///   by zero.
///   In any case, the internal value beta_j is set to one to allow the
///   accumulation in the tile.
/// Host OpenMP task implementation.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemmA(internal::TargetType<Target::HostTask>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  Matrix<scalar_t>& C,
           Layout layout, int priority, int queue_index,
           Options const& opts)
{
    // check dimensions
    assert( A.nt() == B.mt() );
    assert( A.mt() == C.mt() );

    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    int err   = 0;
    // This assumes that if a tile has to be acquired, then all tiles
    // have to be acquired
    // TODO make it a matrix of the C tiles involved c.TileAcquire(i, k)
    int c_tile_acquired = 0;
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        int nlocal_A_tiles = 0;
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal( i, j )) {
                nlocal_A_tiles++;
                #pragma omp task slate_omp_default_none \
                    shared( A, B, C, err, c_tile_acquired ) \
                    firstprivate(i, j, layout) priority(priority)
                {
                    try {
                        A.tileGetForReading( i, j, HostNum, LayoutConvert( layout ) );
                        for (int64_t k = 0; k < B.nt(); ++k) {
                            B.tileGetForReading(
                                j, k, HostNum, LayoutConvert( layout ) );

                            if (C.tileIsLocal( i, k )) {
                                C.tileGetForWriting(
                                    i, k, HostNum, LayoutConvert( layout ) );
                            }
                            else {
                                if (! C.tileExists( i, k, HostNum )) {
                                    c_tile_acquired = 1;
                                    #pragma omp critical
                                    {
                                        C.tileAcquire( i, k, HostNum, layout );
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
        // Set or scale tiles of C when the tiles are not updated but
        // are part of the distribution, i.e., will be used for the reduction.
        if (nlocal_A_tiles == 0 && beta != one) {
            for (int64_t j = 0; j < B.nt(); ++j) {
                if (C.tileIsLocal( i, j )) {
                    #pragma omp task slate_omp_default_none \
                        shared( C, beta ) firstprivate( i, j, zero, layout )
                    {
                        C.tileGetForWriting( i, j, HostNum, LayoutConvert( layout ) );
                        if (beta == zero) {
                            C( i, j ).set( zero );
                        }
                        else {
                            tile::scale( beta, C( i, j ) );
                        }
                    }
                }
            }
        }
    }

    if (err)
        slate_error(
            std::string( "Error in omp-task line: " ) + std::to_string( err ) );

    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        #pragma omp task slate_omp_default_none \
            shared( A, B, C, err ) \
            firstprivate(i, alpha, beta, zero, one, c_tile_acquired) \
            priority(priority)
        {
            try {
                scalar_t beta_j;
                for (int64_t k = 0; k < B.nt(); ++k) {
                    if (! c_tile_acquired || C.tileIsLocal( i, k )) {
                        beta_j = beta;
                    }
                    else {
                        beta_j = zero;
                    }
                    bool Cik_modified = false;
                    for (int64_t j = 0; j < A.nt(); ++j) {
                        if (A.tileIsLocal( i, j )) {
                            tile::gemm(
                                alpha,  A( i, j ), B( j, k ),
                                beta_j, C( i, k ) );

                            beta_j = one;

                            A.tileTick( i, j );
                            B.tileTick( j, k );
                            Cik_modified = true;
                        }
                    }
                    if (Cik_modified)
                        // mark this tile modified
                        C.tileModified( i, k );
                }
            }
            catch (std::exception& e) {
                err = __LINE__;
            }
        }
    }

    if (err)
        slate_error(
            std::string( "Error in omp-task line: " ) + std::to_string( err ) );
}

template <typename scalar_t>
void gemmA(internal::TargetType<Target::HostNest>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  Matrix<scalar_t>& C,
           Layout layout, int priority, int queue_index,
           Options const& opts)
{
    gemmA( internal::TargetType<Target::HostTask>(),
            alpha, A, B, beta, C, layout, priority, queue_index, opts );
}

template <typename scalar_t>
void gemmA(internal::TargetType<Target::HostBatch>,
           scalar_t alpha, Matrix<scalar_t>& A,
                           Matrix<scalar_t>& B,
           scalar_t beta,  Matrix<scalar_t>& C,
           Layout layout, int priority, int queue_index,
           Options const& opts)
{
    gemmA( internal::TargetType<Target::HostTask>(),
            alpha, A, B, beta, C, layout, priority, queue_index, opts );
}

//------------------------------------------------------------------------------
/// General matrix multiply for a left-looking update
/// where TODO
/// GPU device batched-BLAS implementation.
/// GPU can use either ColMajor or RowMajor.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemmA(internal::TargetType<Target::Devices>,
          scalar_t alpha, Matrix< scalar_t >& A,
                          Matrix< scalar_t >& B,
          scalar_t beta,  Matrix< scalar_t >& C,
          Layout layout, int priority, int64_t queue_index,
          Options const& opts)
{
    using blas::conj;
    using std::swap;
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    TileReleaseStrategy tile_release_strategy = get_option(
            opts, Option::TileReleaseStrategy, TileReleaseStrategy::All );

    // check dimensions
    // TODO add more?
    assert( A.mt() == C.mt() );
    assert( B.nt() == C.nt() );
    assert( B.nt() == 1 ); // TODO for now, one. Could it be more?

    assert(C.num_devices() > 0);

    int err = 0;
    const scalar_t zero = 0;
    const scalar_t one  = 1.0;

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

    if (err)
        slate_error( std::to_string( err ) );

    if (C.op() == Op::ConjTrans) {
        alpha = conj( alpha );
        beta  = conj( beta );
    }


    // In the case where some C tiles are not touched locally but involved
    // in the reduce process, we scale it here first.
    if (beta != one) {
        std::set<int> queues_to_sync;
        #pragma omp taskgroup
        for (int64_t i = 0; i < A.mt(); ++i) {
            int nlocal_A_row_i_tiles_touched = 0;
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (A.tileIsLocal( i, j )) {
                    nlocal_A_row_i_tiles_touched++;
                }
            }

            if (nlocal_A_row_i_tiles_touched == 0 && C.tileIsLocal( i, 0 )) {
                int device = C.tileDevice( i, 0 );

                blas::Queue* queue = A.compute_queue( device, queue_index );
                assert( queue != nullptr );
                queues_to_sync.insert( device );

                #pragma omp task slate_omp_default_none \
                    shared( C ) \
                    firstprivate( i, beta, zero, one, layout, device, queue ) \
                    priority( priority )
                {
                    // TODO Perform the scaling where the tile origins.
                    // Unless it got modified and so it should be
                    // performed where it was last modified.

                    C.tileGetForWriting(
                        i, 0, device, LayoutConvert( layout ) );

                    auto T = C( i, 0, device );
                    int64_t T_mb = T.mb();
                    int64_t T_nb = T.nb();
                    if (T.op() != Op::NoTrans) {
                        swap( T_mb, T_nb );
                    }

                    if (beta == zero) {
                        device::geset( T_mb, T_nb, beta, beta,
                            T.data(), T.stride(), *queue );
                    }
                    else{
                        device::gescale( T_mb, T_nb, beta, one,
                            T.data(), T.stride(), *queue );
                    }
                }
            }
        }
        for (int device : queues_to_sync) {
            blas::Queue* queue = A.compute_queue( device, queue_index );
            assert( queue != nullptr );
            queue->sync();
        }
    }

    #pragma omp taskgroup
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, B, C) priority(priority) \
            firstprivate(alpha, beta, layout, queue_index, device, tile_release_strategy)
        {
            blas::Queue* queue = A.compute_queue( device, queue_index );
            assert( queue != nullptr );

            std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal( i, j )) {
                        if (device == A.tileDevice( i, j )) {
                            A_tiles_set.insert( {i, j} );
                            B_tiles_set.insert( {j, 0} );
                            C_tiles_set.insert( {i, 0} );

                            if (! C.tileExists( i, 0, device )
                                && ! C.tileExists( i, 0, HostNum ))
                            {
                                // XXX since at least cuBLAS does not
                                // take beta as a vector, we have to
                                // set new tiles to 0 explicitly.
                                C.tileInsertWorkspace( i, 0, device );
                                C.tileModified( i, 0, device );
                                auto T = C( i, 0, device );
                                int64_t T_mb = T.mb();
                                int64_t T_nb = T.nb();
                                if (T.op() != Op::NoTrans) {
                                    swap( T_mb, T_nb );
                                }

                                device::geset( T_mb, T_nb, zero, zero,
                                    T.data(), T.stride(), *queue );
                            }
                        }
                    }
                }
            }

            int64_t batch_size = A_tiles_set.size();
            if (batch_size > 0) {

                #pragma omp taskgroup
                {
                    #pragma omp task slate_omp_default_none \
                        shared( A, A_tiles_set ) firstprivate( layout, device )
                    {
                        A.tileGetForReading(
                            A_tiles_set, device, LayoutConvert( layout ) );
                    }
                    #pragma omp task slate_omp_default_none \
                        shared( B, B_tiles_set ) firstprivate( layout, device )
                    {
                        B.tileGetForReading(
                            B_tiles_set, device, LayoutConvert( layout ) );
                    }
                    #pragma omp task slate_omp_default_none \
                        shared( C, C_tiles_set ) firstprivate( layout, device )
                    {
                        C.tileGetForWriting(
                            C_tiles_set, device, LayoutConvert( layout ) );
                    }
                }

                // Use A's batched arrays since C's may be too small
                scalar_t** a_array_host = A.array_host(device, queue_index);
                scalar_t** b_array_host = a_array_host + batch_size;
                scalar_t** c_array_host = b_array_host + batch_size;

                if (C.op() != Op::NoTrans) {
                    // swap A <=> B; swap m <=> n
                    swap( opA, opB );
                }

                std::vector<Op> opA_(1, opA);
                std::vector<Op> opB_(1, opB);
                std::vector<scalar_t> alpha_(1, alpha);
                std::vector<scalar_t> beta_(1, beta);
                // info size 0 disables slow checks in batched BLAS++.
                std::vector<int64_t> info;

                for (int64_t j = 0; j < A.nt(); ++j) {
                    auto A_j = A.sub( 0, A.mt()-1, j, j );
                    auto B_j = B.sub( j, j, 0, 0 );
                    // A comes first since we do computation for a local A
                    auto group_params = device_regions_build<false, 3, scalar_t>(
                            {A_j, B_j, C},
                            {a_array_host, b_array_host, c_array_host},
                            device );

                    trace::Block trace_block("blas::batch::gemm");

                    std::vector<int64_t> k(1, A.tileNb(j));

                    for (size_t g = 0; g < group_params.size(); ++g) {

                        int64_t group_count = group_params[ g ].count;

                        std::vector<int64_t>    m(1, group_params[ g ].mb);
                        std::vector<int64_t>    n(1, C.tileNb(0));
                        std::vector<int64_t> ldda(1, group_params[ g ].ld[0]);
                        std::vector<int64_t> lddb(1, group_params[ g ].ld[1]);
                        std::vector<int64_t> lddc(1, group_params[ g ].ld[2]);

                        std::vector<scalar_t*> a_array(a_array_host, a_array_host+group_count);
                        std::vector<scalar_t*> b_array(b_array_host, b_array_host+group_count);
                        std::vector<scalar_t*> c_array(c_array_host, c_array_host+group_count);

                        if (C.op() != Op::NoTrans) {
                            swap(m, n);
                            swap(a_array, b_array);
                            swap(ldda, lddb);
                        }

                        blas::batch::gemm(
                            layout, opA_, opB_,
                            m, n, k,
                            alpha_, a_array, ldda,
                                    b_array, lddb,
                            beta_,  c_array, lddc,
                            group_count, info, *queue);

                        a_array_host += group_count;
                        b_array_host += group_count;
                        c_array_host += group_count;
                    }

                    // Only scale C once
                    // TODO relax assumption on the distribution
                    if (group_params.size() > 0) {
                        beta_[0] = one;
                    }
                }

                {
                    trace::Block trace_block("blas::batch::gemm");
                    queue->sync();
                }

                if (tile_release_strategy == TileReleaseStrategy::Internal
                    || tile_release_strategy == TileReleaseStrategy::All)
                {
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        for (int64_t j = 0; j < A.nt(); ++j) {
                            if (A.tileIsLocal( i, j )) {
                                if (device == A.tileDevice( i, j )) {
                                    A.tileRelease( i, j, device );
                                    // erase tmp local and remote device tiles;
                                    B.tileRelease( j, 0, device ); // XXX Should it stay here?
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void gemmA<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

// ----------------------------------------
template
void gemmA<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

// ----------------------------------------
template
void gemmA<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

// ----------------------------------------
template
void gemmA<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

template
void gemmA< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    Layout layout, int priority, int64_t queue_index,
    Options const& opts);

} // namespace internal
} // namespace slate
