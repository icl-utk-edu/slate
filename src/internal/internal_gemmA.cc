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
/// General matrix multiply for a left-looking update.
/// Does computation where the A tiles are local.
/// If needed (e.g., not local), inserts C tiles and sets beta to 0.
/// On output, partial products Cik exist distributed wherever Aij is local,
/// for all i = 0 : A.mt, j = 0 : A.nt, k=.
/// Dispatches to target implementations.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conjTranspose;
/// if $op(C)$ is conjTranspose, then $op(A)$ and $op(B)$ cannot be transpose.
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

    int err   = 0;
    // This assumes that if a tile has to be acquired, then all tiles
    // have to be acquired
    // TODO make it a matrix of the C tiles involved c.TileAcquire(i, k)
    int c_tile_acquired = 0;
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal( i, j )) {
                #pragma omp task slate_omp_default_none \
                    shared( A, B, C, err, c_tile_acquired ) \
                    firstprivate(i, j, layout) priority(priority)
                {
                    try {
                        A.tileGetForReading( i, j, LayoutConvert( layout ) );
                        for (int64_t k = 0; k < B.nt(); ++k) {
                            B.tileGetForReading(
                                j, k, LayoutConvert( layout ) );

                            if (C.tileIsLocal( i, k )) {
                                C.tileGetForWriting(
                                    i, k, LayoutConvert( layout ) );
                            }
                            else {
                                if (! C.tileExists( i, k )) {
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
    }

    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        #pragma omp task slate_omp_default_none \
            shared( A, B, C, err ) \
            firstprivate(i, alpha, beta, c_tile_acquired) priority(priority)
        {
            try {
                const scalar_t zero = 0.0;
                const scalar_t one  = 1.0;

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
    const scalar_t one = 1.0;

    // In the case where some C tiles are not touched locally but involved
    // in the reduce process, we scale it here first.
    if (beta != one) {
        #pragma omp taskgroup
        for (int64_t i = 0; i < A.mt(); ++i) {
            int cpt = 0;
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (A.tileIsLocal( i, j )) {
                    cpt++;
                }
            }
            if (cpt == 0 && C.tileIsLocal( i, 0 )) {
                #pragma omp task shared( beta ) firstprivate( i )
                {
                    // TODO Perform the scaling where the tile origins.
                    // Unless it got modified and so it should be
                    // performed where it was last modified.
                    tile::scale( beta, C( i, 0 ) );
                }
            }
        }
    }

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
                alpha = conj( alpha );
                beta  = conj( beta );
            }

            std::set<ij_tuple> A_tiles_set, B_tiles_set, C_tiles_set;
            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal( i, j )) {
                        if (device == A.tileDevice( i, j )) {
                            A_tiles_set.insert( {i, j} );
                            B_tiles_set.insert( {j, 0} );
                            C_tiles_set.insert( {i, 0} );

                            if ((! C.tileExists( i, 0, device ))
                                && (! C.tileExists( i, 0, HostNum )))
                            {
                                // XXX since at least cuBLAS does not
                                // take beta as a vector, we have to
                                // set new tiles to 0 explicitly.
                                // TODO we should insert and set it directly
                                // on the device.
                                C.tileInsert( i, 0 );
                                C( i, 0 ).set( 0 );
                            }
                        }
                    }
                }
            }

            int64_t batch_size = C_tiles_set.size();
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

                // interior, first column, and excluding bottom row
                std::vector<scalar_t*> a_array0_;
                std::vector<scalar_t*> b_array0_;
                std::vector<scalar_t*> c_array0_;
                a_array0_.reserve( batch_size );
                b_array0_.reserve( batch_size );
                c_array0_.reserve( batch_size );

                int64_t lda0_ = 0;
                int64_t ldb0_ = 0;
                int64_t ldc0_ = 0;
                int64_t mb0_ = C.tileMb( 0 );
                int64_t nb0_ = C.tileNb( 0 );
                int64_t kb   = A.tileNb( 0 );
                {
                    if (A.nt() > 1) {
                        int j = 0;
                        for (int64_t i = 0; i < A.mt()-1; ++i) {
                            if (A.tileIsLocal( i, j )) {
                                if (device == A.tileDevice( i, j )) {
                                    a_array0_.push_back(
                                        A( i, j, device ).data() );
                                    b_array0_.push_back(
                                        B( j, 0, device ).data() );
                                    c_array0_.push_back(
                                        C( i, 0, device ).data() );
                                    lda0_ = A( i, j, device ).stride();
                                    ldb0_ = B( j, 0, device ).stride();
                                    ldc0_ = C( i, 0, device ).stride();
                                }
                            }
                        }
                    }
                }

                // bottom row, first column
                scalar_t* a_array1_ = nullptr;
                scalar_t* b_array1_ = nullptr;
                scalar_t* c_array1_ = nullptr;

                int64_t lda1_ = 0;
                int64_t ldb1_ = 0;
                int64_t ldc1_ = 0;
                int64_t mb1_ = C.tileMb( C.mt()-1 );
                int64_t nb1_ = C.tileNb( 0 );
                // same kb as above
                {
                    if (A.nt() > 1 && A.mt() > 1) {
                        int64_t i = A.mt()-1;
                        int j = 0;
                        if (A.tileIsLocal( i, j )) {
                            if (device == A.tileDevice( i, j )) {
                                a_array1_ = A( i, j, device ).data();
                                b_array1_ = B( j, 0, device ).data();
                                c_array1_ = C( i, 0, device ).data();
                                lda1_ = A( i, j, device ).stride();
                                ldb1_ = B( j, 0, device ).stride();
                                ldc1_ = C( i, 0, device ).stride();
                            }
                        }
                    }
                }

                // interior, excluding first column, bottom row,
                // and right column
                std::vector< std::vector< scalar_t* > > a_array00j;
                std::vector< std::vector< scalar_t* > > b_array00j;
                std::vector< std::vector< scalar_t* > > c_array00j;

                int64_t lda00 = 0;
                int64_t ldb00 = 0;
                int64_t ldc00 = 0;
                int64_t mb00 = C.tileMb( 0 );
                int64_t nb00 = C.tileNb( 0 );
                int64_t a00i_batch_size = A.mt() - 1;
                if (A.nt() > 1) {
                    a_array00j.reserve( A.nt() - 2 );
                    b_array00j.reserve( A.nt() - 2 );
                    c_array00j.reserve( A.nt() - 2 );
                    for (int64_t j = 1; j < A.nt()-1; ++j) {
                        std::vector<scalar_t*> a_tmp;
                        std::vector<scalar_t*> b_tmp;
                        std::vector<scalar_t*> c_tmp;
                        a_tmp.reserve( a00i_batch_size );
                        b_tmp.reserve( a00i_batch_size );
                        c_tmp.reserve( a00i_batch_size );
                        for (int64_t i = 0; i < A.mt()-1; ++i) {
                            if (A.tileIsLocal( i, j )) {
                                if (device == A.tileDevice( i, j )) {
                                    a_tmp.push_back( A( i, j, device ).data() );
                                    b_tmp.push_back( B( j, 0, device ).data() );
                                    c_tmp.push_back( C( i, 0, device ).data() );
                                    lda00 = A( i, j, device ).stride();
                                    ldb00 = B( j, 0, device ).stride();
                                    ldc00 = C( i, 0, device ).stride();
                                }
                            }
                        }
                        if (a_tmp.size() > 0) {
                            a_array00j.push_back( std::move(a_tmp) );
                            b_array00j.push_back( std::move(b_tmp) );
                            c_array00j.push_back( std::move(c_tmp) );
                        }
                    }
                }

                // bottom row, excluding first and last columns
                std::vector< scalar_t* > a_array10j;
                std::vector< scalar_t* > b_array10j;
                std::vector< scalar_t* > c_array10j;

                int64_t lda10 = 0;
                int64_t ldb10 = 0;
                int64_t ldc10 = 0;
                int64_t mb10 = C.tileMb( C.mt()-1 );
                int64_t nb10 = C.tileNb( 0 );
                // same kb as above
                if (A.nt() > 1) {
                    a_array10j.reserve( A.nt() - 2 );
                    b_array10j.reserve( A.nt() - 2 );
                    c_array10j.reserve( A.nt() - 2 );
                    {
                        int64_t i = A.mt()-1;
                        for (int64_t j = 1; j < A.nt()-1; ++j) {
                            if (A.tileIsLocal( i, j )) {
                                if (device == A.tileDevice( i, j )) {
                                    a_array10j.push_back(
                                            A( i, j, device ).data() );
                                    b_array10j.push_back(
                                        B( j, 0, device ).data() );
                                    c_array10j.push_back(
                                        C( i, 0, device ).data() );
                                    lda10 = A( i, j, device ).stride();
                                    ldb10 = B( j, 0, device ).stride();
                                    ldc10 = C( i, 0, device ).stride();
                                }
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
                int64_t mb01 = C.tileMb( 0 );
                int64_t nb01 = C.tileNb( C.nt()-1 );
                int64_t kb1  = A.tileNb( A.nt()-1 );
                {
                    int64_t j = A.nt()-1;
                    for (int64_t i = 0; i < A.mt()-1; ++i) {
                        if (A.tileIsLocal( i, j )) {
                            if (device == A.tileDevice( i, j )) {
                                a_array01.push_back( A( i, j, device ).data() );
                                b_array01.push_back( B( j, 0, device ).data() );
                                c_array01.push_back( C( i, 0, device ).data() );
                                lda01 = A( i, j, device ).stride();
                                ldb01 = B( j, 0, device ).stride();
                                ldc01 = C( i, 0, device ).stride();
                            }
                        }
                    }
                }

                // bottom-right corner
                scalar_t* a_array11 = nullptr;
                scalar_t* b_array11 = nullptr;
                scalar_t* c_array11 = nullptr;

                int64_t lda11 = 0;
                int64_t ldb11 = 0;
                int64_t ldc11 = 0;
                int64_t mb11 = C.tileMb( C.mt()-1 );
                int64_t nb11 = C.tileNb( C.nt()-1 );
                // same kb1 as above
                {
                    int64_t i = A.mt()-1;
                    int64_t j = A.nt()-1;
                    if (A.tileIsLocal( i, j )) {
                        if (device == A.tileDevice( i, j )) {
                            a_array11 = A( i, j, device ).data();
                            b_array11 = B( j, 0, device ).data();
                            c_array11 = C( i, 0, device ).data();
                            lda11 = A( i, j, device ).stride();
                            ldb11 = B( j, 0, device ).stride();
                            ldc11 = C( i, 0, device ).stride();
                        }
                    }
                }

                if (C.op() != Op::NoTrans) {
                    // swap A <=> B; swap m <=> n
                    swap( opA, opB );
                    swap( a_array0_,  b_array0_ );
                    swap( a_array1_,  b_array1_ );
                    swap( a_array00j, b_array00j );
                    swap( a_array10j, b_array10j );
                    swap( a_array01,  b_array01 );
                    swap( a_array11,  b_array11 );
                    swap( lda0_, ldb0_ );
                    swap( lda1_, ldb1_ );
                    swap( lda00, ldb00 );
                    swap( lda10, ldb10 );
                    swap( lda01, ldb01 );
                    swap( lda11, ldb11 );
                    swap( mb0_, nb0_ );
                    swap( mb1_, nb1_ );
                    swap( mb00, nb00 );
                    swap( mb10, nb10 );
                    swap( mb01, nb01 );
                    swap( mb11, nb11 );
                }

                {
                    trace::Block trace_block("blas::batch::gemm");

                    std::vector<Op> opA_( 1, opA );
                    std::vector<Op> opB_( 1, opB );
                    std::vector<scalar_t> alpha_( 1, alpha );
                    std::vector<scalar_t> beta0_( 1, beta );
                    std::vector<scalar_t> beta1_( 1, beta );
                    std::vector<int64_t>  k( 1, kb );
                    std::vector<int64_t> k1( 1, kb1 );
                    // info size 0 disables slow checks in batched BLAS++.
                    std::vector<int64_t> info;

                    blas::Queue* queue = A.compute_queue( device, queue_index );
                    assert( queue != nullptr );

                    if (c_array0_.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb0_ );
                        std::vector<int64_t>    n( 1,  nb0_ );
                        std::vector<int64_t> ldda( 1, lda0_ );
                        std::vector<int64_t> lddb( 1, ldb0_ );
                        std::vector<int64_t> lddc( 1, ldc0_ );
                        blas::batch::gemm(
                            layout, opA_, opB_,
                            m, n, k,
                            alpha_, a_array0_, ldda,
                                    b_array0_, lddb,
                            beta0_, c_array0_, lddc,
                            c_array0_.size(), info, *queue );

                        beta0_[ 0 ] = one;
                    }

                    if (c_array1_ != nullptr) {
                        blas::gemm(
                            layout, opA, opB,
                            mb1_, nb1_, kb,
                            alpha,     a_array1_, lda1_,
                                       b_array1_, ldb1_,
                            beta1_[0], c_array1_, ldc1_,
                            *queue );

                        beta1_[ 0 ] = one;
                    }

                    if (c_array00j.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb00 );
                        std::vector<int64_t>    n( 1,  nb00 );
                        std::vector<int64_t> ldda( 1, lda00 );
                        std::vector<int64_t> lddb( 1, ldb00 );
                        std::vector<int64_t> lddc( 1, ldc00 );
                        for (size_t j = 0; j < c_array00j.size(); ++j) {
                            blas::batch::gemm(
                                layout, opA_, opB_,
                                m, n, k,
                                alpha_, a_array00j[ j ], ldda,
                                        b_array00j[ j ], lddb,
                                beta0_, c_array00j[ j ], lddc,
                                c_array00j[ j ].size(), info, *queue );
                            beta0_[ 0 ] = one;
                        }
                    }

                    if (c_array10j.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb10 );
                        std::vector<int64_t>    n( 1,  nb10 );
                        std::vector<int64_t> ldda( 1, lda10 );
                        std::vector<int64_t> lddb( 1, ldb10 );
                        std::vector<int64_t> lddc( 1, ldc10 );
                        for (size_t j = 0; j < c_array10j.size(); ++j) {
                            blas::gemm(
                                layout, opA, opB,
                                mb10, nb10, kb,
                                alpha,       a_array10j[ j ], lda10,
                                             b_array10j[ j ], ldb10,
                                beta1_[ 0 ], c_array10j[ j ], ldc10,
                                *queue );
                            beta1_[ 0 ] = one;
                        }
                    }

                    if (c_array01.size() > 0) {
                        std::vector<int64_t>    m( 1,  mb01 );
                        std::vector<int64_t>    n( 1,  nb01 );
                        std::vector<int64_t> ldda( 1, lda01 );
                        std::vector<int64_t> lddb( 1, ldb01 );
                        std::vector<int64_t> lddc( 1, ldc01 );
                        blas::batch::gemm(
                            layout, opA_, opB_,
                            m, n, k1,
                            alpha_, a_array01, ldda,
                                    b_array01, lddb,
                            beta0_,  c_array01, lddc,
                            c_array01.size(), info, *queue );
                    }

                    if (c_array11 != nullptr) {
                        blas::gemm(
                            layout, opA, opB,
                            mb11, nb11, kb1,
                            alpha,       a_array11, lda11,
                                         b_array11, ldb11,
                            beta1_[ 0 ], c_array11, ldc11,
                            *queue );
                    }

                    queue->sync();
                }

                if (tile_release_strategy == TileReleaseStrategy::Internal
                    || tile_release_strategy == TileReleaseStrategy::All) {
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        for (int64_t j = 0; j < A.nt(); ++j) {
                            if (A.tileIsLocal( i, j )) {
                                if (device == A.tileDevice( i, j )) {
                                    // erase tmp local and remote device tiles;
                                    B.tileRelease( j, 0, device );
                                    // decrement life for remote tiles
                                    B.tileTick( j, 0 );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (err)
        slate_error( std::to_string( err ) );
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
template

// ----------------------------------------
// Devices instantiation
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
