// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_geqrf.hh"
#include "internal/internal.hh"
#include "lapack.hh"
#include "lapack/device.hh"
#include "blas/device.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// QR factorization of a column of tiles.
/// Dispatches to target implementations.
/// @ingroup geqrf_internal
///
template <Target target, typename scalar_t>
void geqrf(
    Matrix<scalar_t>&& A, Matrix<scalar_t>&& T,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority)
{
    geqrf(internal::TargetType<target>(),
          A, T, dwork_array, work_size,
          ib, max_panel_threads, priority);
}

//------------------------------------------------------------------------------
/// QR factorization of a column of tiles, HostTask implementation.
/// @ingroup geqrf_internal
///
template <typename scalar_t>
void geqrf(
    internal::TargetType<Target::HostTask>,
    Matrix<scalar_t>& A, Matrix<scalar_t>& T,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority)
{
    using real_t = blas::real_type<scalar_t>;

    assert(A.nt() == 1);

    // Move the panel to the host.
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            #pragma omp task slate_omp_default_none \
                shared( A ) firstprivate( i ) priority( priority )
            {
                A.tileGetForWriting(i, 0, LayoutConvert::ColMajor);
            }
        }
    }

    // todo: What about coherency protocol for T?
    // Should be invalidated in device memory if exists.

    // Build lists of local tiles and indices.
    std::vector< Tile<scalar_t> > tiles;
    std::vector<int64_t> tile_indices;

    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            tiles.push_back(A(i, 0));
            tile_indices.push_back(i);
        }
    }

    // If participating in the panel factorization.
    if (tiles.size() > 0) {

        // Launch the panel tasks.
        int thread_size = max_panel_threads;
        if (int(tiles.size()) < max_panel_threads)
            thread_size = tiles.size();

        T.tileInsert(tile_indices[0], 0);
        T.tileModified(tile_indices[0], 0);
        auto T00 = T(tile_indices[0], 0);
        T00.set(0);

        ThreadBarrier thread_barrier;
        std::vector<real_t> scale(thread_size);
        std::vector<real_t> sumsq(thread_size);
        real_t xnorm;
        std::vector< std::vector<scalar_t> > W(thread_size);

        #if 1
            #pragma omp parallel slate_omp_default_none \
                num_threads(thread_size) \
                shared(thread_barrier, scale, sumsq, xnorm, W, A, T00) \
                shared(tile_indices, tiles) \
                firstprivate(ib, thread_size)
        #else
            #pragma omp taskloop slate_omp_default_none \
                num_tasks(thread_size) \
                shared(thread_barrier, scale, sumsq, xnorm, W, A, T00) \
                shared(tile_indices, tiles) \
                firstprivate(ib, thread_size)
        #endif
        {
            // Factor the panel in parallel.
            // todo: double check the size of W.
            int thread_rank = omp_get_thread_num();
            W.at(thread_rank).resize(ib*A.tileNb(0));
            geqrf(ib,
                  tiles, tile_indices, T00,
                  thread_rank, thread_size,
                  thread_barrier,
                  scale, sumsq, xnorm, W);
        }
    }
}

//------------------------------------------------------------------------------
/// QR factorization of a column of tiles, HostNest implementation.
/// @ingroup geqrf_internal
///
/// Forwarding to HostTask as there is no implementation currently.
///
template <typename scalar_t>
void geqrf(
    internal::TargetType<Target::HostNest>,
    Matrix<scalar_t>& A, Matrix<scalar_t>& T,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority)
{
    geqrf( internal::TargetType<Target::HostTask>(),
          A, T, dwork_array, work_size,
          ib, max_panel_threads, priority);
}

//------------------------------------------------------------------------------
/// QR factorization of a column of tiles, HostBatch implementation.
/// @ingroup geqrf_internal
///
/// Forwarding to HostTask as there is no implementation currently.
///
template <typename scalar_t>
void geqrf(
    internal::TargetType<Target::HostBatch>,
    Matrix<scalar_t>& A, Matrix<scalar_t>& T,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority)
{
    geqrf( internal::TargetType<Target::HostTask>(),
          A, T, dwork_array, work_size,
          ib, max_panel_threads, priority);
}

//------------------------------------------------------------------------------
/// QR factorization of a column of tiles, device implementation.
/// @ingroup geqrf_internal
///
template <typename scalar_t>
void geqrf(
    internal::TargetType<Target::Devices>,
    Matrix<scalar_t>& A, Matrix<scalar_t>& T,
    std::vector< scalar_t* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority)
{

    assert(A.nt() == 1);

    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
    using lapack::device_info_int;

    const Layout layout = Layout::ColMajor;
    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;

    int     device   = -1;
    int64_t temp_loc = 0;
    int64_t nb       = A.tileNb( 0 );
    int64_t mlocal   = 0;

    std::set<ij_tuple> A_tiles_set;
    int64_t tile_index_zero = -1;

    size_t dsize, hsize;
    char* hwork = nullptr;

    // Find local tiles and length of factorization.
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, 0 )) {
            if (tile_index_zero < 0) {
                tile_index_zero = i;
                device = A.tileDevice( i, 0 );
            }
            else {
                // Assuming devices have a 1-D distribution.
                assert( device == A.tileDevice( i, 0 ) );
            }
            A_tiles_set.insert({ i, 0 });
            mlocal += A.tileMb( i );
        }
    }

    if (device < 0) {
       return;
    }
    assert(device >= 0);

    A.tileGetForWriting( A_tiles_set, device, LayoutConvert( layout ) );

    lapack::Queue* queue = A.compute_queue( device, 0 );

    int64_t diag_len = std::min( mlocal, nb );
    int64_t size_A   = mlocal * nb;

    std::vector<scalar_t> htau( diag_len );
    scalar_t* work = dwork_array[ device ];
    scalar_t* dA   = work;

    // Copy tile memory into contiguous memory.
    temp_loc = 0;
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, 0 )) {
            Tile Ai0 = A( i, 0, device );
            blas::device_memcpy_2d<scalar_t>(
                    &dA[ temp_loc ], mlocal,
                    Ai0.data(), Ai0.stride(),
                    Ai0.mb(), nb,
                    blas::MemcpyKind::Default, *queue );
            temp_loc += Ai0.mb();
        }
    }

    // Find workspace size, used for reference that input memory is large enough.
    lapack::geqrf_work_size_bytes( mlocal, nb,
                                   dA, mlocal, &dsize, &hsize, *queue );

    size_t tot_size = size_A + diag_len + ceildiv(dsize, sizeof(scalar_t))
                      + ceildiv( sizeof(device_info_int), sizeof(scalar_t));

    assert(tot_size <= work_size);
    assert(hsize == 0);

    // Point to correct location(s) in work array.
    scalar_t* dtau  = &work[ size_A ];
    void*     dwork = &work[ size_A + diag_len ];
    device_info_int* dinfo = (device_info_int*) &work[ size_A + diag_len
                                            + ceildiv( dsize, sizeof(scalar_t) )];

    std::vector<char> hwork_vector( hsize );
    hwork = hwork_vector.data();

    // Compute QR factorization on device.
    lapack::geqrf( mlocal, nb, dA, mlocal, dtau,
                   dwork, dsize, hwork, hsize, dinfo, *queue );

    // Copy V and R back into associated tiles.
    temp_loc = 0;
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, 0 )) {
            Tile Ai0 = A( i, 0, device );
            blas::device_memcpy_2d<scalar_t>(
                    Ai0.data(), Ai0.stride(),
                    &dA[ temp_loc ], mlocal,
                    Ai0.mb(), nb,
                    blas::MemcpyKind::Default, *queue );
            temp_loc += Ai0.mb();
        }
    }

    // Copy tau to host for trmm used later.
    //
    // TODO: Would be better to have trmm allow for pointer vs scalar to avoid
    //       this copy.
    blas::device_memcpy<scalar_t>( htau.data(), dtau, diag_len,
                                   blas::MemcpyKind::Default, *queue);

    // Constructing T-(blocking)-factor.
    T.tileInsert  ( tile_index_zero, 0, device );
    T.tileModified( tile_index_zero, 0, device );

    int64_t lddt = T( tile_index_zero, 0, device ).stride();
    auto  T00 = T.at( tile_index_zero, 0, device );

    slate::device::tzset( Uplo::Upper, diag_len, diag_len,
                          zero, one, dA, mlocal, *queue );

    // TODO: HERKX would be better.
    // (V^H * V) into T
    blas::gemm( Layout::ColMajor,
                Op::ConjTrans, Op::NoTrans,
                diag_len, diag_len, mlocal,
                one,  dA, mlocal,
                      dA, mlocal,
                zero, T00.data(), lddt, *queue );

    // Copy tau onto diagonal of T.
    blas::copy( diag_len, dtau, 1, T00.data(), lddt+1, *queue );

    for (int64_t k = 0; k < diag_len; k += ib) {

        int64_t kb = std::min( diag_len-k, ib );
        // Construct diagonal kb-block of panel T.
        for (int64_t j = k+1; j < k+kb; ++j) {

            blas::trmm( blas::Layout::ColMajor,
                        Side::Left, Uplo::Upper,
                        Op::NoTrans, Diag::NonUnit,
                        j-k, 1,
                        -htau[j], &(T00.data()[k+k*lddt]), lddt,
                                  &(T00.data()[k+j*lddt]), lddt, *queue );
        }

        // Finish rectangular block of kb-panel in T.
        if (k > 0) {
            blas::trmm( Layout::ColMajor,
                        Side::Right, Uplo::Upper,
                        Op::NoTrans, Diag::NonUnit,
                        k, kb,
                        -one, &(T00.data()[k+k*lddt]), lddt,
                              &(T00.data()[k*lddt]), lddt, *queue );

            blas::trmm( Layout::ColMajor,
                        Side::Left, Uplo::Upper,
                        Op::NoTrans, Diag::NonUnit,
                        k, kb,
                        one,   T00.data(), lddt,
                             &(T00.data()[k*lddt]), lddt, *queue );
        }
    }
    queue->sync();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void geqrf<Target::HostTask, float>(
    Matrix<float>&& A, Matrix<float>&& T,
    std::vector< float* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostTask, double>(
    Matrix<double>&& A, Matrix<double>&& T,
    std::vector< double* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostTask, std::complex<float> >(
    Matrix<std::complex<float>>&& A, Matrix< std::complex<float> >&& T,
    std::vector< std::complex<float>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostTask, std::complex<double> >(
    Matrix<std::complex<double>>&& A, Matrix< std::complex<double> >&& T,
    std::vector< std::complex<double>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostNest, float>(
    Matrix<float>&& A, Matrix<float>&& T,
    std::vector< float* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostNest, double>(
    Matrix<double>&& A, Matrix<double>&& T,
    std::vector< double* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostNest, std::complex<float> >(
    Matrix<std::complex<float>>&& A, Matrix< std::complex<float> >&& T,
    std::vector< std::complex<float>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostNest, std::complex<double> >(
    Matrix<std::complex<double>>&& A, Matrix< std::complex<double> >&& T,
    std::vector< std::complex<double>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostBatch, float>(
    Matrix<float>&& A, Matrix<float>&& T,
    std::vector< float* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostBatch, double>(
    Matrix<double>&& A, Matrix<double>&& T,
    std::vector< double* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostBatch, std::complex<float> >(
    Matrix<std::complex<float>>&& A, Matrix< std::complex<float> >&& T,
    std::vector< std::complex<float>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::HostBatch, std::complex<double> >(
    Matrix<std::complex<double>>&& A, Matrix< std::complex<double> >&& T,
    std::vector< std::complex<double>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::Devices, float>(
    Matrix<float>&& A, Matrix<float>&& T,
    std::vector< float* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::Devices, double>(
    Matrix<double>&& A, Matrix<double>&& T,
    std::vector< double* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::Devices, std::complex<float> >(
    Matrix<std::complex<float>>&& A, Matrix< std::complex<float> >&& T,
    std::vector< std::complex<float>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

// ----------------------------------------
template
void geqrf<Target::Devices, std::complex<double> >(
    Matrix<std::complex<double>>&& A, Matrix< std::complex<double> >&& T,
    std::vector< std::complex<double>* > dwork_array, size_t work_size,
    int64_t ib, int max_panel_threads, int priority);

} // namespace internal
} // namespace slate
