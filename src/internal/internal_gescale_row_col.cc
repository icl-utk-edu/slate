// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/device.hh"
#include "internal/internal_batch.hh"
#include "internal/internal.hh"
#include "internal/DevVector.hh"
#include "slate/internal/util.hh"
#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "tile/scale_row_col.hh"

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
/// Apply row or column scaling, or both, to a Matrix.
/// TODO handle transpose A case
/// Host OpenMP task implementation.
/// @ingroup scale_internal
///
template <typename scalar_t, typename scalar_t2>
void scale_row_col(
    TargetType<Target::HostTask>,
    Equed equed,
    std::vector< scalar_t2 > const& R,
    std::vector< scalar_t2 > const& C,
    Matrix<scalar_t>& A )
{
    // trace::Block trace_block("scale");
    int64_t ii = 0;
    #pragma omp taskgroup
    for (int64_t i = 0; i < A.mt(); ++i) {
        int64_t jj = 0;
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task slate_omp_default_none \
                    shared( R, C, A ) \
                    firstprivate( equed, i, j, ii, jj )
                {
                    A.tileGetForWriting( i, j, LayoutConvert::None );
                    tile::scale_row_col( equed, &R[ii], &C[jj], A(i, j) );
                }
            }
            jj += A.tileNb( j );
        }
        ii += A.tileMb( i );
    }
}

//------------------------------------------------------------------------------
/// Apply row or column scaling, or both, to a Matrix.
/// TODO handle transpose A case
/// GPU device implementation.
/// @ingroup scale_internal
///
template <typename scalar_t, typename scalar_t2>
void scale_row_col(
    TargetType<Target::Devices>,
    Equed equed,
    std::vector< scalar_t2 > const& R,
    std::vector< scalar_t2 > const& C,
    Matrix<scalar_t>& A )
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    int64_t mt = A.mt();
    int64_t nt = A.nt();

    // Find ranges of matching mb's.
    std::vector< int64_t > irange, range_ioffset;
    {
        int64_t last_mb = -1;
        int64_t ioffset = 0;
        for (int64_t i = 0; i < mt; ++i) {
            int64_t mb = A.tileMb( i );
            if (mb != last_mb) {
                last_mb = mb;
                irange.push_back( i );
                range_ioffset.push_back( ioffset );
                ioffset += mb;
            }
        }
        irange.push_back( mt );
    }

    // Find ranges of matching nb's.
    std::vector< int64_t > jrange, range_joffset;
    {
        int last_nb = -1;
        int64_t joffset = 0;
        for (int64_t j = 0; j < nt; ++j) {
            int64_t nb = A.tileNb( j );
            if (nb != last_nb) {
                last_nb = nb;
                jrange.push_back( j );
                range_joffset.push_back( joffset );
                joffset += nb;
            }
        }
        jrange.push_back( nt );
    }

    #pragma omp taskgroup
    for (int device = 0; device < A.num_devices(); ++device) {
        #pragma omp task shared( R, C, A, irange, jrange, range_ioffset, range_joffset ) \
            firstprivate( equed, device )
        {
            bool want_row = equed == Equed::Both || equed == Equed::Row;
            bool want_col = equed == Equed::Both || equed == Equed::Col;

            int queue_index = 0;
            blas::Queue* queue = A.compute_queue( device, queue_index );

            std::vector< scalar_t2* > r_array_host, c_array_host;
            DevVector< scalar_t2* > r_array_dev, c_array_dev;
            DevVector< scalar_t2 > dR, dC;

            if (want_row) {
                dR.resize( R.size(), device, *queue );
                blas::device_memcpy( dR.data(), R.data(), R.size(), *queue );
                r_array_host.resize( A.batchArraySize() );
                r_array_dev .resize( A.batchArraySize(), device, *queue );
            }
            if (want_col) {
                dC.resize( C.size(), device, *queue );
                blas::device_memcpy( dC.data(), C.data(), C.size(), *queue );
                c_array_host.resize( A.batchArraySize() );
                c_array_dev .resize( A.batchArraySize(), device, *queue );
            }

            // temporarily, convert both into same layout
            // todo: this is in-efficient, because both matrices may have same layout already
            //       and possibly wrong, because an input matrix is being altered
            // todo: best, handle directly through the CUDA kernels
            auto layout = LayoutConvert( Layout::ColMajor );
            std::set<ij_tuple> A_tiles_set;
            for (int64_t i = 0; i < A.mt(); ++i) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (A.tileIsLocal(i, j) && device == A.tileDevice(i, j)) {
                        A_tiles_set.insert( {i, j} );
                    }
                }
            }
            A.tileGetForWriting( A_tiles_set, device, layout );

            scalar_t** a_array_host = A.array_host( device, queue_index );

            int64_t batch_count = 0;
            struct Params {
                int64_t count, mb, nb, lda;
            };
            std::vector<Params> group_params;
            for (size_t jj = 0; jj < jrange.size() - 1; ++jj) {
            for (size_t ii = 0; ii < irange.size() - 1; ++ii) {
                Params group = { 0, -1, -1, -1 };
                int64_t joffset = range_joffset[ jj ];
                for (int64_t j = jrange[ jj ]; j < jrange[ jj+1 ]; ++j) {
                    int64_t ioffset = range_ioffset[ ii ];
                    for (int64_t i = irange[ ii ]; i < irange[ ii+1 ]; ++i) {
                        if (A.tileIsLocal( i, j ) && device == A.tileDevice( i, j )) {
                            auto Aij = A( i, j, device );
                            if (want_row)
                                r_array_host[ batch_count ] = &dR[ ioffset ];
                            if (want_col)
                                c_array_host[ batch_count ] = &dC[ joffset ];
                            a_array_host[ batch_count ] = Aij.data();
                            if (group.count == 0) {
                                group.mb  = Aij.mb();
                                group.nb  = Aij.nb();
                                group.lda = Aij.stride();
                            }
                            else {
                                assert( group.mb  == Aij.mb() );
                                assert( group.nb  == Aij.nb() );
                                assert( group.lda == Aij.stride() );
                            }
                            ++group.count;
                            ++batch_count;
                        }
                        ioffset += A.tileMb( i );
                    } // for i
                    joffset += A.tileNb( j );
                } // for j
                if (group.count > 0) {
                    group_params.push_back( group );
                }
            }} // for jj, ii

            scalar_t** a_array_dev = A.array_device( device, queue_index );

            blas::device_memcpy< scalar_t* >(
                &a_array_dev[ 0 ], &a_array_host[ 0 ], batch_count, *queue);

            if (want_row) {
                blas::device_memcpy< scalar_t2* >(
                    &r_array_dev[ 0 ], &r_array_host[ 0 ], batch_count, *queue);
            }

            if (want_col) {
                blas::device_memcpy< scalar_t2* >(
                    &c_array_dev[ 0 ], &c_array_host[ 0 ], batch_count, *queue);
            }

            // r_array_data, c_array_data may be null, in which case
            // they are incremented below but never dereferenced in
            // gescale_row_col_batch.
            scalar_t2** r_array_data = r_array_dev.data();
            scalar_t2** c_array_data = c_array_dev.data();

            for (size_t g = 0; g < group_params.size(); ++g) {
                int64_t group_count = group_params[ g ].count;
                device::gescale_row_col_batch(
                        equed,
                        group_params[ g ].mb, group_params[ g ].nb,
                        r_array_data, c_array_data,
                        a_array_dev, group_params[ g ].lda,
                        group_count, *queue);
                r_array_data += group_count;
                c_array_data += group_count;
                a_array_dev  += group_count;
            }

            // Clear the DevVectors, freeing device memory
            r_array_dev.clear(*queue);
            c_array_dev.clear(*queue);
            dR.clear(*queue);
            dC.clear(*queue);

            queue->sync();
        }
    }
}

//------------------------------------------------------------------------------
/// Apply row or column scaling, or both, to a Matrix.
/// Dispatches to target implementations.
/// @ingroup scale_internal
///
template <Target target, typename scalar_t, typename scalar_t2>
void scale_row_col(
    Equed equed,
    std::vector< scalar_t2 > const& R,
    std::vector< scalar_t2 > const& C,
    Matrix<scalar_t>&& A )
{
    scale_row_col( TargetType<target>(),
                   equed, R, C, A );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void scale_row_col<Target::HostTask, float>(
    Equed equed,
    std::vector<float> const& R,
    std::vector<float> const& C,
    Matrix<float>&& A );

template
void scale_row_col<Target::Devices, float>(
    Equed equed,
    std::vector<float> const& R,
    std::vector<float> const& C,
    Matrix<float>&& A );

// ----------------------------------------
template
void scale_row_col<Target::HostTask, double>(
    Equed equed,
    std::vector<double> const& R,
    std::vector<double> const& C,
    Matrix<double>&& A );

template
void scale_row_col<Target::Devices, double>(
    Equed equed,
    std::vector<double> const& R,
    std::vector<double> const& C,
    Matrix<double>&& A );

// ----------------------------------------
// real R, C
template
void scale_row_col< Target::HostTask, std::complex<float> >(
    Equed equed,
    std::vector<float> const& R,
    std::vector<float> const& C,
    Matrix< std::complex<float> >&& A );

template
void scale_row_col< Target::Devices, std::complex<float> >(
    Equed equed,
    std::vector<float> const& R,
    std::vector<float> const& C,
    Matrix< std::complex<float> >&& A );

// ----------------------------------------
template
void scale_row_col< Target::HostTask, std::complex<double> >(
    Equed equed,
    std::vector<double> const& R,
    std::vector<double> const& C,
    Matrix< std::complex<double> >&& A );

template
void scale_row_col< Target::Devices, std::complex<double> >(
    Equed equed,
    std::vector<double> const& R,
    std::vector<double> const& C,
    Matrix< std::complex<double> >&& A );

// ----------------------------------------
// complex R, C
template
void scale_row_col< Target::HostTask, std::complex<float> >(
    Equed equed,
    std::vector< std::complex<float> > const& R,
    std::vector< std::complex<float> > const& C,
    Matrix< std::complex<float> >&& A );

template
void scale_row_col< Target::Devices, std::complex<float> >(
    Equed equed,
    std::vector< std::complex<float> > const& R,
    std::vector< std::complex<float> > const& C,
    Matrix< std::complex<float> >&& A );

// ----------------------------------------
template
void scale_row_col< Target::HostTask, std::complex<double> >(
    Equed equed,
    std::vector< std::complex<double> > const& R,
    std::vector< std::complex<double> > const& C,
    Matrix< std::complex<double> >&& A );

template
void scale_row_col< Target::Devices, std::complex<double> >(
    Equed equed,
    std::vector< std::complex<double> > const& R,
    std::vector< std::complex<double> > const& C,
    Matrix< std::complex<double> >&& A );

} // namespace internal
} // namespace slate
