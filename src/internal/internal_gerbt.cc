// Copyright (c) 2020-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/types.hh"
#include "internal/internal.hh"
#include "internal/Tile_gerbt.hh"


namespace slate {

namespace internal {

//------------------------------------------------------------------------------
/// Applies a single butterfly matrix to each side of A.  The matrices are
/// divided into the submatrices along the halfs of the butterfly matrices.
///
/// @ingroup gesv_internal
///
template<typename scalar_t>
void gerbt(Matrix<scalar_t> A11,
           Matrix<scalar_t> A12,
           Matrix<scalar_t> A21,
           Matrix<scalar_t> A22,
           Matrix<scalar_t> U1,
           Matrix<scalar_t> U2,
           Matrix<scalar_t> V1,
           Matrix<scalar_t> V2)
{
    // Assuming U and V have same structure as the A's where they're multiplied

    slate_assert(A11.mt() >= A22.mt());
    slate_assert(A11.nt() >= A22.nt());

    slate_assert(A11.mt() == A12.mt());
    slate_assert(A22.nt() == A12.nt());
    slate_assert(A22.mt() == A21.mt());
    slate_assert(A11.nt() == A21.nt());

    const int64_t mt = A22.mt();
    const int64_t nt = A22.nt();
    const int64_t mt_full = A11.mt();
    const int64_t nt_full = A11.nt();

    // Used to manage OpenMP task dependencies
    std::vector<uint8_t> task_vect(mt_full*nt_full);
    uint8_t* task = task_vect.data();
    SLATE_UNUSED( task ); // Used only by OpenMP

    std::vector<MPI_Request> requests;
    for (int64_t ii = 0; ii < mt_full; ++ii) {
        for (int64_t jj = 0; jj < nt_full; ++jj) {
            const int64_t tag = 4*(ii*nt_full + jj);
            MPI_Request r;
            if (A11.tileIsLocal(ii, jj)) {
                if (jj < nt) {
                    A12.tileIrecv( ii, jj, A12.tileRank(ii, jj),
                                  Layout::ColMajor, tag+1, &r );
                    if (r != MPI_REQUEST_NULL) {
                        requests.push_back(r);
                    }
                }
                if (ii < mt) {
                    A21.tileIrecv( ii, jj, A21.tileRank(ii, jj),
                                  Layout::ColMajor, tag+2, &r );
                    if (r != MPI_REQUEST_NULL) {
                        requests.push_back(r);
                    }
                }
                if (ii < mt && jj < nt) {
                    A22.tileIrecv( ii, jj, A22.tileRank(ii, jj),
                                  Layout::ColMajor, tag+3, &r );
                    if (r != MPI_REQUEST_NULL) {
                        requests.push_back(r);
                    }
                }
            }
            else {
                const int64_t compute_rank = A11.tileRank(ii, jj);
                // Don't need to keep the requests since we don't touch the tile
                // until receiving the finished data
                if (jj < nt && A12.tileIsLocal(ii, jj)) {
                    A12.tileIsend( ii, jj, compute_rank, tag+1, &r );
                    MPI_Request_free(&r);
                }
                if (ii < mt && A21.tileIsLocal(ii, jj)) {
                    A21.tileIsend( ii, jj, compute_rank, tag+2, &r );
                    MPI_Request_free(&r);
                }
                if (ii < mt && jj < nt && A22.tileIsLocal(ii, jj)) {
                    A22.tileIsend( ii, jj, compute_rank, tag+3, &r );
                    MPI_Request_free(&r);
                }
            }
        }
    }
    slate_mpi_call(MPI_Waitall(requests.size(), requests.data(),
                               MPI_STATUSES_IGNORE));
    requests.clear();

    for (int64_t ii = 0; ii < mt_full; ++ii) {
        for (int64_t jj = 0; jj < nt_full; ++jj) {
            if (A11.tileIsLocal(ii, jj)) {

                #pragma omp task shared(A11, A12, A21, A22, U1, U2, V1, V2) \
                                 firstprivate(ii, jj) priority(1) \
                                 depend(inout:task[ii*nt_full + jj])
                {
                    scalar_t dummy;

                    A11.tileGetForWriting(ii, jj, LayoutConvert::None);
                    U1.tileGetForReading(ii, 0, LayoutConvert::None);
                    V1.tileGetForReading(jj, 0, LayoutConvert::None);
                    Tile<scalar_t> a11 = A11(ii, jj);
                    Tile<scalar_t> u1 = U1(ii, 0);
                    Tile<scalar_t> v1 = V1(jj, 0);

                    Tile<scalar_t> a12;
                    Tile<scalar_t> a21;
                    Tile<scalar_t> a22;
                    Tile<scalar_t> u2;
                    Tile<scalar_t> v2;

                    if (ii < mt) {
                        A21.tileGetForWriting(ii, jj, LayoutConvert::None);
                        U2.tileGetForWriting(ii, 0, LayoutConvert::None);
                        a21 = A21(ii, jj);
                        u2 = U2(ii, 0);
                    }
                    else {
                        a21 = Tile<scalar_t>(0, a11.nb(), &dummy, 0, HostNum,
                                             TileKind::SlateOwned, Layout::ColMajor);
                        u2  = Tile<scalar_t>(0, u1.nb(), &dummy, 0, HostNum,
                                             TileKind::SlateOwned, Layout::ColMajor);
                    }

                    if (jj < nt) {
                        A12.tileGetForWriting(ii, jj, LayoutConvert::None);
                        V2.tileGetForWriting(jj, 0, LayoutConvert::None);
                        a12 = A12(ii, jj);
                        v2 = V2(jj, 0);
                    }
                    else {
                        a12 = Tile<scalar_t>(a11.mb(), 0, &dummy, a11.mb(), HostNum,
                                             TileKind::SlateOwned, Layout::ColMajor);
                        v2  = Tile<scalar_t>(0, v1.nb(), &dummy, 0, HostNum,
                                             TileKind::SlateOwned, Layout::ColMajor);
                    }

                    if (ii < mt && jj < nt) {
                        A22.tileGetForWriting(ii, jj, LayoutConvert::None);
                        a22 = A22(ii, jj);
                    }
                    else {
                        a22 = Tile<scalar_t>(a21.mb(), a12.nb(), &dummy, a21.mb(), HostNum,
                                             TileKind::SlateOwned, Layout::ColMajor);
                    }
                    tile::gerbt( a11, a12, a21, a22, u1, u2, v1, v2 );
                }
            }
        }
    }

    for (int64_t ii = 0; ii < mt_full; ++ii) {
        for (int64_t jj = 0; jj < nt_full; ++jj) {
            MPI_Request r;
            const int64_t tag = 4*(ii*nt_full + jj);
            if (A11.tileIsLocal(ii, jj)) {
                // Use undefered task as a selective barrier
                // TODO change to taskwait depend(...) once OpenMP 5.0 supported
                #pragma omp task if(false) depend(in:task[ii*nt_full + jj])
                {}
                if (jj < nt) {
                    A12.tileIsend( ii, jj, A12.tileRank(ii, jj), tag+1, &r );
                    if (r != MPI_REQUEST_NULL) {
                        requests.push_back(r);
                    }
                }
                if (ii < mt) {
                    A21.tileIsend( ii, jj, A21.tileRank(ii, jj), tag+2, &r );
                    if (r != MPI_REQUEST_NULL) {
                        requests.push_back(r);
                    }
                }
                if (ii < mt && jj < nt) {
                    A22.tileIsend( ii, jj, A22.tileRank(ii, jj), tag+3, &r );
                    if (r != MPI_REQUEST_NULL) {
                        requests.push_back(r);
                    }
                }
            }
            else {
                const int64_t compute_rank = A11.tileRank(ii, jj);
                if (jj < nt && A12.tileIsLocal(ii, jj)) {
                    A12.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                   tag+1, &r );
                    requests.push_back(r);
                }
                if (ii < mt && A21.tileIsLocal(ii, jj)) {
                    A21.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                   tag+2, &r );
                    requests.push_back(r);
                }
                if (ii < mt && jj < nt && A22.tileIsLocal(ii, jj)) {
                    A22.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                   tag+3, &r );
                    requests.push_back(r);
                }
            }
        }
    }

    slate_mpi_call(MPI_Waitall(requests.size(), requests.data(),
                               MPI_STATUSES_IGNORE));
    #pragma omp taskwait

    A12.releaseRemoteWorkspace();
    A21.releaseRemoteWorkspace();
    A22.releaseRemoteWorkspace();
}

template
void gerbt(Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>);

template
void gerbt(Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>);

template
void gerbt(Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>);

template
void gerbt(Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>);

//------------------------------------------------------------------------------
/// Applies a single butterfly matrix to one side of B.  The matrices are
/// divided into the submatrices along the half of the butterfly matrix.
///
/// @ingroup gesv_internal
///
template<typename scalar_t>
void gerbt(Side side,
           Op trans,
           Matrix<scalar_t> B1,
           Matrix<scalar_t> B2,
           Matrix<scalar_t> U1,
           Matrix<scalar_t> U2)
{
    slate_assert(B1.mt() >= B2.mt());
    slate_assert(B1.nt() >= B2.nt());
    const int64_t mt      = B2.mt();
    const int64_t mt_full = B1.mt();
    const int64_t nt      = B2.nt();
    const int64_t nt_full = B1.nt();
    const bool leftp = side == Side::Left;
    const bool transp = trans == Op::Trans;

    // Used to manage OpenMP task dependencies
    std::vector<uint8_t> task_vect (mt_full*nt_full);
    uint8_t* task = task_vect.data();
    SLATE_UNUSED( task ); // Used only by OpenMP

    std::vector<MPI_Request> requests;
    for (int64_t ii = 0; ii < mt; ++ii) {
        for (int64_t jj = 0; jj < nt; ++jj) {
            const int64_t tag = ii*nt + jj;
            MPI_Request r;
            if (B1.tileIsLocal(ii, jj)) {
                B2.tileIrecv( ii, jj, B2.tileRank(ii, jj),
                              Layout::ColMajor, tag, &r );
                if (r != MPI_REQUEST_NULL) {
                    requests.push_back(r);
                }
            }
            else if (B2.tileIsLocal(ii, jj)) {
                const int64_t compute_rank = B1.tileRank(ii, jj);
                B2.tileIsend( ii, jj, compute_rank, tag, &r );
                MPI_Request_free(&r);
            }
        }
    }
    slate_mpi_call(MPI_Waitall(requests.size(), requests.data(),
                               MPI_STATUSES_IGNORE));
    requests.clear();

    for (int64_t ii = 0; ii < mt; ++ii) {
        for (int64_t jj = 0; jj < nt; ++jj) {
            if (B1.tileIsLocal(ii, jj)) {
                #pragma omp task shared(B1, B2, U1, U2) firstprivate(ii, jj) \
                                 priority(1) depend(inout:task[ii*nt_full + jj])
                {
                    B1.tileGetForWriting(ii, jj, LayoutConvert::None);
                    B2.tileGetForWriting(ii, jj, LayoutConvert::None);
                    U1.tileGetForReading(ii, 0, LayoutConvert::None);
                    U2.tileGetForReading(ii, 0, LayoutConvert::None);

                    if (leftp) {
                        if (transp) {
                            tile::gerbt_left_trans( B1( ii, jj ),
                                                    B2( ii, jj ),
                                                    U1( ii, 0  ),
                                                    U2( ii, 0  ) );
                        }
                        else {
                            tile::gerbt_left_notrans( B1( ii, jj ),
                                                      B2( ii, jj ),
                                                      U1( ii, 0  ),
                                                      U2( ii, 0  ) );
                        }
                    }
                    else {
                        if (transp) {
                            tile::gerbt_right_trans( B1( ii, jj ),
                                                     B2( ii, jj ),
                                                     U1( jj, 0  ),
                                                     U2( jj, 0  ) );
                        }
                        else {
                            tile::gerbt_right_notrans( B1( ii, jj ),
                                                       B2( ii, jj ),
                                                       U1( jj, 0  ),
                                                       U2( jj, 0  ) );
                        }
                    }
                }
            }
        }
    }
    if (leftp) {
        for (int64_t ii = mt; ii < mt_full; ++ii) {
            for (int64_t jj = 0; jj < nt; ++jj) {
                if (B1.tileIsLocal(ii, jj)) {
                    #pragma omp task shared(B1, U1) firstprivate(ii, jj) \
                                     priority(1) depend(inout:task[ii*nt_full + jj])
                    {
                        scalar_t dummy;

                        B1.tileGetForWriting(ii, jj, LayoutConvert::None);
                        U1.tileGetForReading(ii, 0, LayoutConvert::None);

                        Tile<scalar_t> b1 = B1(ii, jj);
                        Tile<scalar_t> b2 (0, b1.nb(), &dummy, 0, 0,
                                           TileKind::SlateOwned, Layout::ColMajor);
                        Tile<scalar_t> u1 = U1(ii, 0);
                        Tile<scalar_t> u2 (0, u1.nb(), &dummy, 0, 0,
                                            TileKind::SlateOwned, Layout::ColMajor);

                        if (transp) {
                            tile::gerbt_left_trans( b1, b2, u1, u2 );
                        }
                        else {
                            tile::gerbt_left_notrans( b1, b2, u1, u2 );
                        }
                    }
                }
            }
        }
    }
    else {
        for (int64_t ii = 0; ii < mt; ++ii) {
            for (int64_t jj = nt; jj < nt_full; ++jj) {
                if (B1.tileIsLocal(ii, jj)) {
                    #pragma omp task shared(B1, U1) firstprivate(ii, jj) \
                                     priority(1) depend(inout:task[ii*nt_full + jj])
                    {
                        scalar_t dummy;

                        B1.tileGetForWriting(ii, jj, LayoutConvert::None);
                        U1.tileGetForReading(ii, 0, LayoutConvert::None);

                        Tile<scalar_t> b1 = B1(ii, jj);
                        Tile<scalar_t> b2 (b1.mb(), 0, &dummy, b1.mb(), 0,
                                           TileKind::SlateOwned, Layout::ColMajor);
                        Tile<scalar_t> u1 = U1(jj, 0);
                        Tile<scalar_t> u2 (0, u1.nb(), &dummy, 0, 0,
                                            TileKind::SlateOwned, Layout::ColMajor);

                        if (transp) {
                            tile::gerbt_right_trans( b1, b2, u1, u2 );
                        }
                        else {
                            tile::gerbt_right_notrans( b1, b2, u1, u2 );
                        }
                    }
                }
            }
        }
    }

    for (int64_t ii = 0; ii < mt; ++ii) {
        for (int64_t jj = 0; jj < nt; ++jj) {
            const int64_t tag = ii*nt + jj;
            MPI_Request r = MPI_REQUEST_NULL;
            if (B1.tileIsLocal(ii, jj)) {
                // Use undefered task as a selective barrier
                #pragma omp task if(false) depend(in:task[ii*nt_full + jj])
                {}
                B2.tileIsend( ii, jj, B2.tileRank(ii, jj), tag, &r );
                if (r != MPI_REQUEST_NULL) {
                    requests.push_back(r);
                }
            }
            else if (B2.tileIsLocal(ii, jj)) {
                const int64_t compute_rank = B1.tileRank(ii, jj);
                B2.tileIrecv( ii, jj, compute_rank, Layout::ColMajor, tag, &r );
                requests.push_back(r);
            }
        }
    }

    slate_mpi_call(MPI_Waitall(requests.size(), requests.data(),
                               MPI_STATUSES_IGNORE));
    #pragma omp taskwait
    B1.releaseRemoteWorkspace();
}

template
void gerbt(Side,
           Op,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>);

template
void gerbt(Side,
           Op,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>);

template
void gerbt(Side,
           Op,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>);

template
void gerbt(Side,
           Op,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>);


} // namespace internal

} // namespace slate
