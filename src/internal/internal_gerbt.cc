// Copyright (c) 2020-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/types.hh"
#include "internal/internal.hh"
#include "internal/Tile_gerbt.hh"


namespace slate {

namespace internal {

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

    std::vector<MPI_Request> requests;
    #pragma omp task shared(A11, A12, A21, A22, U1, U2, V1, V2) \
                     shared(requests) priority(2) depend(out:requests)
    {
        for (int64_t ii = 0; ii < mt; ++ii) {
            for (int64_t jj = 0; jj < nt; ++jj) {
                if (! A11.tileIsLocal(ii, jj)) {
                    const int64_t tag = 4*(ii*nt_full + jj);
                    const int64_t compute_rank = A11.tileRank(ii, jj);
                    if (A12.tileIsLocal(ii, jj)) {
                        MPI_Request r;
                        A12.tileIsend( ii, jj, compute_rank, tag+1, &r );
                        MPI_Request_free(&r);

                        A12.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                       tag+1, &r );
                        requests.push_back(r);
                    }
                    if (A21.tileIsLocal(ii, jj)) {
                        MPI_Request r;
                        A21.tileIsend( ii, jj, compute_rank, tag+2, &r );
                        MPI_Request_free(&r);

                        A21.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                       tag+2, &r );
                        requests.push_back(r);
                    }
                    if (A22.tileIsLocal(ii, jj)) {
                        MPI_Request r;
                        A22.tileIsend(ii, jj, compute_rank, tag+3, &r);
                        MPI_Request_free(&r);

                        A22.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                       tag+3, &r );
                        requests.push_back(r);
                    }
                }
            }
        }

        for (int64_t ii = 0; ii < mt; ++ii) {
            for (int64_t jj = nt; jj < nt_full; ++jj) {
                if (! A11.tileIsLocal(ii, jj) && A21.tileIsLocal(ii, jj)) {
                    const int64_t tag = 4*(ii*nt_full + jj);
                    const int64_t compute_rank = A11.tileRank(ii, jj);

                    MPI_Request r;
                    A21.tileIsend(ii, jj, compute_rank, tag+2, &r);
                    MPI_Request_free(&r);

                    A21.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                   tag+2, &r );
                    requests.push_back(r);
                }
            }
        }

        for (int64_t ii = mt; ii < mt_full; ++ii) {
            for (int64_t jj = 0; jj < nt; ++jj) {
                if (! A11.tileIsLocal(ii, jj) && A12.tileIsLocal(ii, jj)) {
                    const int64_t tag = 4*(ii*nt_full + jj);
                    const int64_t compute_rank = A11.tileRank(ii, jj);

                    MPI_Request r;
                    A12.tileIsend(ii, jj, compute_rank, tag+1, &r);
                    MPI_Request_free(&r);

                    A12.tileIrecv( ii, jj, compute_rank, Layout::ColMajor,
                                   tag+1, &r );
                    requests.push_back(r);
                }
            }
        }
    }

    for (int64_t ii = 0; ii < mt; ++ii) {
        for (int64_t jj = 0; jj < nt; ++jj) {
            if (A11.tileIsLocal(ii, jj)) {
                #pragma omp task shared(A11, A12, A21, A22, U1, U2, V1, V2) \
                                 firstprivate(ii, jj) priority(1)
                {
                    const int64_t tag = 4*(ii*nt_full + jj);
                    A12.tileRecv( ii, jj, A12.tileRank(ii, jj),
                                  Layout::ColMajor, tag+1 );
                    A21.tileRecv( ii, jj, A21.tileRank(ii, jj),
                                  Layout::ColMajor, tag+2 );
                    A22.tileRecv( ii, jj, A22.tileRank(ii, jj),
                                  Layout::ColMajor, tag+3 );

                    gerbt( A11(ii, jj), A12(ii, jj), A21(ii, jj), A22(ii, jj),
                           U1(ii, 0), U2(ii, 0), V1(jj, 0), V2(jj, 0) );

                    A12.tileSend( ii, jj, A12.tileRank(ii, jj), tag+1 );
                    A21.tileSend( ii, jj, A21.tileRank(ii, jj), tag+2 );
                    A22.tileSend( ii, jj, A22.tileRank(ii, jj), tag+3 );

                    A12.tileRelease(ii, jj);
                    A21.tileRelease(ii, jj);
                    A22.tileRelease(ii, jj);
                    U1.tileTick(ii, 0);
                    U2.tileTick(ii, 0);
                    V1.tileTick(jj, 0);
                    V2.tileTick(jj, 0);
                }
            }
        }
    }

    for (int64_t ii = 0; ii < mt; ++ii) {
        for (int64_t jj = nt; jj < nt_full; ++jj) {
            if (A11.tileIsLocal(ii, jj)) {
                #pragma omp task shared(A11, A21, U1, U2) firstprivate(ii, jj) \
                                 priority(1)
                {
                    scalar_t dummy;

                    const int64_t tag = 4*(ii*nt_full + jj);
                    A21.tileRecv( ii, jj, A21.tileRank(ii, jj),
                                  Layout::ColMajor, tag+2 );

                    Tile<scalar_t> a11 = A11(ii, jj);
                    Tile<scalar_t> a21 = A21(ii, jj);
                    Tile<scalar_t> a12 (a11.mb(), 0, &dummy, a11.mb(), 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> a22 (a21.mb(), 0, &dummy, a11.mb(), 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> v1 = V1(jj, 0);
                    Tile<scalar_t> v2 (0, v1.nb(), &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);

                    gerbt( a11, a12, a21, a22,
                           U1(ii, 0), U2(ii, 0), v1, v2 );

                    A21.tileSend(ii, jj, A21.tileRank(ii, jj), tag+2);
                    A21.tileRelease(ii, jj);
                    U1.tileTick(ii, 0);
                    U2.tileTick(ii, 0);
                    V1.tileTick(ii, 0);
                }
            }
        }
    }

    for (int64_t ii = mt; ii < mt_full; ++ii) {
        for (int64_t jj = 0; jj < nt; ++jj) {
            if (A11.tileIsLocal(ii, jj)) {
                #pragma omp task shared(A11, A12, V1, V2) firstprivate(ii, jj) \
                                 priority(1)
                {
                    scalar_t dummy;

                    const int64_t tag = 4*(ii*nt_full + jj);
                    A12.tileRecv( ii, jj, A12.tileRank(ii, jj),
                                  Layout::ColMajor, tag+1 );

                    Tile<scalar_t> a11 = A11(ii, jj);
                    Tile<scalar_t> a12 = A12(ii, jj);
                    Tile<scalar_t> a21 (0, a11.nb(), &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> a22 (0, a12.nb(), &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> u1 = U1(ii, 0);
                    Tile<scalar_t> u2 (0, u1.nb(), &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);

                    gerbt( a11, a12, a21, a22,
                           u1, u2, V1(jj, 0), V2(jj, 0) );

                    A12.tileSend( ii, jj, A12.tileRank(ii, jj), tag+1 );
                    A12.tileRelease(ii, jj);
                    U1.tileTick(jj, 0);
                    V1.tileTick(jj, 0);
                    V2.tileTick(jj, 0);
                }
            }
        }
    }
    for (int64_t ii = mt; ii < mt_full; ++ii) {
        for (int64_t jj = nt; jj < nt_full; ++jj) {
            if (A11.tileIsLocal(ii, jj)) {
                #pragma omp task shared(A11, A12, V1, V2) firstprivate(ii, jj) \
                                 priority(1)
                {
                    scalar_t dummy;

                    Tile<scalar_t> a11 = A11(ii, jj);
                    Tile<scalar_t> a12 (0, a11.nb(), &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> a21 (a11.mb(), 0, &dummy, a11.mb(), 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> a22 (0, 0, &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> u1 = U1(ii, 0);
                    Tile<scalar_t> u2 (0, u1.nb(), &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);
                    Tile<scalar_t> v1 = V1(jj, 0);
                    Tile<scalar_t> v2 (0, v1.nb(), &dummy, 0, 0,
                                        TileKind::UserOwned, Layout::ColMajor);

                    gerbt( a11, a12, a21, a22,
                           u1, u2, v1, v2 );

                    U1.tileTick(ii, 0);
                    V1.tileTick(jj, 0);
                }
            }
        }
    }

    #pragma omp task depend(in:requests) shared(requests)
    slate_mpi_call(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));

    #pragma omp taskwait
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

    std::vector<MPI_Request> requests;
    #pragma omp task depend(out:requests) priority(2) \
                     shared(B1, B2) shared(requests)
    for (int64_t ii = 0; ii < mt; ++ii) {
        for (int64_t jj = 0; jj < nt; ++jj) {
            if (! B1.tileIsLocal(ii, jj) && B2.tileIsLocal(ii, jj)) {
                const int64_t tag = ii*nt + jj;
                const int64_t compute_rank = B1.tileRank(ii, jj);
                MPI_Request r;
                B2.tileIsend( ii, jj, compute_rank, tag, &r );
                MPI_Request_free(&r);

                B2.tileIrecv( ii, jj, compute_rank, Layout::ColMajor, tag, &r );
                requests.push_back(r);
            }
        }
    }


    for (int64_t ii = 0; ii < mt; ++ii) {
        for (int64_t jj = 0; jj < nt; ++jj) {
            if (B1.tileIsLocal(ii, jj)) {
                #pragma omp task shared(B1, B2, U1, U2) firstprivate(ii, jj) \
                                 priority(1)
                {
                    const int64_t tag = ii*nt + jj;
                    B2.tileRecv( ii, jj, B2.tileRank(ii, jj),
                                 Layout::ColMajor, tag );

                    if (leftp) {
                        if (transp) {
                            gerbt_left_trans( B1(ii, jj),
                                              B2(ii, jj),
                                              U1(ii, 0),
                                              U2(ii, 0) );
                        }
                        else {
                            gerbt_left_notrans( B1(ii, jj),
                                                B2(ii, jj),
                                                U1(ii, 0),
                                                U2(ii, 0) );
                        }
                        U1.tileTick(ii, 0);
                        U2.tileTick(ii, 0);
                    }
                    else {
                        if (transp) {
                            gerbt_right_trans( B1(ii, jj),
                                               B2(ii, jj),
                                               U1(jj, 0),
                                               U2(jj, 0) );
                        }
                        else {
                            gerbt_right_notrans( B1(ii, jj),
                                                 B2(ii, jj),
                                                 U1(jj, 0),
                                                 U2(jj, 0) );
                        }
                        U1.tileTick(jj, 0);
                        U2.tileTick(jj, 0);
                    }
                    B2.tileSend( ii, jj, B2.tileRank(ii, jj), tag );
                    B2.tileRelease(ii, jj);
                }
            }
        }
    }
    if (leftp) {
        for (int64_t ii = mt; ii < mt_full; ++ii) {
            for (int64_t jj = 0; jj < nt; ++jj) {
                if (B1.tileIsLocal(ii, jj)) {
                    #pragma omp task shared(B1, U1) firstprivate(ii, jj) \
                                     priority(1)
                    {
                        scalar_t dummy;

                        Tile<scalar_t> b1 = B1(ii, jj);
                        Tile<scalar_t> b2 (0, b1.nb(), &dummy, 0, 0,
                                           TileKind::UserOwned, Layout::ColMajor);
                        Tile<scalar_t> u1 = U1(ii, 0);
                        Tile<scalar_t> u2 (0, u1.nb(), &dummy, 0, 0,
                                            TileKind::UserOwned, Layout::ColMajor);

                        if (transp) {
                            gerbt_left_trans( b1, b2, u1, u2 );
                        }
                        else {
                            gerbt_left_notrans( b1, b2, u1, u2 );
                        }
                        U1.tileTick(ii, 0);
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
                                     priority(1)
                    {
                        scalar_t dummy;

                        Tile<scalar_t> b1 = B1(ii, jj);
                        Tile<scalar_t> b2 (b1.mb(), 0, &dummy, b1.mb(), 0,
                                           TileKind::UserOwned, Layout::ColMajor);
                        Tile<scalar_t> u1 = U1(jj, 0);
                        Tile<scalar_t> u2 (0, u1.nb(), &dummy, 0, 0,
                                            TileKind::UserOwned, Layout::ColMajor);

                        if (transp) {
                            gerbt_right_trans( b1, b2, u1, u2 );
                        }
                        else {
                            gerbt_right_notrans( b1, b2, u1, u2 );
                        }
                        U1.tileTick(jj, 0);
                    }
                }
            }
        }
    }

    #pragma omp task depend(inout:requests) shared(requests)
    slate_mpi_call(MPI_Waitall(requests.size(), requests.data(),
                               MPI_STATUSES_IGNORE));

    #pragma omp taskwait
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
