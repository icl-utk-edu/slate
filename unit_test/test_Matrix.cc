// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/internal/util.hh"

#include "unit_test.hh"
#include "util_matrix.hh"

using slate::ceildiv;
using slate::roundup;
using slate::GridOrder;
using slate::HostNum;

namespace test {

//------------------------------------------------------------------------------
// global variables
int m, n, k, mb, nb, p, q;
int mpi_rank;
int mpi_size;
MPI_Comm mpi_comm;
int num_devices = 0;
int verbose = 0;

//==============================================================================
// Constructors

//------------------------------------------------------------------------------
/// default constructor
/// Tests Matrix(), m, n, mt, nt, op, gridinfo.
void test_Matrix_default()
{
    slate::Matrix<double> A;

    test_assert(A.m() == 0);
    test_assert(A.n() == 0);
    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    GridOrder order;
    int myp, myq, myrow, mycol;
    A.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Unknown );
    test_assert( myp == -1 );
    test_assert( myq == -1 );
    test_assert( myrow == -1 );
    test_assert( mycol == -1 );

    // todo: What is reasonable in this case? It segfaults right now.
    // auto tileMb_     = A.tileMbFunc();
    // auto tileNb_     = A.tileNbFunc();
    // auto tileRank_   = A.tileRankFunc();
    // auto tileDevice_ = A.tileDeviceFunc();
    // test_assert( tileMb_(0) == mb );
    // test_assert( tileNb_(0) == nb );
    // test_assert( tileRank_( {0, 0} ) == 0 );
    // // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    // if (num_devices > 0)
    //     test_assert( tileDevice_( {0, 0} ) == 0 );
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor, both square and rectangular tiles
/// Tests Matrix(m, n, nb, ...), m, n, mt, nt, op, gridinfo.
void test_Matrix_empty()
{
    // square tiles
    slate::Matrix<double> A(m, n, nb, p, q, mpi_comm);
    test_assert(A.m() == m);
    test_assert(A.n() == n);
    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    GridOrder order;
    int myp, myq, myrow, mycol;
    A.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Col );
    test_assert( myp == p );
    test_assert( myq == q );
    test_assert( myrow == mpi_rank % p );
    test_assert( mycol == mpi_rank / p );

    //----------
    // rectangular tiles
    slate::Matrix<double> B(m, n, mb, nb, p, q, mpi_comm);
    test_assert(B.m() == m);
    test_assert(B.n() == n);
    test_assert(B.mt() == ceildiv(m, mb));
    test_assert(B.nt() == ceildiv(n, nb));
    test_assert(B.op() == blas::Op::NoTrans);
    test_assert(B.uplo() == slate::Uplo::General);

    auto tileMb_     = B.tileMbFunc();
    auto tileNb_     = B.tileNbFunc();
    auto tileRank_   = B.tileRankFunc();
    auto tileDevice_ = B.tileDeviceFunc();
    test_assert( tileMb_(0) == mb );
    test_assert( tileNb_(0) == nb );
    test_assert( tileRank_( {0, 0} ) == 0 );
    // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    if (num_devices > 0)
        test_assert( tileDevice_( {0, 0} ) == 0 );

    // Construct Bf same as B, but float instead of double.
    slate::Matrix<float> Bf(m, n, tileMb_, tileNb_, tileRank_, tileDevice_,
                            mpi_comm);
    test_assert(Bf.m() == m);
    test_assert(Bf.n() == n);
    test_assert(Bf.tileMb(0) == mb);
    test_assert(Bf.tileNb(0) == nb);
    test_assert(Bf.tileRank( 0, 0 ) == 0);
    if (num_devices > 0)
        test_assert( Bf.tileDevice( 0, 0 ) == 0 );

    //----------
    // rectangular tiles, Col grid order
    slate::Matrix<double> C( m, n, mb, nb, GridOrder::Col, p, q, mpi_comm );
    test_assert( C.m() == m );
    test_assert( C.n() == n );
    test_assert( C.mt() == ceildiv( m, mb ) );
    test_assert( C.nt() == ceildiv( n, nb ) );
    test_assert( C.op() == blas::Op::NoTrans );
    test_assert( C.uplo() == slate::Uplo::General );

    C.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Col );
    test_assert( myp == p );
    test_assert( myq == q );
    test_assert( myrow == mpi_rank % p );  // col major
    test_assert( mycol == mpi_rank / p );

    // Check col major.
    int rank = 0;
    for (int j = 0; j < q; ++j) {
        for (int i = 0; i < p; ++i) {
            test_assert( C.tileRank( i, j ) == rank );
            rank += 1;
        }
    }

    //----------
    // rectangular tiles, Row grid order
    slate::Matrix<double> D( m, n, mb, nb, GridOrder::Row, p, q, mpi_comm );
    test_assert( D.m() == m );
    test_assert( D.n() == n );
    test_assert( D.mt() == ceildiv( m, mb ) );
    test_assert( D.nt() == ceildiv( n, nb ) );
    test_assert( D.op() == blas::Op::NoTrans );
    test_assert( D.uplo() == slate::Uplo::General );

    D.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Row );
    test_assert( myp == p );
    test_assert( myq == q );
    test_assert( myrow == mpi_rank / q );  // row major
    test_assert( mycol == mpi_rank % q );

    // Check row major.
    rank = 0;
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < q; ++j) {
            test_assert( D.tileRank( i, j ) == rank );
            rank += 1;
        }
    }
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor, both square and rectangular tiles,
/// using lambda functions for tileMb, tileNb, tileRank, tileDevice.
/// Tests Matrix(m, n, tileMb, ...), m, n, mt, nt, op, gridinfo.
void test_Matrix_lambda()
{
    int mb_ = mb;  // local copy to capture
    std::function< int64_t (int64_t i) >
    tileMb = [mb_](int64_t i)
    {
        return (i % 2 == 0 ? mb_/2 : mb_);
    };

    int nb_ = nb;  // local copy to capture
    std::function< int64_t (int64_t j) >
    tileNb = [nb_](int64_t j)
    {
        return (j % 2 == 0 ? 2*nb_ : nb_);
    };

    // 1D block column cyclic
    int p_ = p;  // local copy to capture
    std::function< int (std::tuple<int64_t, int64_t> ij) >
    tileRank = [p_](std::tuple<int64_t, int64_t> ij)
    {
        int64_t i = std::get<0>(ij);
        int64_t j = std::get<1>(ij);
        return int(i%p_ + j*p_);
    };

    // 1D block row cyclic
    int num_devices_ = num_devices;  // local copy to capture
    std::function< int (std::tuple<int64_t, int64_t> ij) >
    tileDevice = [num_devices_](std::tuple<int64_t, int64_t> ij)
    {
        int64_t i = std::get<0>(ij);
        return int(i)%num_devices_;
    };

    // ----------
    slate::Matrix<double> A(m, n, tileMb, tileNb, tileRank, tileDevice, mpi_comm);

    // verify mt, tileMb(i), and sum tileMb(i) == m
    int mt = A.mt();
    int ii = 0;
    for (int i = 0; i < mt; ++i) {
        test_assert( A.tileMb(i) == blas::min( tileMb(i), m - ii ) );
        ii += A.tileMb(i);
    }
    test_assert( ii == m );

    // verify nt, tileNb(i), and sum tileNb(i) == n
    int nt = A.nt();
    int jj = 0;
    for (int j = 0; j < nt; ++j) {
        test_assert( A.tileNb(j) == blas::min( tileNb(j), n - jj ) );
        jj += A.tileNb(j);
    }
    test_assert( jj == n );

    test_assert(A.m() == m);
    test_assert(A.n() == n);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    // SLATE doesn't know distribution.
    GridOrder order;
    int myp, myq, myrow, mycol;
    A.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Unknown );
    test_assert( myp == -1 );
    test_assert( myq == -1 );
    test_assert( myrow == -1 );
    test_assert( mycol == -1 );

    auto tileMb_     = A.tileMbFunc();
    auto tileNb_     = A.tileNbFunc();
    auto tileRank_   = A.tileRankFunc();
    auto tileDevice_ = A.tileDeviceFunc();
    test_assert( tileMb_(0) == tileMb(0) );
    test_assert( tileNb_(0) == tileNb(0) );
    test_assert( tileRank_( {0, 0} ) == tileRank( {0, 0} ) );
    // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    if (num_devices > 0)
        test_assert( tileDevice_( {0, 0} ) == tileDevice( {0, 0} ) );
}

//------------------------------------------------------------------------------
/// fromLAPACK
/// Test Matrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
void test_Matrix_fromLAPACK()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_lapack(A, i, j, nb, m, n, Ad.data(), lda);
        }
    }
}

//------------------------------------------------------------------------------
/// fromLAPACK with rectangular tiles
/// Test Matrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
void test_Matrix_fromLAPACK_rect()
{
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );

    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    test_assert(A.mt() == ceildiv(m, mb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_lapack(A, i, j, mb, nb, m, n, Ad.data(), lda);
        }
    }
}

//------------------------------------------------------------------------------
/// fromScaLAPACK
/// Test Matrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
void test_Matrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);


    GridOrder order;
    int myp, myq, myrow, mycol;
    A.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Col );
    test_assert( myp == p );
    test_assert( myq == q );
    test_assert( myrow == mpi_rank % p );
    test_assert( mycol == mpi_rank / p );

    auto tileMb_     = A.tileMbFunc();
    auto tileNb_     = A.tileNbFunc();
    auto tileRank_   = A.tileRankFunc();
    auto tileDevice_ = A.tileDeviceFunc();
    test_assert( tileMb_(0) == nb );  // square
    test_assert( tileNb_(0) == nb );
    test_assert( tileRank_  ( {0, 0} ) == 0 );
    // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    if (num_devices > 0)
        test_assert( tileDevice_( {0, 0} ) == 0 );

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_scalapack(A, i, j, nb, m, n, Ad.data(), lda);
        }
    }
}

//------------------------------------------------------------------------------
/// fromScaLAPACK with rectangular tiles
/// Test Matrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
void test_Matrix_fromScaLAPACK_rect()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, mb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    GridOrder order;
    int myp, myq, myrow, mycol;
    A.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Col );
    test_assert( myp == p );
    test_assert( myq == q );
    test_assert( myrow == mpi_rank % p );
    test_assert( mycol == mpi_rank / p );

    auto tileMb_     = A.tileMbFunc();
    auto tileNb_     = A.tileNbFunc();
    auto tileRank_   = A.tileRankFunc();
    auto tileDevice_ = A.tileDeviceFunc();
    test_assert( tileMb_(0) == mb );  // rect
    test_assert( tileNb_(0) == nb );
    test_assert( tileRank_  ( {0, 0} ) == 0 );
    // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    if (num_devices > 0)
        test_assert( tileDevice_( {0, 0} ) == 0 );

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_scalapack(A, i, j, mb, nb, m, n, Ad.data(), lda);
        }
    }
}

//------------------------------------------------------------------------------
/// fromDevices
/// Test Matrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
void test_Matrix_fromDevices()
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    // device specific queues
    std::vector< blas::Queue* > dev_queues(num_devices);
    for (int dev = 0; dev < num_devices; ++dev)
        dev_queues[dev] = new blas::Queue(dev, 0);

    double** Aarray = new double*[ num_devices ];
    for (int dev = 0; dev < num_devices; ++dev) {
        int ntiles_local2, ntiles_dev, n_dev;
        get_cyclic_dimensions(num_devices, dev, n_local, nb,
                              ntiles_local2, ntiles_dev, n_dev);
        assert(ntiles_local == ntiles_local2);

        // blas::device_malloc returns null if len = 0, so make it at least 1.
        int64_t len = std::max(lda * n_dev, 1);
        Aarray[dev] = blas::device_malloc<double>(len, *dev_queues[dev]);
        assert(Aarray[dev] != nullptr);
    }

    auto A = slate::Matrix<double>::fromDevices(
        m, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_device(A, i, j, nb, m, n, Aarray, lda);
        }
    }

    for (int dev = 0; dev < num_devices; ++dev) {
        blas::device_free(Aarray[dev], *dev_queues[dev]);
    }
    delete[] Aarray;

    // free the device specific queues
    for (int dev = 0; dev < num_devices; ++dev)
        delete dev_queues[dev];
}

//==============================================================================
// Methods

//------------------------------------------------------------------------------
/// emptyLike with rectangular tiles
void test_Matrix_emptyLike()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, mb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    auto B = A.template emptyLike<float>();

    test_assert(B.m()    == A.m());
    test_assert(B.n()    == A.n());
    test_assert(B.mt()   == A.mt());
    test_assert(B.nt()   == A.nt());
    test_assert(B.op()   == A.op());
    test_assert(B.uplo() == A.uplo());

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            test_assert( A.tileIsLocal(i, j) == B.tileIsLocal(i, j) );
            test_assert( A.tileMb(i) == B.tileMb(i) );
            test_assert( A.tileNb(j) == B.tileNb(j) );
            test_assert_throw_std( B(i, j) );  // tiles don't exist
        }
    }

    // ----------
    auto Asub = A.sub( 1, 3, 1, 4 );
    auto Bsub = Asub.emptyLike();

    test_assert(Bsub.m()    == Asub.m());
    test_assert(Bsub.n()    == Asub.n());
    test_assert(Bsub.mt()   == Asub.mt());
    test_assert(Bsub.nt()   == Asub.nt());
    test_assert(Bsub.op()   == Asub.op());
    test_assert(Bsub.uplo() == Asub.uplo());

    for (int j = 0; j < Asub.nt(); ++j) {
        for (int i = 0; i < Asub.mt(); ++i) {
            test_assert( Asub.tileIsLocal(i, j) == Bsub.tileIsLocal(i, j) );
            test_assert( Asub.tileMb(i) == Bsub.tileMb(i) );
            test_assert( Asub.tileNb(j) == Bsub.tileNb(j) );
            test_assert_throw_std( Bsub(i, j) );  // tiles don't exist
        }
    }

    // ----------
    auto Atrans = transpose( A );
    auto Btrans = Atrans.emptyLike();

    test_assert(Btrans.m()    == Atrans.m());
    test_assert(Btrans.n()    == Atrans.n());
    test_assert(Btrans.mt()   == Atrans.mt());
    test_assert(Btrans.nt()   == Atrans.nt());
    test_assert(Btrans.op()   == Atrans.op());
    test_assert(Btrans.uplo() == Atrans.uplo());

    for (int j = 0; j < Atrans.nt(); ++j) {
        for (int i = 0; i < Atrans.mt(); ++i) {
            test_assert( Atrans.tileIsLocal(i, j) == Btrans.tileIsLocal(i, j) );
            test_assert( Atrans.tileMb(i) == Btrans.tileMb(i) );
            test_assert( Atrans.tileNb(j) == Btrans.tileNb(j) );
            test_assert_throw_std( Btrans(i, j) );  // tiles don't exist
        }
    }

    // ----------
    auto Asub_trans = transpose( Asub );
    auto Bsub_trans = Asub_trans.emptyLike();

    test_assert(Bsub_trans.m()    == Asub_trans.m());
    test_assert(Bsub_trans.n()    == Asub_trans.n());
    test_assert(Bsub_trans.mt()   == Asub_trans.mt());
    test_assert(Bsub_trans.nt()   == Asub_trans.nt());
    test_assert(Bsub_trans.op()   == Asub_trans.op());
    test_assert(Bsub_trans.uplo() == Asub_trans.uplo());

    for (int j = 0; j < Asub_trans.nt(); ++j) {
        for (int i = 0; i < Asub_trans.mt(); ++i) {
            test_assert( Asub_trans.tileIsLocal(i, j) == Bsub_trans.tileIsLocal(i, j) );
            test_assert( Asub_trans.tileMb(i) == Bsub_trans.tileMb(i) );
            test_assert( Asub_trans.tileNb(j) == Bsub_trans.tileNb(j) );
            test_assert_throw_std( Bsub_trans(i, j) );  // tiles don't exist
        }
    }
}

//------------------------------------------------------------------------------
/// emptyLike with mb, nb overriding size.
void test_Matrix_emptyLikeMbNb()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, mb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    auto Asub = A.sub( 1, 3, 1, 4 );
    auto Asub_trans = transpose( Asub );
    if (verbose) {
        printf( "A  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                llong( Asub.m() ), llong( Asub.mt() ),
                llong( Asub.n() ), llong( Asub.nt() ),
                llong( Asub.tileMb(0) ), llong( Asub.tileNb(0) ) );
        printf( "AT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                llong( Asub_trans.m() ), llong( Asub_trans.mt() ),
                llong( Asub_trans.n() ), llong( Asub_trans.nt() ),
                llong( Asub_trans.tileMb(0) ), llong( Asub_trans.tileNb(0) ) );
    }

    for (int mb2: std::vector<int>({ 0, 7 })) {
        for (int nb2: std::vector<int>({ 0, 5 })) {
            // ----- no trans
            auto B = Asub.emptyLike( mb2, nb2 );

            if (verbose) {
                printf( "\n" );
                printf( "A  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub.m() ), llong( Asub.mt() ),
                        llong( Asub.n() ), llong( Asub.nt() ),
                        llong( Asub.tileMb(0) ), llong( Asub.tileNb(0) ) );
                printf( "AT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub_trans.m() ), llong( Asub_trans.mt() ),
                        llong( Asub_trans.n() ), llong( Asub_trans.nt() ),
                        llong( Asub_trans.tileMb(0) ), llong( Asub_trans.tileNb(0) ) );
                printf( "B  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld (mb2 %3d, nb2 %3d)\n",
                        llong( B.m() ), llong( B.mt() ),
                        llong( B.n() ), llong( B.nt() ),
                        llong( B.tileMb(0) ), llong( B.tileNb(0) ),
                        mb2, nb2 );
            }
            test_assert(B.m() == (mb2 == 0 ? Asub.m() : Asub.mt() * mb2));
            test_assert(B.n() == (nb2 == 0 ? Asub.n() : Asub.nt() * nb2));
            test_assert(B.mt() == Asub.mt());
            test_assert(B.nt() == Asub.nt());

            for (int j = 0; j < Asub.nt(); ++j) {
                for (int i = 0; i < Asub.mt(); ++i) {
                    test_assert( B.tileIsLocal(i, j) == Asub.tileIsLocal(i, j) );
                    test_assert( B.tileMb(i) == (mb2 == 0 ? Asub.tileMb(i) : mb2) );
                    test_assert( B.tileNb(j) == (nb2 == 0 ? Asub.tileNb(j) : nb2) );
                    test_assert_throw_std( B(i, j) );  // tiles don't exist
                }
            }

            // ----- trans
            auto BT = Asub_trans.emptyLike( mb2, nb2 );

            if (verbose) {
                printf( "\n" );
                printf( "A  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub.m() ), llong( Asub.mt() ),
                        llong( Asub.n() ), llong( Asub.nt() ),
                        llong( Asub.tileMb(0) ), llong( Asub.tileNb(0) ) );
                printf( "AT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub_trans.m() ), llong( Asub_trans.mt() ),
                        llong( Asub_trans.n() ), llong( Asub_trans.nt() ),
                        llong( Asub_trans.tileMb(0) ), llong( Asub_trans.tileNb(0) ) );
                printf( "BT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld (mb2 %3d, nb2 %3d)\n",
                        llong( BT.m() ), llong( BT.mt() ),
                        llong( BT.n() ), llong( BT.nt() ),
                        llong( BT.tileMb(0) ), llong( BT.tileNb(0) ),
                        mb2, nb2 );
            }
            test_assert(BT.m() == (mb2 == 0 ? Asub_trans.m() : Asub_trans.mt() * mb2));
            test_assert(BT.n() == (nb2 == 0 ? Asub_trans.n() : Asub_trans.nt() * nb2));
            test_assert(BT.mt() == Asub_trans.mt());
            test_assert(BT.nt() == Asub_trans.nt());

            for (int j = 0; j < Asub_trans.nt(); ++j) {
                for (int i = 0; i < Asub_trans.mt(); ++i) {
                    test_assert( BT.tileIsLocal(i, j) == Asub_trans.tileIsLocal(i, j) );
                    test_assert( BT.tileMb(i) == (mb2 == 0 ? Asub_trans.tileMb(i) : mb2) );
                    test_assert( BT.tileNb(j) == (nb2 == 0 ? Asub_trans.tileNb(j) : nb2) );
                    test_assert_throw_std( BT(i, j) );  // tiles don't exist
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// emptyLike with mb, nb overriding size, and op to deep transpose.
void test_Matrix_emptyLikeOp()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, mb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    auto Asub = A.sub( 1, 3, 1, 4 );
    auto Asub_trans = transpose( Asub );
    if (verbose) {
        printf( "A     m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                llong( A.m() ), llong( A.mt() ),
                llong( A.n() ), llong( A.nt() ),
                llong( A.tileMb(0) ), llong( A.tileNb(0) ) );
        printf( "Asub  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                llong( Asub.m() ), llong( Asub.mt() ),
                llong( Asub.n() ), llong( Asub.nt() ),
                llong( Asub.tileMb(0) ), llong( Asub.tileNb(0) ) );
        printf( "AsubT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                llong( Asub_trans.m() ), llong( Asub_trans.mt() ),
                llong( Asub_trans.n() ), llong( Asub_trans.nt() ),
                llong( Asub_trans.tileMb(0) ), llong( Asub_trans.tileNb(0) ) );
    }

    for (int mb2: std::vector<int>({ 0, 7 })) {
        for (int nb2: std::vector<int>({ 0, 5 })) {
            // ----- no trans
            auto B = Asub.emptyLike( mb2, nb2, slate::Op::Trans );

            // just like test_Matrix_emptyLikeMbNb,
            // but swap B's (m, n), (mt, nt), etc.
            if (verbose) {
                printf( "\n" );
                printf( "A  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub.m() ), llong( Asub.mt() ),
                        llong( Asub.n() ), llong( Asub.nt() ),
                        llong( Asub.tileMb(0) ), llong( Asub.tileNb(0) ) );
                printf( "AT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub_trans.m() ), llong( Asub_trans.mt() ),
                        llong( Asub_trans.n() ), llong( Asub_trans.nt() ),
                        llong( Asub_trans.tileMb(0) ), llong( Asub_trans.tileNb(0) ) );
                printf( "B  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld (mb2 %3d, nb2 %3d)\n",
                        llong( B.m() ), llong( B.mt() ),
                        llong( B.n() ), llong( B.nt() ),
                        llong( B.tileMb(0) ), llong( B.tileNb(0) ),
                        mb2, nb2 );
            }
            test_assert(B.n() == (mb2 == 0 ? Asub.m() : Asub.mt() * mb2));
            test_assert(B.m() == (nb2 == 0 ? Asub.n() : Asub.nt() * nb2));
            test_assert(B.nt() == Asub.mt());
            test_assert(B.mt() == Asub.nt());

            for (int j = 0; j < Asub.nt(); ++j) {
                for (int i = 0; i < Asub.mt(); ++i) {
                    test_assert( B.tileIsLocal(j, i) == Asub.tileIsLocal(i, j) );
                    test_assert( B.tileNb(i) == (mb2 == 0 ? Asub.tileMb(i) : mb2) );
                    test_assert( B.tileMb(j) == (nb2 == 0 ? Asub.tileNb(j) : nb2) );
                    test_assert_throw_std( B(j, i) );  // tiles don't exist
                }
            }

            // ----- trans
            auto BT = Asub_trans.emptyLike( mb2, nb2, slate::Op::Trans );

            if (verbose) {
                printf( "\n" );
                printf( "A  m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub.m() ), llong( Asub.mt() ),
                        llong( Asub.n() ), llong( Asub.nt() ),
                        llong( Asub.tileMb(0) ), llong( Asub.tileNb(0) ) );
                printf( "AT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld\n",
                        llong( Asub_trans.m() ), llong( Asub_trans.mt() ),
                        llong( Asub_trans.n() ), llong( Asub_trans.nt() ),
                        llong( Asub_trans.tileMb(0) ), llong( Asub_trans.tileNb(0) ) );
                printf( "BT m %3lld/%3lld, n %3lld/%3lld, mb %3lld, nb %3lld (mb2 %3d, nb2 %3d)\n",
                        llong( BT.m() ), llong( BT.mt() ),
                        llong( BT.n() ), llong( BT.nt() ),
                        llong( BT.tileMb(0) ), llong( BT.tileNb(0) ),
                        mb2, nb2 );
            }
            test_assert(BT.n() == (mb2 == 0 ? Asub_trans.m() : Asub_trans.mt() * mb2));
            test_assert(BT.m() == (nb2 == 0 ? Asub_trans.n() : Asub_trans.nt() * nb2));
            test_assert(BT.nt() == Asub_trans.mt());
            test_assert(BT.mt() == Asub_trans.nt());

            for (int j = 0; j < Asub_trans.nt(); ++j) {
                for (int i = 0; i < Asub_trans.mt(); ++i) {
                    test_assert( BT.tileIsLocal(j, i) == Asub_trans.tileIsLocal(i, j) );
                    test_assert( BT.tileNb(j) == (mb2 == 0 ? Asub_trans.tileMb(i) : mb2) );
                    test_assert( BT.tileMb(i) == (nb2 == 0 ? Asub_trans.tileNb(j) : nb2) );
                    test_assert_throw_std( BT(j, i) );  // tiles don't exist
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test transpose(A).
void test_Matrix_transpose()
{
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    auto AT = transpose( A );

    test_assert(AT.mt() == ceildiv(n, nb));
    test_assert(AT.nt() == ceildiv(m, mb));
    test_assert(AT.op() == slate::Op::Trans);
    test_assert(AT.uplo() == slate::Uplo::General);

    for (int j = 0; j < AT.nt(); ++j) {
        for (int i = 0; i < AT.mt(); ++i) {
            if (AT.tileIsLocal(i, j)) {
                int ib = std::min( nb, n - i*nb );
                int jb = std::min( mb, m - j*mb );
                test_assert(AT(i, j).data() == &Ad[j*mb + i*nb*lda]);
                test_assert(AT(i, j).op() == slate::Op::Trans);
                test_assert(AT(i, j).uplo() == slate::Uplo::General);
                test_assert(AT(i, j).mb() == AT.tileMb(i));
                test_assert(AT(i, j).nb() == AT.tileNb(j));
                test_assert(AT(i, j).mb() == ib);
                test_assert(AT(i, j).nb() == jb);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test conjTranspose(A).
void test_Matrix_conjTranspose()
{
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    auto AT = conjTranspose( A );

    test_assert(AT.mt() == ceildiv(n, nb));
    test_assert(AT.nt() == ceildiv(m, mb));
    test_assert(AT.op() == slate::Op::ConjTrans);
    test_assert(AT.uplo() == slate::Uplo::General);

    for (int j = 0; j < AT.nt(); ++j) {
        for (int i = 0; i < AT.mt(); ++i) {
            if (AT.tileIsLocal(i, j)) {
                int ib = std::min( nb, n - i*nb );
                int jb = std::min( mb, m - j*mb );
                test_assert(AT(i, j).data() == &Ad[j*mb + i*nb*lda]);
                test_assert(AT(i, j).op() == slate::Op::ConjTrans);
                test_assert(AT(i, j).uplo() == slate::Uplo::General);
                test_assert(AT(i, j).mb() == AT.tileMb(i));
                test_assert(AT(i, j).nb() == AT.tileNb(j));
                test_assert(AT(i, j).mb() == ib);
                test_assert(AT(i, j).nb() == jb);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test swap(A, B).
void test_Matrix_swap()
{
    // A has rectangular mb x nb tiles
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    // B has square nb x nb tiles
    int ldb = roundup(n, nb);
    std::vector<double> Bd( ldb*k );
    auto B = slate::Matrix<double>::fromLAPACK(
        n, k, Bd.data(), ldb, nb, nb, p, q, mpi_comm );

    slate::Matrix<double> C = transpose( A );

    test_assert(C.mt() == ceildiv(n, nb));
    test_assert(C.nt() == ceildiv(m, mb));
    test_assert(C.op() == slate::Op::Trans);
    test_assert(C.uplo() == slate::Uplo::General);
    if (C.tileIsLocal(0, 0))
        test_assert(C(0, 0).data() == Ad.data());

    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(k, nb));
    test_assert(B.op() == slate::Op::NoTrans);
    test_assert(B.uplo() == slate::Uplo::General);
    if (B.tileIsLocal(0, 0))
        test_assert(B(0, 0).data() == Bd.data());

    swap(B, C);

    // Compared to above asserts, swap B <=> C in asserts.
    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(m, mb));
    test_assert(B.op() == slate::Op::Trans);
    test_assert(B.uplo() == slate::Uplo::General);
    if (B.tileIsLocal(0, 0))
        test_assert(B(0, 0).data() == Ad.data());

    test_assert(C.mt() == ceildiv(n, nb));
    test_assert(C.nt() == ceildiv(k, nb));
    test_assert(C.op() == slate::Op::NoTrans);
    test_assert(C.uplo() == slate::Uplo::General);
    if (C.tileIsLocal(0, 0))
        test_assert(C(0, 0).data() == Bd.data());
}

//------------------------------------------------------------------------------
/// Test tileInsert( i, j ).
void test_Matrix_tileInsert_new()
{
    auto A = slate::Matrix<double>( m, n, mb, nb, p, q, mpi_comm );

    // Manually insert new tiles, which are allocated by SLATE.
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            int ib = std::min( mb, m - i*mb );
            int jb = std::min( nb, n - j*nb );

            auto T_ptr = A.tileInsert( i, j, HostNum );
            test_assert( T_ptr->mb() == ib );
            test_assert( T_ptr->nb() == jb );
            test_assert( T_ptr->op() == slate::Op::NoTrans );
            test_assert( T_ptr->uplo() == slate::Uplo::General );

            T_ptr->at(0, 0) = i + j / 10000.;
        }
    }

    // Make sure clearing workspace doesn't nuke inserted tiles.
    A.clearWorkspace();

    // Verify tiles.
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            int ib = std::min( mb, m - i*mb );
            int jb = std::min( nb, n - j*nb );

            auto T = A(i, j);
            test_assert( T(0, 0) == i + j / 10000. );
            test_assert( T.mb() == ib );
            test_assert( T.nb() == jb );
            test_assert( T.op() == slate::Op::NoTrans );
            test_assert( T.uplo() == slate::Uplo::General );
        }
    }
}

//------------------------------------------------------------------------------
/// Test tileInsert( i, j, data ).
void test_Matrix_tileInsert_data()
{
    auto A = slate::Matrix<double>( m, n, mb, nb, p, q, mpi_comm );

    // Manually insert tiles from a PLASMA-style tiled matrix.
    // Section A11 has full mb-by-nb tiles.
    // Sections A12, A21, A22 have partial tiles.
    //
    //          n1      n2
    //     +----------+---+
    //     |          |   |    m1 = m - (m % mb)
    //     |          |   |    m2 = m % mb
    // m1  |    A11   |A12|    n1 = n - (n % nb)
    //     |          |   |    n2 = n % nb
    //     |          |   |
    //     +----------+---+
    // m2  |    A21   |A22|
    //     +----------+---+
    int m2 = m % mb;
    int m1 = m - m2;
    int n2 = n % nb;
    int n1 = n - n2;

    std::vector<double> Ad( m*n );
    double* A11 = Ad.data();
    double* A21 = A11 + m1*n1;
    double* A12 = A21 + m2*n1;
    double* A22 = A12 + m1*n2;

    double* Td;
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            int ib = std::min( mb, m - i*mb );
            int jb = std::min( nb, n - j*nb );
            if (i*mb < m1) {
                if (j*nb < n1)
                    Td = A11 + i*mb*nb + j*m1*nb;
                else
                    Td = A12 + i*mb*n2;
            }
            else {
                if (j*nb < n1)
                    Td = A21 + j*m2*nb;
                else
                    Td = A22;
            }
            auto T_ptr = A.tileInsert( i, j, HostNum, Td, ib );
            test_assert( T_ptr->data() == Td );
            test_assert( T_ptr->mb() == ib );
            test_assert( T_ptr->nb() == jb );
            test_assert( T_ptr->op() == slate::Op::NoTrans );
            test_assert( T_ptr->uplo() == slate::Uplo::General );

            T_ptr->at(0, 0) = i + j / 10000.;
        }
    }

    // Make sure clearing workspace doesn't nuke inserted tiles.
    A.clearWorkspace();

    // Verify tiles.
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            int ib = std::min( mb, m - i*mb );
            int jb = std::min( nb, n - j*nb );

            auto T = A(i, j);
            test_assert( T(0, 0) == i + j / 10000. );
            test_assert( T.mb() == ib );
            test_assert( T.nb() == jb );
            test_assert( T.op() == slate::Op::NoTrans );
            test_assert( T.uplo() == slate::Uplo::General );
        }
    }
}

//------------------------------------------------------------------------------
/// Test tileLife.
void test_Matrix_tileLife()
{
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    const int max_life = 4;
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (! A.tileIsLocal(i, j))
                A.tileInsert(i, j);
            A.tileLife(i, j, max_life);
        }
    }

    for (int life = max_life; life > 0; --life) {
        for (int j = 0; j < A.nt(); ++j) {
            for (int i = 0; i < A.mt(); ++i) {
                if (! A.tileIsLocal(i, j)) {
                    // non-local tiles get decremented, and deleted when life reaches 0.
                    test_assert( A.tileLife(i, j) == life );
                    A.tileTick(i, j);
                    if (life - 1 == 0)
                        test_assert_throw_std( A.at(i, j) ); // std::exception (map::at)
                    else
                        test_assert( A.tileLife(i, j) == life - 1 );
                }
                else {
                    // local tiles don't get decremented
                    test_assert( A.tileLife(i, j) == max_life );
                    A.tileTick(i, j);
                    test_assert( A.tileLife(i, j) == max_life );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test tileErase.
void test_Matrix_tileErase()
{
    std::vector<double> Td( mb*nb );

    auto A = slate::Matrix<double>( m, n, mb, nb, p, q, mpi_comm );
    slate::Tile<double> T;

    int i = rand() % A.mt();
    int j = rand() % A.nt();

    A.tileInsert( i, j, HostNum, Td.data(), mb );
    test_assert_no_throw( T = A( i, j ) );
    A.tileErase( i, j, HostNum );
    test_assert_throw_std( T = A( i, j ) );

    // TODO: hard to tell if memory is actually deleted.
    A.tileInsert( i, j, HostNum );
    test_assert_no_throw( T = A( i, j ) );
    A.tileErase( i, j, HostNum );
    test_assert_throw_std( T = A( i, j ) );
}

//------------------------------------------------------------------------------
/// Tests Matrix(), mt, nt, op, insertLocalTiles on host.
void test_Matrix_insertLocalTiles()
{
    slate::Matrix<double> A(m, n, mb, nb, p, q, mpi_comm);

    test_assert(A.mt() == ceildiv(m, mb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    A.insertLocalTiles();
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                auto T = A(i, j);
                test_assert(T.mb() == A.tileMb(i));
                test_assert(T.nb() == A.tileNb(j));
                test_assert(T.stride() == A.tileMb(i));
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Tests Matrix(), mt, nt, op, insertLocalTiles on devices.
void test_Matrix_insertLocalTiles_dev()
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    slate::Matrix<double> A(m, n, mb, nb, p, q, mpi_comm);

    test_assert(A.mt() == ceildiv(m, mb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::General);

    A.insertLocalTiles( slate::Target::Devices );
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                auto T = A(i, j, A.tileDevice(i, j));
                test_assert(T.mb() == A.tileMb(i));
                test_assert(T.nb() == A.tileNb(j));
                test_assert(T.stride() == A.tileMb(i));
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test allocateBatchArrays, clearBatchArrays, batchArraySize.
///
void test_Matrix_allocateBatchArrays()
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, Ad.size(), Ad.data() );

    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // initially, batch arrays are null
    test_assert( A.batchArraySize() == 0 );
    for (int device = 0; device < num_devices; ++device) {
        test_assert( A.array_host(device) == nullptr );
        test_assert( A.array_device(device) == nullptr );
    }

    // allocate size 10
    A.allocateBatchArrays( 10 );
    test_assert( A.batchArraySize() == 10 );
    for (int device = 0; device < num_devices; ++device) {
        test_assert( A.array_host(device) != nullptr );
        test_assert( A.array_device(device) != nullptr );
    }

    // increase to size 20
    A.allocateBatchArrays( 20 );
    test_assert( A.batchArraySize() == 20 );

    // requesting 15 should leave it at 20
    A.allocateBatchArrays( 15 );
    test_assert( A.batchArraySize() == 20 );

    int num = 0;
    for (int device = 0; device < num_devices; ++device) {
        num = blas::max( num, A.getMaxDeviceTiles( device ) );
    }

    // request enough for local tiles
    A.allocateBatchArrays();
    test_assert( A.batchArraySize() == blas::max( num, 20 ) );

    // clear should free arrays
    A.clearBatchArrays();
    test_assert( A.batchArraySize() == 0 );
    for (int device = 0; device < num_devices; ++device) {
        test_assert( A.array_host(device) == nullptr );
        test_assert( A.array_device(device) == nullptr );
    }
}

//==============================================================================
// Sub-matrices

//------------------------------------------------------------------------------
/// Tests A.sub( i1, i2, j1, j2 ).
void test_Matrix_sub()
{
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                A(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    // 1st tile
    auto Asub = A.sub( 0, 0, 0, 0 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::NoTrans );
    test_assert( Asub.uplo() == slate::Uplo::General );
    if (Asub.tileIsLocal(0, 0)) {
        test_assert( Asub(0, 0).at(0, 0) == 0.0 );
        test_assert( Asub(0, 0).op() == slate::Op::NoTrans );
        test_assert( Asub(0, 0).uplo() == slate::Uplo::General);
    }

    // 1st column
    Asub = A.sub( 0, A.mt()-1, 0, 0 );
    test_assert( Asub.mt() == A.mt() );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::NoTrans );
    test_assert( Asub.uplo() == slate::Uplo::General );
    for (int i = 0; i < Asub.mt(); ++i) {
        if (Asub.tileIsLocal(i, 0)) {
            test_assert( Asub(i, 0).at(0, 0) == i );
            test_assert( Asub(i, 0).op() == slate::Op::NoTrans );
            test_assert( Asub(i, 0).uplo() == slate::Uplo::General );
        }
    }

    // 1st row
    Asub = A.sub( 0, 0, 0, A.nt()-1 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == A.nt() );
    test_assert( Asub.op() == slate::Op::NoTrans );
    test_assert( Asub.uplo() == slate::Uplo::General );
    for (int j = 0; j < Asub.nt(); ++j) {
        if (Asub.tileIsLocal(0, j)) {
            test_assert( Asub(0, j).at(0, 0) == j / 10000. );
            test_assert( Asub(0, j).op() == slate::Op::NoTrans );
            test_assert( Asub(0, j).uplo() == slate::Uplo::General );
        }
    }

    // Arbitrary regions. 70% of time, set i1 <= i2, j1 <= j2.
    // i1 > i2 or j1 > j2 are empty matrices.
    for (int cnt = 0; cnt < 10; ++cnt) {
        int i1 = rand() % A.mt();
        int i2 = rand() % A.mt();
        int j1 = rand() % A.nt();
        int j2 = rand() % A.nt();
        if (rand() / double(RAND_MAX) <= 0.7) {
            if (i2 < i1)
                std::swap( i1, i2 );
            if (j2 < j1)
                std::swap( j1, j2 );
        }
        //printf( "sub( %3d, %3d, %3d, %3d )\n", i1, i2, j1, j2 );
        Asub = A.sub( i1, i2, j1, j2 );
        test_assert( Asub.mt() == std::max( i2 - i1 + 1, 0 ) );
        test_assert( Asub.nt() == std::max( j2 - j1 + 1, 0 ) );
        test_assert( Asub.op() == slate::Op::NoTrans );
        test_assert( Asub.uplo() == slate::Uplo::General );
        for (int j = 0; j < Asub.nt(); ++j) {
            for (int i = 0; i < Asub.mt(); ++i) {
                if (Asub.tileIsLocal(i, j)) {
                    test_assert( Asub(i, j).at(0, 0)
                            == (i1 + i) + (j1 + j) / 10000. );
                    test_assert( Asub(i, j).op() == slate::Op::NoTrans );
                    test_assert( Asub(i, j).uplo() == slate::Uplo::General );
                }
            }
        }

        // sub-matrix of Asub
        if (Asub.mt() > 0 && Asub.nt() > 0) {
            int i1_b = rand() % Asub.mt();
            int i2_b = rand() % Asub.mt();
            int j1_b = rand() % Asub.nt();
            int j2_b = rand() % Asub.nt();
            //printf( "   ( %3d, %3d, %3d, %3d )\n", i1_b, i2_b, j1_b, j2_b );
            auto Asub_b = Asub.sub( i1_b, i2_b, j1_b, j2_b );
            test_assert( Asub_b.mt() == std::max( i2_b - i1_b + 1, 0 ) );
            test_assert( Asub_b.nt() == std::max( j2_b - j1_b + 1, 0 ) );
            test_assert( Asub_b.op() == slate::Op::NoTrans );
            test_assert( Asub_b.uplo() == slate::Uplo::General );
            for (int j = 0; j < Asub_b.nt(); ++j) {
                for (int i = 0; i < Asub_b.mt(); ++i) {
                    if (Asub_b.tileIsLocal(i, j)) {
                        test_assert( Asub_b(i, j).at(0, 0)
                                == (i1 + i1_b + i) + (j1 + j1_b + j) / 10000. );
                        test_assert( Asub_b(i, j).op() == slate::Op::NoTrans );
                        test_assert( Asub_b(i, j).uplo() == slate::Uplo::General );
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Tests transpose( A ).sub( i1, i2, j1, j2 ).
void test_Matrix_sub_trans()
{
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                A(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    auto AT = transpose( A );

    auto Asub = AT.sub( 0, 0, 0, 0 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::Trans );
    test_assert( Asub.uplo() == slate::Uplo::General );
    if (Asub.tileIsLocal(0, 0)) {
        test_assert( Asub(0, 0).at(0, 0) == 0.0 );
        test_assert( Asub(0, 0).op() == slate::Op::Trans );
        test_assert( Asub(0, 0).uplo() == slate::Uplo::General );
    }

    // 1st column
    Asub = AT.sub( 0, AT.mt()-1, 0, 0 );
    test_assert( Asub.mt() == AT.mt() );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::Trans );
    test_assert( Asub.uplo() == slate::Uplo::General );
    for (int i = 0; i < Asub.mt(); ++i) {
        if (Asub.tileIsLocal(i, 0)) {
            test_assert( Asub(i, 0).at(0, 0) == i / 10000. );
            test_assert( Asub(i, 0).op() == slate::Op::Trans );
            test_assert( Asub(i, 0).uplo() == slate::Uplo::General );
        }
    }

    // 1st row
    Asub = AT.sub( 0, 0, 0, AT.nt()-1 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == AT.nt() );
    test_assert( Asub.op() == slate::Op::Trans );
    test_assert( Asub.uplo() == slate::Uplo::General );
    for (int j = 0; j < Asub.nt(); ++j) {
        if (Asub.tileIsLocal(0, j)) {
            test_assert( Asub(0, j).at(0, 0) == j );
            test_assert( Asub(0, j).op() == slate::Op::Trans );
            test_assert( Asub(0, j).uplo() == slate::Uplo::General );
        }
    }

    // Arbitrary regions. At least 70% of time, set i1 <= i2, j1 <= j2.
    // i1 > i2 or j1 > j2 are empty matrices.
    for (int cnt = 0; cnt < 10; ++cnt) {
        int i1 = rand() % AT.mt();
        int i2 = rand() % AT.mt();
        int j1 = rand() % AT.nt();
        int j2 = rand() % AT.nt();
        if (rand() / double(RAND_MAX) <= 0.7) {
            if (i2 < i1)
                std::swap( i1, i2 );
            if (j2 < j1)
                std::swap( j1, j2 );
        }
        //printf( "sub( %3d, %3d, %3d, %3d )\n", i1, i2, j1, j2 );
        Asub = AT.sub( i1, i2, j1, j2 );
        test_assert( Asub.mt() == std::max( i2 - i1 + 1, 0 ) );
        test_assert( Asub.nt() == std::max( j2 - j1 + 1, 0 ) );
        test_assert( Asub.op() == slate::Op::Trans );
        test_assert( Asub.uplo() == slate::Uplo::General );
        for (int j = 0; j < Asub.nt(); ++j) {
            for (int i = 0; i < Asub.mt(); ++i) {
                if (Asub.tileIsLocal(i, j)) {
                    test_assert( Asub(i, j).at(0, 0) == (j1 + j) + (i1 + i) / 10000. );
                    test_assert( Asub(i, j).op() == slate::Op::Trans );
                    test_assert( Asub(i, j).uplo() == slate::Uplo::General );
                }
            }
        }

        // sub-matrix of Asub
        auto AsubT = transpose( Asub );
        if (AsubT.mt() > 0 && AsubT.nt() > 0) {
            int i1_b = rand() % AsubT.mt();
            int i2_b = rand() % AsubT.mt();
            int j1_b = rand() % AsubT.nt();
            int j2_b = rand() % AsubT.nt();
            //printf( "   ( %3d, %3d, %3d, %3d )\n", i1_b, i2_b, j1_b, j2_b );
            auto Asub_b = AsubT.sub( i1_b, i2_b, j1_b, j2_b );
            test_assert( Asub_b.mt() == std::max( i2_b - i1_b + 1, 0 ) );
            test_assert( Asub_b.nt() == std::max( j2_b - j1_b + 1, 0 ) );
            test_assert( Asub_b.op() == slate::Op::NoTrans );
            test_assert( Asub_b.uplo() == slate::Uplo::General );
            for (int j = 0; j < Asub_b.nt(); ++j) {
                for (int i = 0; i < Asub_b.mt(); ++i) {
                    if (Asub_b.tileIsLocal(i, j)) {
                        test_assert( Asub_b(i, j).at(0, 0) == (j1 + i1_b + i) + (i1 + j1_b + j) / 10000. );
                        test_assert( Asub_b(i, j).op() == slate::Op::NoTrans );
                        test_assert( Asub_b(i, j).uplo() == slate::Uplo::General );
                    }
                }
            }
        }
    }
}

}  // namespace test

//==============================================================================
// To access BaseMatrix protected members, stick these in the slate::Debug class.
// Admittedly a hack, since this is different than the Debug class in Debug.hh.
namespace slate {
class Debug {
public:

//------------------------------------------------------------------------------
// verify that B == op( A( row1 : row2, col1 : col2 ) )
static void verify_slice(
    slate::Matrix<double>& A,
    int row1, int row2, int col1, int col2,
    int mb, int nb, slate::Op trans )
{
    if (trans == slate::Op::NoTrans) {
        test_assert( A.m() == row2 - row1 + 1 );
        test_assert( A.n() == col2 - col1 + 1 );
        test_assert( A.row0_offset() == row1 % mb );
        test_assert( A.col0_offset() == col1 % nb );
        test_assert( A.mt() == ceildiv( int(A.row0_offset() + row2 - row1 + 1), mb ) );
        test_assert( A.nt() == ceildiv( int(A.col0_offset() + col2 - col1 + 1), nb ) );
        int n_ = col1;  // start of tile j
        for (int j = 0; j < A.nt(); ++j) {
            int m_ = row1;  // start of tile i
            for (int i = 0; i < A.mt(); ++i) {
                if (A.tileIsLocal( i, j )) {
                    auto T = A.at( i, j );
                    test_assert( T.mb() == A.tileMb( i ) );
                    test_assert( T.nb() == A.tileNb( j ) );
                    for (int jj = 0; jj < T.nb(); ++jj) {
                        for (int ii = 0; ii < T.mb(); ++ii) {
                            test_assert( T.at( ii, jj ) == (ii + m_) + (jj + n_)/100. );
                        }
                    }
                }
                m_ += A.tileMb( i );
            }
            n_ += A.tileNb( j );
        }
    }
    else {
        test_assert( A.n() == row2 - row1 + 1 );
        test_assert( A.m() == col2 - col1 + 1 );
        test_assert( A.row0_offset() == row1 % mb );  // todo: nb?
        test_assert( A.col0_offset() == col1 % nb );
        test_assert( A.mt() == ceildiv( int(A.col0_offset() + col2 - col1 + 1), nb ) );
        test_assert( A.nt() == ceildiv( int(A.row0_offset() + row2 - row1 + 1), mb ) );
        int n_ = row1;  // start of tile j
        for (int j = 0; j < A.nt(); ++j) {
            int m_ = col1;  // start of tile i
            for (int i = 0; i < A.mt(); ++i) {
                if (A.tileIsLocal( i, j )) {
                    auto T = A.at( i, j );
                    test_assert( T.mb() == A.tileMb( i ) );
                    test_assert( T.nb() == A.tileNb( j ) );
                    for (int jj = 0; jj < T.nb(); ++jj) {
                        for (int ii = 0; ii < T.mb(); ++ii) {
                            test_assert( T.at( ii, jj ) == (jj + n_) + (ii + m_)/100. );
                        }
                    }
                }
                m_ += A.tileMb( i );
            }
            n_ += A.tileNb( j );
        }
    }
}

//------------------------------------------------------------------------------
/// Tests A.slice( row1, row2, col1, col2 ).
static void test_Matrix_slice()
{
    using namespace test;  // for globals mpi_rank, etc.

    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                auto T = A(i, j);
                for (int jj = 0; jj < T.nb(); ++jj)
                    for (int ii = 0; ii < T.mb(); ++ii)
                        T.at(ii, jj) = i*mb + ii + (j*nb + jj)/100.;
            }
        }
    }

    auto AT = transpose( A );

    for (int cnt = 0; cnt < 100; ++cnt) {
        //printf( "----- cnt %2d\n", cnt );
        int row1, row2, col1, col2;
        row1 = rand() % m;
        row2 = rand() % m;
        col1 = rand() % n;
        col2 = rand() % n;
        if (row1 > row2)
            std::swap( row1, row2 );
        if (col1 > col2)
            std::swap( col1, col2 );
        //printf( "B = A.slice( %2d : %2d, %2d : %2d )\n", row1, row2, col1, col2 );
        auto B = A.slice( row1, row2, col1, col2 );

        int m2 = row2 - row1 + 1;
        int n2 = col2 - col1 + 1;

        test_assert( B.m() == m2 );
        test_assert( B.n() == n2 );
        test_assert( B.row0_offset() == row1 % mb );
        test_assert( B.col0_offset() == col1 % nb );
        test_assert( B.ioffset() == int(row1 / mb) );
        test_assert( B.joffset() == int(col1 / nb) );
        test_assert( B.mt() == ceildiv( int(B.row0_offset() + B.m()), mb ) );
        test_assert( B.nt() == ceildiv( int(B.col0_offset() + B.n()), nb ) );

        verify_slice( B, row1, row2, col1, col2, mb, nb, slate::Op::NoTrans );

        //printf( "BT = AT.slice( ... )\n" );
        auto BT = AT.slice( col1, col2, row1, row2 );

        test_assert( BT.op() == slate::Op::Trans );
        test_assert( BT.uplo() == slate::Uplo::General );
        test_assert( BT.m() == n2 );  // trans
        test_assert( BT.n() == m2 );  // trans
        test_assert( BT.row0_offset() == row1 % mb );
        test_assert( BT.col0_offset() == col1 % nb );
        test_assert( BT.ioffset() == int(row1 / mb) );
        test_assert( BT.joffset() == int(col1 / nb) );
        // trans col, row
        test_assert( BT.mt() == ceildiv( int(BT.col0_offset() + BT.m()), nb ) );
        test_assert( BT.nt() == ceildiv( int(BT.row0_offset() + BT.n()), mb ) );

        verify_slice( BT, row1, row2, col1, col2, mb, nb, slate::Op::Trans );

        //printf( "BT2 = transpose( B )\n" );
        auto BT2 = transpose( B );

        int row3, row4, col3, col4;
        row3 = rand() % m2;
        row4 = rand() % m2;
        col3 = rand() % n2;
        col4 = rand() % n2;
        if (row3 > row4)
            std::swap( row3, row4 );
        if (col3 > col4)
            std::swap( col3, col4 );
        //printf( "C = B.slice( %2d : %2d, %2d : %2d ) => ( %2d : %2d, %2d : %2d )\n",
        //        row3, row4, col3, col4,
        //        row1 + row3, row1 + row4,
        //        col1 + col3, col1 + col4 );
        auto C = B.slice( row3, row4, col3, col4 );

        int m3 = row4 - row3 + 1;
        int n3 = col4 - col3 + 1;

        test_assert( C.op() == slate::Op::NoTrans );
        test_assert( C.uplo() == slate::Uplo::General );
        test_assert( C.m() == m3 );
        test_assert( C.n() == n3 );
        test_assert( C.row0_offset() == (row1 + row3) % mb );
        test_assert( C.col0_offset() == (col1 + col3) % nb );
        test_assert( C.ioffset() == int((row1 + row3) / mb) );
        test_assert( C.joffset() == int((col1 + col3) / nb) );
        test_assert( C.mt() == ceildiv( int(C.row0_offset() + C.m()), mb ) );
        test_assert( C.nt() == ceildiv( int(C.col0_offset() + C.n()), nb ) );

        verify_slice( C, row1 + row3, row1 + row4, col1 + col3, col1 + col4,
                      mb, nb, slate::Op::NoTrans );

        //printf( "CT = BT.slice( ... )\n" );
        auto CT = BT.slice( col3, col4, row3, row4 );

        test_assert( CT.op() == slate::Op::Trans );
        test_assert( CT.uplo() == slate::Uplo::General );
        test_assert( CT.m() == n3 );  // trans
        test_assert( CT.n() == m3 );  // trans
        test_assert( CT.row0_offset() == (row1 + row3) % mb );
        test_assert( CT.col0_offset() == (col1 + col3) % nb );
        test_assert( CT.ioffset() == int((row1 + row3) / mb) );
        test_assert( CT.joffset() == int((col1 + col3) / nb) );
        // col, row trans
        test_assert( CT.mt() == ceildiv( int(CT.col0_offset() + CT.m()), nb ) );
        test_assert( CT.nt() == ceildiv( int(CT.row0_offset() + CT.n()), mb ) );

        verify_slice( CT, row1 + row3, row1 + row4, col1 + col3, col1 + col4,
                      mb, nb, slate::Op::Trans );
    }
}

}; // class Debug
}  // namespace slate

namespace test {

//------------------------------------------------------------------------------
/// Tests Matrix( orig, i1, i2, j1, j2 ).
/// Does the same thing as A.sub( i1, i2, j1, j2 ), just more verbose.
void test_Matrix_sub_Matrix()
{
    int lda = roundup(m, mb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

    int i1 = rand() % A.mt();
    int i2 = rand() % A.mt();
    int j1 = rand() % A.nt();
    int j2 = rand() % A.nt();
    if (i1 > i2)
        std::swap(i1, i2);
    if (j1 > j2)
        std::swap(j1, j2);

    auto B1 = slate::Matrix<double>(A, i1, i2, j1, j2);
    auto B2 = A.sub(i1, i2, j1, j2);

    test_assert(B1.mt() == i2 - i1 + 1);
    test_assert(B1.nt() == j2 - j1 + 1);
    for (int j = 0; j < B1.nt(); ++j) {
        for (int i = 0; i < B1.mt(); ++i) {
            if (B1.tileIsLocal(i, j)) {
                test_assert(B1(i, j).data() == A(i + i1, j + j1).data());
                test_assert(B2(i, j).data() == A(i + i1, j + j1).data());
            }
        }
    }
}

//==============================================================================
// Communication

//------------------------------------------------------------------------------
void test_tileSend_tileRecv()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            int src = A.tileRank(i, j);
            for (int dst = 0; dst < mpi_size; ++dst) {
                if (src != dst) {
                    if (mpi_rank == src) {
                        //printf( "rank %d: send A(%d, %d) from %d to %d\n",
                        //        mpi_rank, i, j, src, dst );

                        // Send tile, then receive updated tile back.
                        auto T = A(i, j);
                        T.at(0, 0) = i + j/1000. + src*1000.;
                        A.tileSend(i, j, dst);
                        A.tileRecv(i, j, dst, A.layout());
                        test_assert( T(0, 0) == i + j/1000. + 1000*dst );
                    }
                    else if (mpi_rank == dst) {
                        //printf( "rank %d: recv A(%d, %d) from %d to %d\n",
                        //        mpi_rank, i, j, src, dst );

                        // Receive tile, update, then send updated tile back.
                        A.tileRecv(i, j, src, A.layout());
                        auto T = A(i, j);
                        test_assert( T(0, 0) == i + j/1000. + 1000*src );
                        T.at(0, 0) = i + j/1000. + 1000*dst;
                        A.tileSend(i, j, src);
                    }
                }
            }
        }
    }
}

//==============================================================================
// tile MOSI & Layout conversion

//------------------------------------------------------------------------------
/// compare tiles data.
void test_Tile_compare_layout(slate::Tile<double> const& Atile,
                              slate::Tile<double> const& Btile,
                              bool same_layout)
{
    test_assert( (Atile.layout() == Btile.layout()) == same_layout );

    const double* Adata = Atile.data();
    const double* Bdata = Btile.data();
    int64_t Astride = Atile.stride();
    int64_t Bstride = Btile.stride();

    for (int jj = 0; jj < Atile.nb(); ++jj) {
        for (int ii = 0; ii < Atile.mb(); ++ii) {
            // Check that actual data is transposed.
            if (same_layout)
                test_assert(Adata[ ii + jj*Astride ] == Bdata[ ii + jj*Bstride ]);
            else
                test_assert(Adata[ jj + ii*Astride ] == Bdata[ ii + jj*Bstride ]);
            // Atile(i, j) takes col/row-major into account.
            test_assert(Atile(ii, jj) == Btile(ii, jj));
        }
    }
}

//------------------------------------------------------------------------------
//
void test_Matrix_MOSI()
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, Ad.size(), Ad.data() );

    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    std::vector<double> Bd = Ad;
    auto B = slate::Matrix<double>::fromLAPACK(
        m, n, Bd.data(), lda, nb, p, q, mpi_comm );

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileState(i, j) == slate::MOSI::Shared);
            }
        }
    }

    A.reserveDeviceWorkspace();

    A.tileGetAllForReadingOnDevices(slate::LayoutConvert::None);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileState(i, j) == slate::MOSI::Shared);
                test_assert(A.tileState(i, j, A.tileDevice(i, j)) == slate::MOSI::Shared);
            }
        }
    }

    A.tileGetAllForWritingOnDevices(slate::LayoutConvert::None);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileState(i, j) == slate::MOSI::Invalid);
                test_assert(A.tileState(i, j, A.tileDevice(i, j)) == slate::MOSI::Modified);
            }
        }
    }

    A.tileUpdateAllOrigin();

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileState(i, j) == slate::MOSI::Shared);
                test_assert(A.tileState(i, j, A.tileDevice(i, j)) == slate::MOSI::Shared);
                // verify data is still correct
                test_Tile_compare_layout(A(i, j), B(i, j), true);
            }
        }
    }

    A.releaseWorkspace();

    A.tileGetAllForReading( HostNum, slate::LayoutConvert::RowMajor );

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileLayout(i, j) == slate::Layout::RowMajor);
                test_assert(A(i, j).extended() != (A.tileMb(i) == A.tileNb(j)));
                test_Tile_compare_layout(A(i, j), B(i, j), false);
            }
        }
    }

    A.tileGetAllForWritingOnDevices(slate::LayoutConvert::None);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileLayout(i, j, A.tileDevice(i, j)) == slate::Layout::RowMajor);
                test_assert(! A(i, j, A.tileDevice(i, j)).extended());
            }
        }
    }

    A.tileGetAllForWriting( HostNum, slate::LayoutConvert::ColMajor );

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileLayout(i, j) == slate::Layout::ColMajor);
                // verify data is still correct
                test_Tile_compare_layout(A(i, j), B(i, j), true);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test tileLayoutConvert.
void test_Matrix_tileLayoutConvert()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, Ad.size(), Ad.data() );

    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    std::vector<double> Bd = Ad;
    auto B = slate::Matrix<double>::fromLAPACK(
        m, n, Bd.data(), lda, nb, p, q, mpi_comm );

    slate::Layout newLayout = A.layout() == slate::Layout::ColMajor ?
                                slate::Layout::RowMajor :
                                slate::Layout::ColMajor;

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileLayout(i, j) == A.layout());
                if (A.tileMb(i) == A.tileNb(j)) {
                    test_assert( A.tileLayoutIsConvertible(i, j) );
                }
                else {
                    test_assert(! A.tileLayoutIsConvertible(i, j) );
                }
                A.tileLayoutConvert(i, j, newLayout);
                test_assert(A.tileLayout(i, j) == newLayout);

                test_Tile_compare_layout(A(i, j), B(i, j), false);
            }
        }
    }

    if (num_devices == 0) {
        test_skip("remainder of test requires num_devices > 0");
    }

    #pragma omp parallel
    #pragma omp master
    {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();

        A.tileGetAllForWritingOnDevices(slate::LayoutConvert::None);

        for (int j = 0; j < A.nt(); ++j) {
            for (int i = 0; i < A.mt(); ++i) {
                if (A.tileIsLocal(i, j)) {
                    test_assert(A.tileLayout(i, j, A.tileDevice(i, j)) == newLayout);
                }
            }
        }

        A.tileLayoutConvertOnDevices(A.layout(), false);

        for (int j = 0; j < A.nt(); ++j) {
            for (int i = 0; i < A.mt(); ++i) {
                if (A.tileIsLocal(i, j)) {
                    test_assert(A.tileLayout(i, j, A.tileDevice(i, j)) == A.layout());
                }
            }
        }

        A.tileGetAllForReading( HostNum, slate::LayoutConvert::None );

        for (int j = 0; j < A.nt(); ++j) {
            for (int i = 0; i < A.mt(); ++i) {
                if (A.tileIsLocal(i, j)) {
                    test_assert(A.tileLayout(i, j) == A.layout());
                    test_Tile_compare_layout(A(i, j), B(i, j), true);
                }
            }
        }

        A.tileLayoutReset();
    }

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                test_assert(A.tileLayout(i, j) == A.layout());
                test_Tile_compare_layout(A(i, j), B(i, j), true);
                test_assert(! A(i, j).extended() );
            }
        }
    }
    A.releaseWorkspace();
}

template <class scalar_t>
void test_BaseMatrix_tileReduceFromSet(
    slate::BaseMatrix<scalar_t>& A, int64_t i, int64_t j,
    std::set<int>& reduce_set)
{
    int tag       = 0;
    int sol_value = 0;
    int root      = A.tileRank(i, j);
    slate::Layout layout = A.layout();

    // Insert the tile, set it locally to the mpi_rank, and compute the solution
    for (auto rank : reduce_set) {
        if (rank == mpi_rank) {
            if (! A.tileIsLocal(i, j)) {
                A.tileInsert(i, j);
            }
            // Set the value
            A.at(i, j).set(mpi_rank);
        }
        sol_value += rank;
    }

    // Routine to test
    A.tileReduceFromSet(i, j, root, reduce_set, 2, tag, layout);

    // Check the result of the reduction
    if (mpi_rank == root) {
        int64_t nrow    = 0;
        int64_t ncol    = 0;
        int64_t Tstride = A.at(i, j).stride();
        scalar_t* Tdata = A.at(i, j).data();

        if (A.at(i, j).op() == slate::Op::NoTrans) {
            nrow = A.at(i, j).mb();
            ncol = A.at(i, j).nb();
        }
        else {
            nrow = A.at(i, j).nb();
            ncol = A.at(i, j).mb();
        }

        for (int ii = 0; ii < nrow; ++ii) {
            for (int jj = 0; jj < ncol; ++jj) {
                test_assert(Tdata[ii + jj * Tstride] == sol_value);
            }
        }
    }

    if (A.tileExists(i, j) && ! A.tileIsLocal(i, j))
        A.tileErase(i, j);
}

template <class scalar_t=double>
void test_Matrix_tileReduceFromSet()
{
    // square tiles
    // TODO rectangular tiles?
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
                 m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto AT = transpose( A );
    auto AH = conjTranspose( A );

    std::set<int> all_reduce_set;

    // Case: all ranks will be part of the reduction
    for (int rank = 0; rank < mpi_size; ++rank) {
        all_reduce_set.insert(rank);
    }

    std::list<slate::Matrix<double>> matrices{ A, AT, AH };
    std::list<std::set<int>> reduce_sets{ all_reduce_set };

    for (auto M : matrices) {
        for (auto reduce_set : reduce_sets) {
            for (int64_t i = 0; i < M.mt(); ++i) {
                for (int64_t j = 0; j < M.nt(); ++j) {
                    int root = M.tileRank(i, j);

                    // Make sure the root is in the reduce_set
                    reduce_set.insert(root);

                    test_BaseMatrix_tileReduceFromSet(M, i, j, reduce_set);
                }
            }
        }
    }
}

//==============================================================================
// todo
// BaseMatrix
//     num_devices
//     tileBcast
//     listBcast
//     tileCopyToDevice
//     tileCopyToHost
//     tileMoveToDevice
//     tileMoveToHost
//     getRanks
//     getLocalDevices
//     numLocalTiles
//     clear
//     clearWorkspace
//     clearBatchArrays
//     [abc]_array_{host, device}
//     compute_stream
//     comm_stream
//
// Matrix
// x   swap
//     getMaxHostTiles
//     getMaxDeviceTiles
//     reserveHostWorkspace
//     reserveDeviceWorkspace
//     gather

//==============================================================================
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_Matrix_default,            "Matrix()",                   mpi_comm);
    run_test(test_Matrix_empty,              "Matrix(m, n, nb, ...)",      mpi_comm);
    run_test(test_Matrix_lambda,             "Matrix(m, n, tileMb, ...)",  mpi_comm);
    run_test(test_Matrix_fromLAPACK,         "Matrix::fromLAPACK",         mpi_comm);
    run_test(test_Matrix_fromLAPACK_rect,    "Matrix::fromLAPACK_rect",    mpi_comm);
    run_test(test_Matrix_fromScaLAPACK,      "Matrix::fromScaLAPACK",      mpi_comm);
    run_test(test_Matrix_fromScaLAPACK_rect, "Matrix::fromScaLAPACK_rect", mpi_comm);
    run_test(test_Matrix_fromDevices,        "Matrix::fromDevices",        mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_Matrix_emptyLike,            "Matrix::emptyLike",                        mpi_comm);
    run_test(test_Matrix_emptyLikeMbNb,        "Matrix::emptyLikeMbNb",                    mpi_comm);
    run_test(test_Matrix_emptyLikeOp,          "Matrix::emptyLikeOp",                      mpi_comm);
    run_test(test_Matrix_transpose,            "transpose",                                mpi_comm);
    run_test(test_Matrix_conjTranspose,        "conjTranspose",                            mpi_comm);
    run_test(test_Matrix_swap,                 "swap",                                     mpi_comm);
    run_test(test_Matrix_tileInsert_new,       "Matrix::tileInsert(i, j, dev) ",           mpi_comm);
    run_test(test_Matrix_tileInsert_data,      "Matrix::tileInsert(i, j, dev, data, lda)", mpi_comm);
    run_test(test_Matrix_tileLife,             "Matrix::tileLife",                         mpi_comm);
    run_test(test_Matrix_tileErase,            "Matrix::tileErase",                        mpi_comm);
    run_test(test_Matrix_tileReduceFromSet,    "Matrix::tileReduceFromSet(i, j, set,...)", mpi_comm);
    run_test(test_Matrix_insertLocalTiles,     "Matrix::insertLocalTiles()",               mpi_comm);
    run_test(test_Matrix_insertLocalTiles_dev, "Matrix::insertLocalTiles(on_devices)",     mpi_comm);
    run_test(test_Matrix_allocateBatchArrays,  "Matrix::allocateBatchArrays",              mpi_comm);
    run_test(test_Matrix_MOSI,                 "Matrix::tileMOSI",                         mpi_comm);
    run_test(test_Matrix_tileLayoutConvert,    "Matrix::tileLayoutConvert",                mpi_comm);

    if (mpi_rank == 0)
        printf("\nSub-matrices and slices\n");
    run_test(test_Matrix_sub,                 "Matrix::sub",      mpi_comm);
    run_test(test_Matrix_sub_trans,           "Matrix::sub(A^T)", mpi_comm);
    run_test(slate::Debug::test_Matrix_slice, "Matrix::slice",    mpi_comm);
    run_test(test_Matrix_sub_Matrix,          "Matrix(orig, i1, i2, j1, j2)", mpi_comm);

    if (mpi_rank == 0)
        printf("\nCommunication\n");
    run_test(test_tileSend_tileRecv, "tileSend, tileRecv", mpi_comm);
}

}  // namespace test

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace test;  // for globals mpi_rank, etc.

    MPI_Init(&argc, &argv);

    mpi_comm = MPI_COMM_WORLD;

    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    num_devices = blas::get_device_count();

    // globals
    m  = 200;
    n  = 100;
    k  = 75;
    mb = 24;
    nb = 16;
    init_process_grid(mpi_size, &p, &q);
    unsigned seed = time( nullptr ) % 10000;  // 4 digit

    // parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i+1 < argc)
            m = atoi( argv[++i] );
        else if (arg == "-n" && i+1 < argc)
            n = atoi( argv[++i] );
        else if (arg == "-k" && i+1 < argc)
            k = atoi( argv[++i] );
        else if (arg == "-mb" && i+1 < argc)
            mb = atoi( argv[++i] );
        else if (arg == "-nb" && i+1 < argc)
            nb = atoi( argv[++i] );
        else if (arg == "-p" && i+1 < argc)
            p = atoi( argv[++i] );
        else if (arg == "-q" && i+1 < argc)
            q = atoi( argv[++i] );
        else if (arg == "-seed" && i+1 < argc)
            seed = atoi( argv[++i] );
        else if (arg == "-v")
            verbose++;
        else {
            printf( "unknown argument: %s\n", argv[i] );
            return 1;
        }
    }
    if (mpi_rank == 0) {
        printf("Usage: %s [-m %d] [-n %d] [-k %d] [-mb %d] [-nb %d] [-p %d] [-q %d] [-seed %d] [-v]\n"
               "num_devices = %d\n",
               argv[0], m, n, k, mb, nb, p, q, seed,
               num_devices);
    }

    MPI_Bcast( &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    srand( seed );

    int err = unit_test_main(mpi_comm);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
