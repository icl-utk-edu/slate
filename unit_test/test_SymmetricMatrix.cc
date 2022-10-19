// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "slate/TrapezoidMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "slate/internal/util.hh"

#include "unit_test.hh"
#include "util_matrix.hh"

using slate::ceildiv;
using slate::roundup;
using slate::GridOrder;

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
/// Tests SymmetricMatrix(), mt, nt, op, uplo.
void test_SymmetricMatrix_default()
{
    slate::SymmetricMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::Lower);

    GridOrder order;
    int myp, myq, myrow, mycol;
    A.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Unknown );
    test_assert( myp == -1 );
    test_assert( myq == -1 );
    test_assert( myrow == -1 );
    test_assert( mycol == -1 );
}

//------------------------------------------------------------------------------
/// n-by-n, no-data constructor
/// Tests SymmetricMatrix( uplo, n, nb, ... ), mt, nt, op, uplo.
void test_SymmetricMatrix_empty()
{
    // ----------
    // lower
    slate::SymmetricMatrix<double> L(blas::Uplo::Lower, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    GridOrder order;
    int myp, myq, myrow, mycol;
    L.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Col );
    test_assert( myp == p );
    test_assert( myq == q );
    test_assert( myrow == mpi_rank % p );
    test_assert( mycol == mpi_rank / p );

    auto tileMb_     = L.tileMbFunc();
    auto tileNb_     = L.tileNbFunc();
    auto tileRank_   = L.tileRankFunc();
    auto tileDevice_ = L.tileDeviceFunc();
    test_assert( tileMb_(0) == nb );  // square
    test_assert( tileNb_(0) == nb );
    test_assert( tileRank_( {0, 0} ) == 0 );
    // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    if (num_devices > 0)
        test_assert( tileDevice_( {0, 0} ) == 0 );

    // ----------
    // upper
    slate::SymmetricMatrix<double> U(blas::Uplo::Upper, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::SymmetricMatrix<double> A(
            blas::Uplo::General, n, nb, p, q, mpi_comm),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// n-by-n, no-data constructor,
/// using lambda functions for tileNb, tileRank, tileDevice.
/// Tests SymmetricMatrix(uplo, n, tileNb, ...), m, n, mt, nt, op.
void test_SymmetricMatrix_lambda()
{
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
    // lower
    slate::SymmetricMatrix<double> L(
        slate::Uplo::Lower, n, tileNb, tileRank, tileDevice, mpi_comm);

    // verify nt, tileNb(i), and sum tileNb(i) == n
    test_assert( L.mt() == L.nt() );
    int nt = L.nt();
    int jj = 0;
    for (int j = 0; j < nt; ++j) {
        test_assert( L.tileNb(j) == blas::min( tileNb(j), n - jj ) );
        test_assert( L.tileNb(j) == L.tileMb(j) );
        jj += L.tileNb(j);
    }
    test_assert( jj == n );

    test_assert(L.m() == n);
    test_assert(L.n() == n);
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == slate::Uplo::Lower);

    auto tileMb_     = L.tileMbFunc();
    auto tileNb_     = L.tileNbFunc();
    auto tileRank_   = L.tileRankFunc();
    auto tileDevice_ = L.tileDeviceFunc();
    test_assert( tileMb_(0) == tileNb(0) );  // square
    test_assert( tileNb_(0) == tileNb(0) );
    test_assert( tileRank_( {0, 0} ) == tileRank( {0, 0} ) );
    // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    if (num_devices > 0)
        test_assert( tileDevice_( {0, 0} ) == tileDevice( {0, 0} ) );

    // ----------
    // upper
    slate::SymmetricMatrix<double> U(
        slate::Uplo::Upper, n, tileNb, tileRank, tileDevice, mpi_comm);

    // verify nt, tileNb(i), and sum tileNb(i) == n
    test_assert( U.mt() == U.nt() );
    nt = U.nt();
    jj = 0;
    for (int j = 0; j < nt; ++j) {
        test_assert( U.tileNb(j) == blas::min( tileNb(j), n - jj ) );
        test_assert( U.tileNb(j) == U.tileMb(j) );
        jj += U.tileNb(j);
    }
    test_assert( jj == n );

    test_assert(U.m() == n);
    test_assert(U.n() == n);
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == slate::Uplo::Upper);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::SymmetricMatrix<double> A(
            blas::Uplo::General, n, tileNb, tileRank, tileDevice, mpi_comm),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromLAPACK
/// Test SymmetricMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromLAPACK.
void test_SymmetricMatrix_fromLAPACK()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    //----------
    // lower
    auto L = slate::SymmetricMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_lapack(L, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // upper
    auto U = slate::SymmetricMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_lapack(U, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::SymmetricMatrix<double>::fromLAPACK(
            blas::Uplo::General, n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromScaLAPACK
/// Test SymmetricMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromScaLAPACK.
void test_SymmetricMatrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb, // square
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    //----------
    // lower
    auto L = slate::SymmetricMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    GridOrder order;
    int myp, myq, myrow, mycol;
    L.gridinfo( &order, &myp, &myq, &myrow, &mycol );
    test_assert( order == GridOrder::Col );
    test_assert( myp == p );
    test_assert( myq == q );
    test_assert( myrow == mpi_rank % p );
    test_assert( mycol == mpi_rank / p );

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // upper
    auto U = slate::SymmetricMatrix<double>::fromScaLAPACK(
        blas::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_scalapack(U, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::SymmetricMatrix<double>::fromScaLAPACK(
            blas::Uplo::General, n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromDevices
/// Test SymmetricMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromDevices.
void test_SymmetricMatrix_fromDevices()
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb, // square
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    // device specific queues
    std::vector< blas::Queue* > dev_queues;
    dev_queues.resize(num_devices);
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

    //----------
    // lower
    auto L = slate::SymmetricMatrix<double>::fromDevices(
        blas::Uplo::Lower, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_device(L, i, j, nb, n, n, Aarray, lda);
        }
    }

    //----------
    // upper
    auto U = slate::SymmetricMatrix<double>::fromDevices(
        blas::Uplo::Upper, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_device(U, i, j, nb, n, n, Aarray, lda);
        }
    }

    for (int dev = 0; dev < num_devices; ++dev) {
        blas::device_free(Aarray[dev], *dev_queues[dev]);
    }
    delete[] Aarray;

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::SymmetricMatrix<double>::fromDevices(
            blas::Uplo::General, n, Aarray, num_devices, lda, nb, p, q, mpi_comm ),
        slate::Exception);

    // free the device specific queues
    for (int dev = 0; dev < num_devices; ++dev)
        delete dev_queues[dev];
}

//==============================================================================
// Methods

//------------------------------------------------------------------------------
/// emptyLike
void test_SymmetricMatrix_emptyLike()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb,  // square
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::SymmetricMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == slate::Uplo::Lower);

    auto B = A.template emptyLike<float>();

    test_assert(B.m() == A.m());
    test_assert(B.n() == A.n());
    test_assert(B.mt() == A.mt());
    test_assert(B.nt() == A.nt());
    test_assert(B.op() == A.op());
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
    auto Asub = A.sub( 1, 3 );
    auto Bsub = Asub.emptyLike();

    test_assert(Bsub.m() == Asub.m());
    test_assert(Bsub.n() == Asub.n());
    test_assert(Bsub.mt() == Asub.mt());
    test_assert(Bsub.nt() == Asub.nt());

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

    test_assert(Atrans.uplo() == slate::Uplo::Upper);
    test_assert(Atrans.op() == blas::Op::Trans);

    test_assert(Btrans.m() == Atrans.m());
    test_assert(Btrans.n() == Atrans.n());
    test_assert(Btrans.mt() == Atrans.mt());
    test_assert(Btrans.nt() == Atrans.nt());

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

    test_assert(Bsub_trans.m() == Asub_trans.m());
    test_assert(Bsub_trans.n() == Asub_trans.n());
    test_assert(Bsub_trans.mt() == Asub_trans.mt());
    test_assert(Bsub_trans.nt() == Asub_trans.nt());

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
void test_SymmetricMatrix_emptyLikeMbNb()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb,  // square
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::SymmetricMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto Asub = A.sub( 1, 3 );
    auto Asub_trans = transpose( Asub );

    for (int nb2: std::vector<int>({ 0, 5 })) {
        // ----- no trans
        auto B = Asub.emptyLike( nb2 );

        test_assert(B.m() == (nb2 == 0 ? Asub.m() : Asub.mt() * nb2));
        test_assert(B.n() == (nb2 == 0 ? Asub.n() : Asub.nt() * nb2));
        test_assert(B.mt() == Asub.mt());
        test_assert(B.nt() == Asub.nt());

        for (int j = 0; j < Asub.nt(); ++j) {
            for (int i = 0; i < Asub.mt(); ++i) {
                test_assert( B.tileIsLocal(i, j) == Asub.tileIsLocal(i, j) );
                test_assert( B.tileMb(i) == (nb2 == 0 ? Asub.tileMb(i) : nb2) );
                test_assert( B.tileNb(j) == (nb2 == 0 ? Asub.tileNb(j) : nb2) );
                test_assert_throw_std( B(i, j) );  // tiles don't exist
            }
        }

        // ----- trans
        auto BT = Asub_trans.emptyLike( nb2 );

        test_assert(BT.m() == (nb2 == 0 ? Asub_trans.m() : Asub_trans.mt() * nb2));
        test_assert(BT.n() == (nb2 == 0 ? Asub_trans.n() : Asub_trans.nt() * nb2));
        test_assert(BT.mt() == Asub_trans.mt());
        test_assert(BT.nt() == Asub_trans.nt());

        for (int j = 0; j < Asub_trans.nt(); ++j) {
            for (int i = 0; i < Asub_trans.mt(); ++i) {
                test_assert( BT.tileIsLocal(i, j) == Asub_trans.tileIsLocal(i, j) );
                test_assert( BT.tileMb(i) == (nb2 == 0 ? Asub_trans.tileMb(i) : nb2) );
                test_assert( BT.tileNb(j) == (nb2 == 0 ? Asub_trans.tileNb(j) : nb2) );
                test_assert_throw_std( BT(i, j) );  // tiles don't exist
            }
        }
    }
}

//------------------------------------------------------------------------------
/// emptyLike with mb, nb overriding size, and op to deep transpose.
void test_SymmetricMatrix_emptyLikeOp()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::SymmetricMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto Asub = A.sub( 1, 3 );
    auto Asub_trans = transpose( Asub );

    for (int nb2: std::vector<int>({ 0, 5 })) {
        // ----- no trans
        auto B = Asub.emptyLike( nb2, slate::Op::Trans );

        // just like test_SymmetricMatrix_emptyLikeMbNb,
        // but swap B's (m, n), (mt, nt), etc.
        test_assert(B.n() == (nb2 == 0 ? Asub.m() : Asub.mt() * nb2));
        test_assert(B.m() == (nb2 == 0 ? Asub.n() : Asub.nt() * nb2));
        test_assert(B.nt() == Asub.mt());
        test_assert(B.mt() == Asub.nt());

        for (int j = 0; j < Asub.nt(); ++j) {
            for (int i = 0; i < Asub.mt(); ++i) {
                test_assert( B.tileIsLocal(j, i) == Asub.tileIsLocal(i, j) );
                test_assert( B.tileNb(i) == (nb2 == 0 ? Asub.tileMb(i) : nb2) );
                test_assert( B.tileMb(j) == (nb2 == 0 ? Asub.tileNb(j) : nb2) );
                test_assert_throw_std( B(j, i) );  // tiles don't exist
            }
        }

        // ----- trans
        auto BT = Asub_trans.emptyLike( nb2, slate::Op::Trans );

        test_assert(BT.n() == (nb2 == 0 ? Asub_trans.m() : Asub_trans.mt() * nb2));
        test_assert(BT.m() == (nb2 == 0 ? Asub_trans.n() : Asub_trans.nt() * nb2));
        test_assert(BT.nt() == Asub_trans.mt());
        test_assert(BT.mt() == Asub_trans.nt());

        for (int j = 0; j < Asub_trans.nt(); ++j) {
            for (int i = 0; i < Asub_trans.mt(); ++i) {
                test_assert( BT.tileIsLocal(j, i) == Asub_trans.tileIsLocal(i, j) );
                test_assert( BT.tileNb(j) == (nb2 == 0 ? Asub_trans.tileMb(i) : nb2) );
                test_assert( BT.tileMb(i) == (nb2 == 0 ? Asub_trans.tileNb(j) : nb2) );
                test_assert_throw_std( BT(j, i) );  // tiles don't exist
            }
        }
    }
}

//==============================================================================
// Sub-matrices

//------------------------------------------------------------------------------
/// Tests A.sub( i1, i2 ).
///
void test_Symmetric_sub()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Mark tiles so they're identifiable.
    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) { // lower
            if (L.tileIsLocal(i, j)) {
                L(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) { // upper
            if (U.tileIsLocal(i, j)) {
                U(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    // Arbitrary regions. 70% of time, set i1 <= i2.
    // i1 > i2 are empty matrices.
    for (int cnt = 0; cnt < 10; ++cnt) {
        int i1 = rand() % L.mt();
        int i2 = rand() % L.mt();
        if (rand() / double(RAND_MAX) <= 0.7) {
            if (i2 < i1)
                std::swap( i1, i2 );
        }

        auto Lsub = L.sub( i1, i2 );
        test_assert( Lsub.mt() == std::max( i2 - i1 + 1, 0 ) );
        test_assert( Lsub.nt() == Lsub.mt() );
        test_assert( Lsub.op() == slate::Op::NoTrans );
        test_assert( Lsub.uplo() == slate::Uplo::Lower );
        for (int j = 0; j < Lsub.nt(); ++j) {
            for (int i = j; i < Lsub.mt(); ++i) { // lower
                if (Lsub.tileIsLocal(i, j)) {
                    test_assert( Lsub(i, j).at(0, 0)
                            == (i1 + i) + (i1 + j) / 10000. );
                    test_assert( Lsub(i, j).op() == slate::Op::NoTrans );
                    if (i == j)
                        test_assert( Lsub(i, j).uplo() == slate::Uplo::Lower );
                    else
                        test_assert( Lsub(i, j).uplo() == slate::Uplo::General );
                }
            }
        }

        auto Usub = U.sub( i1, i2 );
        test_assert( Usub.mt() == std::max( i2 - i1 + 1, 0 ) );
        test_assert( Usub.nt() == Usub.mt() );
        test_assert( Usub.op() == slate::Op::NoTrans );
        test_assert( Usub.uplo() == slate::Uplo::Upper );
        for (int j = 0; j < Usub.nt(); ++j) {
            for (int i = 0; i <= j && i < Usub.mt(); ++i) { // upper
                if (Usub.tileIsLocal(i, j)) {
                    test_assert( Usub(i, j).at(0, 0)
                            == (i1 + i) + (i1 + j) / 10000. );
                    test_assert( Usub(i, j).op() == slate::Op::NoTrans );
                    if (i == j)
                        test_assert( Usub(i, j).uplo() == slate::Uplo::Upper );
                    else
                        test_assert( Usub(i, j).uplo() == slate::Uplo::General );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Tests transpose( A ).sub( i1, i2 ).
///
void test_Symmetric_sub_trans()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Mark tiles so they're identifiable.
    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) { // lower
            if (L.tileIsLocal(i, j)) {
                L(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) { // upper
            if (U.tileIsLocal(i, j)) {
                U(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    auto LT = transpose( L );
    auto UT = transpose( U );

    // Remove 1st block row & col.
    auto L2 = LT.sub( 1, LT.mt()-1 );
    test_assert( L2.mt() == LT.mt()-1 );
    test_assert( L2.nt() == LT.nt()-1 );

    auto U2 = UT.sub( 1, UT.mt()-1 );
    test_assert( U2.mt() == UT.mt()-1 );
    test_assert( U2.nt() == UT.nt()-1 );

    // Arbitrary regions. 70% of time, set i1 <= i2.
    // i1 > i2 are empty matrices.
    for (int cnt = 0; cnt < 10; ++cnt) {
        int i1 = rand() % LT.mt();
        int i2 = rand() % LT.mt();
        if (rand() / double(RAND_MAX) <= 0.7) {
            if (i2 < i1)
                std::swap( i1, i2 );
        }

        auto Lsub = LT.sub( i1, i2 );
        test_assert( Lsub.mt() == std::max( i2 - i1 + 1, 0 ) );
        test_assert( Lsub.nt() == Lsub.mt() );
        test_assert( Lsub.op() == slate::Op::Trans );
        test_assert( Lsub.uplo() == slate::Uplo::Upper );
        for (int j = 0; j < Lsub.nt(); ++j) {
            for (int i = 0; i <= j && i < Lsub.mt(); ++i) { // upper (trans)
                if (Lsub.tileIsLocal(i, j)) {
                    test_assert( Lsub(i, j).at(0, 0)
                            == (i1 + j) + (i1 + i) / 10000. );  // trans
                    test_assert( Lsub(i, j).op() == slate::Op::Trans );
                    if (i == j)
                        test_assert( Lsub(i, j).uplo() == slate::Uplo::Upper );
                    else
                        test_assert( Lsub(i, j).uplo() == slate::Uplo::General );
                }
            }
        }

        auto Usub = UT.sub( i1, i2 );
        test_assert( Usub.mt() == std::max( i2 - i1 + 1, 0 ) );
        test_assert( Usub.nt() == Usub.mt() );
        test_assert( Usub.op() == slate::Op::Trans );
        test_assert( Usub.uplo() == slate::Uplo::Lower );
        for (int j = 0; j < Usub.nt(); ++j) {
            for (int i = j; i < Usub.mt(); ++i) { // lower (trans)
                if (Usub.tileIsLocal(i, j)) {
                    test_assert( Usub(i, j).at(0, 0)
                            == (i1 + j) + (i1 + i) / 10000. );  // trans
                    test_assert( Usub(i, j).op() == slate::Op::Trans );
                    if (i == j)
                        test_assert( Usub(i, j).uplo() == slate::Uplo::Lower );
                    else
                        test_assert( Usub(i, j).uplo() == slate::Uplo::General );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Tests A.slice( i1, i2 ).
///
void test_Symmetric_slice()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Mark entries so they're identifiable.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Ad[ i + j*lda ] = i + j / 10000.;

    // Arbitrary regions.
    // Currently, enforce i2 >= i1.
    // todo: allow i2 < i1 for empty matrix, as in test_Symmetric_sub?
    for (int cnt = 0; cnt < 10; ++cnt) {
        int index1 = rand() % n;
        int index2 = rand() % n;
        if (index2 < index1)
            std::swap( index1, index2 );

        // Get block index for row/col index.
        int i1 = index1 / nb;
        int i2 = index2 / nb;
        //printf( "  index1 %d (%d), index2 %d (%d)\n", index1, i1, index2, i2 );

        auto Lslice = L.slice( index1, index2 );
        test_assert( Lslice.m() == std::max( index2 - index1 + 1, 0 ) );
        test_assert( Lslice.n() == Lslice.m() );
        test_assert( Lslice.mt() == i2 - i1 + 1 );
        test_assert( Lslice.nt() == i2 - i1 + 1 );
        test_assert( Lslice.op() == slate::Op::NoTrans );
        test_assert( Lslice.uplo() == slate::Uplo::Lower );
        for (int j = 0; j < Lslice.nt(); ++j) {
            for (int i = j; i < Lslice.mt(); ++i) { // lower
                if (Lslice.tileIsLocal(i, j)) {
                    // First block row/col starts at index1;
                    // other block row/col start at multiples of nb.
                    int row = (i == 0 ? index1 : (i + i1)*nb);
                    int col = (j == 0 ? index1 : (j + i1)*nb);
                    auto T = Lslice(i, j);
                    test_assert( T.at(0, 0) == row + col / 10000. );
                    test_assert( &T.at(0, 0) == &Ad[ row + col*lda ] );
                    test_assert( T.op() == slate::Op::NoTrans );
                    if (i == j)
                        test_assert( T.uplo() == slate::Uplo::Lower );
                    else
                        test_assert( T.uplo() == slate::Uplo::General );

                    // First and last block rows may be short; row may be both.
                    int mb_ = nb;
                    if (i == Lslice.mt()-1)
                        mb_ = index2 % nb + 1;
                    if (i == 0)
                        mb_ = mb_ - index1 % nb;
                    //printf( "    lower i %d, j %d, mb_ %d, mb %lld\n", i, j, mb_, T.mb() );
                    test_assert( T.mb() == mb_ );

                    // First and last block cols may be short; col may be both.
                    int nb_ = nb;
                    if (j == Lslice.nt()-1)
                        nb_ = index2 % nb + 1;
                    if (j == 0)
                        nb_ = nb_ - index1 % nb;
                    //printf( "    lower i %d, j %d, nb_ %d, nb %lld\n", i, j, nb_, T.nb() );
                    test_assert( T.nb() == nb_ );
                }
            }
        }

        auto Uslice = U.slice( index1, index2 );
        test_assert( Uslice.m() == std::max( index2 - index1 + 1, 0 ) );
        test_assert( Uslice.n() == Uslice.m() );
        test_assert( Uslice.mt() == i2 - i1 + 1 );
        test_assert( Uslice.nt() == i2 - i1 + 1 );
        test_assert( Uslice.op() == slate::Op::NoTrans );
        test_assert( Uslice.uplo() == slate::Uplo::Upper );
        for (int j = 0; j < Uslice.nt(); ++j) {
            for (int i = 0; i <= j && i < Uslice.mt(); ++i) { // upper
                if (Uslice.tileIsLocal(i, j)) {
                    // First block row/col starts at index1;
                    // other block row/col start at multiples of nb.
                    int row = (i == 0 ? index1 : (i + i1)*nb);
                    int col = (j == 0 ? index1 : (j + i1)*nb);
                    auto T = Uslice(i, j);
                    test_assert( T.at(0, 0) == row + col / 10000. );
                    test_assert( &T.at(0, 0) == &Ad[ row + col*lda ] );
                    test_assert( T.op() == slate::Op::NoTrans );
                    if (i == j)
                        test_assert( T.uplo() == slate::Uplo::Upper );
                    else
                        test_assert( T.uplo() == slate::Uplo::General );

                    // First and last block rows may be short; row may be both.
                    int mb_ = nb;
                    if (i == Uslice.mt()-1)
                        mb_ = index2 % nb + 1;
                    if (i == 0)
                        mb_ = mb_ - index1 % nb;
                    //printf( "    lower i %d, j %d, mb_ %d, mb %lld\n", i, j, mb_, T.mb() );
                    test_assert( T.mb() == mb_ );

                    // First and last block cols may be short; col may be both.
                    int nb_ = nb;
                    if (j == Uslice.nt()-1)
                        nb_ = index2 % nb + 1;
                    if (j == 0)
                        nb_ = nb_ - index1 % nb;
                    //printf( "    lower i %d, j %d, nb_ %d, nb %lld\n", i, j, nb_, T.nb() );
                    test_assert( T.nb() == nb_ );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Tests A.slice( i1, i2, j1, j2 ).
///
void test_Symmetric_slice_offdiag()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Mark entries so they're identifiable.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Ad[ i + j*lda ] = i + j / 10000.;

    // Arbitrary regions.
    // For upper: row1 <= row2 <= col1 <= col2.
    // todo: allow row2 > row1, col2 > col1 for empty matrix, as in test_Symmetric_sub?
    for (int cnt = 0; cnt < 10; ++cnt) {
        int idx[4] = {
            rand() % n,
            rand() % n,
            rand() % n,
            rand() % n
        };
        std::sort( idx, idx+4 );

        // Get block index for row/col index.
        int blk[4] = {
            idx[0] / nb,
            idx[1] / nb,
            idx[2] / nb,
            idx[3] / nb
        };

        //printf( "idx [ %d, %d, %d, %d ], blk [ %d, %d, %d, %d ]\n",
        //        idx[0], idx[1], idx[2], idx[3],
        //        blk[0], blk[1], blk[2], blk[3] );

        // For lower: col1 <= col2 <= row1 <= row2.
        auto Lslice = L.slice( idx[2], idx[3], idx[0], idx[1] );
        test_assert( Lslice.m() == std::max( idx[3] - idx[2] + 1, 0 ) );
        test_assert( Lslice.n() == std::max( idx[1] - idx[0] + 1, 0 ) );
        test_assert( Lslice.mt() == blk[3] - blk[2] + 1 );
        test_assert( Lslice.nt() == blk[1] - blk[0] + 1 );
        test_assert( Lslice.op() == slate::Op::NoTrans );
        test_assert( Lslice.uplo() == slate::Uplo::General );
        for (int j = 0; j < Lslice.nt(); ++j) {
            for (int i = j; i < Lslice.mt(); ++i) { // lower
                if (Lslice.tileIsLocal(i, j)) {
                    // First block row/col starts at index1;
                    // other block row/col start at multiples of nb.
                    int row = (i == 0 ? idx[2] : (i + blk[2])*nb);
                    int col = (j == 0 ? idx[0] : (j + blk[0])*nb);
                    auto T = Lslice(i, j);
                    test_assert( T.at(0, 0) == row + col / 10000. );
                    test_assert( &T.at(0, 0) == &Ad[ row + col*lda ] );
                    test_assert( T.op() == slate::Op::NoTrans );
                    test_assert( T.uplo() == slate::Uplo::General );

                    // First and last block rows may be short; row may be both.
                    int mb_ = nb;
                    if (i == Lslice.mt()-1)
                        mb_ = idx[3] % nb + 1;
                    if (i == 0)
                        mb_ = mb_ - idx[2] % nb;
                    //printf( "    lower i %d, j %d, mb_ %d, mb %lld\n", i, j, mb_, T.mb() );
                    test_assert( T.mb() == mb_ );

                    // First and last block cols may be short; col may be both.
                    int nb_ = nb;
                    if (j == Lslice.nt()-1)
                        nb_ = idx[1] % nb + 1;
                    if (j == 0)
                        nb_ = nb_ - idx[0] % nb;
                    //printf( "    lower i %d, j %d, nb_ %d, nb %lld\n", i, j, nb_, T.nb() );
                    test_assert( T.nb() == nb_ );
                }
            }
        }

        auto Uslice = U.slice( idx[0], idx[1], idx[2], idx[3] );
        test_assert( Uslice.m() == std::max( idx[1] - idx[0] + 1, 0 ) );
        test_assert( Uslice.n() == std::max( idx[3] - idx[2] + 1, 0 ) );
        test_assert( Uslice.mt() == blk[1] - blk[0] + 1 );
        test_assert( Uslice.nt() == blk[3] - blk[2] + 1 );
        test_assert( Uslice.op() == slate::Op::NoTrans );
        test_assert( Uslice.uplo() == slate::Uplo::General );
        for (int j = 0; j < Uslice.nt(); ++j) {
            for (int i = 0; i <= j && i < Uslice.mt(); ++i) { // upper
                if (Uslice.tileIsLocal(i, j)) {
                    // First block row/col starts at index1;
                    // other block row/col start at multiples of nb.
                    int row = (i == 0 ? idx[0] : (i + blk[0])*nb);
                    int col = (j == 0 ? idx[2] : (j + blk[2])*nb);
                    auto T = Uslice(i, j);
                    test_assert( T.at(0, 0) == row + col / 10000. );
                    test_assert( &T.at(0, 0) == &Ad[ row + col*lda ] );
                    test_assert( T.op() == slate::Op::NoTrans );
                    test_assert( T.uplo() == slate::Uplo::General );

                    // First and last block rows may be short; row may be both.
                    int mb_ = nb;
                    if (i == Uslice.mt()-1)
                        mb_ = idx[1] % nb + 1;
                    if (i == 0)
                        mb_ = mb_ - idx[0] % nb;
                    //printf( "    lower i %d, j %d, mb_ %d, mb %lld\n", i, j, mb_, T.mb() );
                    test_assert( T.mb() == mb_ );

                    // First and last block cols may be short; col may be both.
                    int nb_ = nb;
                    if (j == Uslice.nt()-1)
                        nb_ = idx[3] % nb + 1;
                    if (j == 0)
                        nb_ = nb_ - idx[2] % nb;
                    //printf( "    lower i %d, j %d, nb_ %d, nb %lld\n", i, j, nb_, T.nb() );
                    test_assert( T.nb() == nb_ );
                }
            }
        }
    }
}

//==============================================================================
// Conversion to Symmetric

//------------------------------------------------------------------------------
/// Tests SymmetricMatrix( uplo, Matrix (BaseMatrix) A ).
///
void test_Symmetric_from_Matrix()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Take sub-matrix, offset by 1 tile.
    A = A.sub( 0, A.mt()-1, 1, A.nt()-1 );

    int64_t min_mt_nt = std::min( A.mt(), A.nt() );
    int64_t min_mn = std::min( A.m(), A.n() );

    // Make square A.
    auto Asquare = A.slice( 0, min_mn-1, 0, min_mn-1 );

    // ----------
    // lower
    auto L = slate::SymmetricMatrix<double>(
        slate::Uplo::Lower, Asquare );
    verify_Symmetric( slate::Uplo::Lower, min_mt_nt, min_mn, L );

    // ----------
    // upper
    auto U = slate::SymmetricMatrix<double>(
        slate::Uplo::Upper, Asquare );
    verify_Symmetric( slate::Uplo::Upper, min_mt_nt, min_mn, U );

    // ----------
    // Rectangular A should fail.
    if (m != n) {
        test_assert_throw(
            auto Lrect = slate::SymmetricMatrix<double>(
                slate::Uplo::Lower, A ),
            slate::Exception);

        test_assert_throw(
            auto Urect = slate::SymmetricMatrix<double>(
                slate::Uplo::Upper, A ),
            slate::Exception);
    }

    // ----------
    // Rectangular tiles (even with square A) should fail.
    if (mb != nb) {
        auto Arect = slate::Matrix<double>::fromLAPACK(
            min_mn, min_mn, Ad.data(), lda, mb, nb, p, q, mpi_comm );

        test_assert_throw(
            auto Lrect = slate::SymmetricMatrix<double>(
                slate::Uplo::Lower, Arect ),
            slate::Exception);

        test_assert_throw(
            auto Urect = slate::SymmetricMatrix<double>(
                slate::Uplo::Upper, Arect ),
            slate::Exception);
    }
}

//------------------------------------------------------------------------------
/// Tests SymmetricMatrix( Trapezoid (BaseTrapezoid) A ).
/// todo: what about Unit diag?
///
void test_Symmetric_from_Trapezoid()
{
    // todo: when Trapezoid has slice, use it as in test_Symmetric_from_Matrix.
    // For now, create as square.
    int64_t min_mn = std::min( m, n );

    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto L0 = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, slate::Diag::NonUnit,
        min_mn, min_mn, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U0 = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, slate::Diag::NonUnit,
        min_mn, min_mn, Ad.data(), lda, nb, p, q, mpi_comm );

    int64_t min_mt_nt = std::min( L0.mt(), L0.nt() );

    // ----------
    // lower
    auto L = slate::SymmetricMatrix<double>( L0 );
    verify_Symmetric( slate::Uplo::Lower, min_mt_nt, min_mn, L );

    // ----------
    // upper
    auto U = slate::SymmetricMatrix<double>( U0 );
    verify_Symmetric( slate::Uplo::Upper, min_mt_nt, min_mn, U );

    // ----------
    // Rectangular A should fail.
    if (m != n) {
        auto L0rect = slate::TrapezoidMatrix<double>::fromLAPACK(
            slate::Uplo::Lower, slate::Diag::NonUnit,
            m, n, Ad.data(), lda, nb, p, q, mpi_comm );

        auto U0rect = slate::TrapezoidMatrix<double>::fromLAPACK(
            slate::Uplo::Upper, slate::Diag::NonUnit,
            m, n, Ad.data(), lda, nb, p, q, mpi_comm );

        test_assert_throw(
            auto Lrect = slate::SymmetricMatrix<double>( L0rect ),
            slate::Exception);

        test_assert_throw(
            auto Urect = slate::SymmetricMatrix<double>( U0rect ),
            slate::Exception);
    }
}

//------------------------------------------------------------------------------
/// Tests SymmetricMatrix( Triangular (BaseTrapezoid) A ).
/// todo: what about Unit diag?
///
void test_Symmetric_from_Triangular()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );
    auto L0 = slate::TriangularMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, slate::Diag::NonUnit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U0 = slate::TriangularMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, slate::Diag::NonUnit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    // ----------
    // lower
    auto L = slate::SymmetricMatrix<double>( L0 );
    verify_Symmetric( slate::Uplo::Lower, L0.mt(), n, L );

    // ----------
    // upper
    auto U = slate::SymmetricMatrix<double>( U0 );
    verify_Symmetric( slate::Uplo::Upper, U0.mt(), n, U );
}

//------------------------------------------------------------------------------
/// Tests SymmetricMatrix( Hermitian (BaseTrapezoid) A ).
///
void test_Symmetric_from_Hermitian()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );
    auto L0 = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Lower,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U0 = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Upper,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    // ----------
    // lower
    auto L = slate::SymmetricMatrix<double>( L0 );
    verify_Symmetric( slate::Uplo::Lower, L0.mt(), n, L );

    // ----------
    // upper
    auto U = slate::SymmetricMatrix<double>( U0 );
    verify_Symmetric( slate::Uplo::Upper, U0.mt(), n, U );
}

//==============================================================================
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_SymmetricMatrix_default,       "SymmetricMatrix()",              mpi_comm);
    run_test(test_SymmetricMatrix_empty,         "SymmetricMatrix(uplo, n, nb, ...)",     mpi_comm);
    run_test(test_SymmetricMatrix_lambda,        "SymmetricMatrix(uplo, n, tileNb, ...)", mpi_comm);
    run_test(test_SymmetricMatrix_fromLAPACK,    "SymmetricMatrix::fromLAPACK",    mpi_comm);
    run_test(test_SymmetricMatrix_fromScaLAPACK, "SymmetricMatrix::fromScaLAPACK", mpi_comm);
    run_test(test_SymmetricMatrix_fromDevices,   "SymmetricMatrix::fromDevices",   mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_SymmetricMatrix_emptyLike,     "SymmetricMatrix::emptyLike()",           mpi_comm);
    run_test(test_SymmetricMatrix_emptyLikeMbNb, "SymmetricMatrix::emptyLike(nb)",         mpi_comm);
    run_test(test_SymmetricMatrix_emptyLikeOp,   "SymmetricMatrix::emptyLike(..., Trans)", mpi_comm);

    if (mpi_rank == 0)
        printf("\nSub-matrices\n");
    run_test(test_Symmetric_sub,           "SymmetricMatrix::sub",                   mpi_comm);
    run_test(test_Symmetric_sub_trans,     "SymmetricMatrix::sub(A^T)",              mpi_comm);
    run_test(test_Symmetric_slice,         "SymmetricMatrix::slice(i1, i2)",         mpi_comm);
    run_test(test_Symmetric_slice_offdiag, "SymmetricMatrix::slice(i1, i2, j1, j2)", mpi_comm);

    if (mpi_rank == 0)
        printf("\nConversion to Symmetric\n");
    run_test(test_Symmetric_from_Matrix,     "SymmetricMatrix( uplo, Matrix )",     mpi_comm);
    run_test(test_Symmetric_from_Hermitian,  "SymmetricMatrix( HermitianMatrix )",  mpi_comm);
    run_test(test_Symmetric_from_Trapezoid,  "SymmetricMatrix( TrapezoidMatrix )",  mpi_comm);
    run_test(test_Symmetric_from_Triangular, "SymmetricMatrix( TriangularMatrix )", mpi_comm);
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
        printf("Usage: %s [-m %d] [-n %d] [-k %d] [-nb %d] [-p %d] [-q %d] [-seed %d] [-v]\n"
               "num_devices = %d\n",
               argv[0], m, n, k, nb, p, q, seed,
               num_devices);
    }

    MPI_Bcast( &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    srand( seed );

    int err = unit_test_main(mpi_comm);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
