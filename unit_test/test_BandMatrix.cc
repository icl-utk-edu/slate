// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/BandMatrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "slate/internal/util.hh"

#include "unit_test.hh"
// #include "auxiliary/Debug.hh"
// #include "../test/print_matrix.hh"

using slate::ceildiv;
using slate::roundup;
using slate::GridOrder;
using slate::HostNum;

namespace test {

//------------------------------------------------------------------------------
// global variables
int m, n, k, kl, ku, nb, p, q;
int mpi_rank;
int mpi_size;
MPI_Comm mpi_comm;
int num_devices = 0;

//------------------------------------------------------------------------------
/// Tests default constructor BandMatrix(), mt, nt, op, lower & upperBandwidth.
void test_BandMatrix()
{
    slate::BandMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.lowerBandwidth() == 0);
    test_assert(A.upperBandwidth() == 0);
}

//------------------------------------------------------------------------------
/// Tests BandMatrix(), mt, nt, op, lower & upperBandwidth.
void test_BandMatrix_empty()
{
    auto A = slate::BandMatrix<double>(m, n, kl, ku, nb, p, q, mpi_comm);

    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.lowerBandwidth() == kl);
    test_assert(A.upperBandwidth() == ku);

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
    test_assert( tileRank_( {0, 0} ) == 0 );
    // todo: What is reasonable if num_devices == 0? Currently divides by zero.
    if (num_devices > 0)
        test_assert( tileDevice_( {0, 0} ) == 0 );
}

//==============================================================================
// Methods

//------------------------------------------------------------------------------
void test_BandMatrix_transpose()
{
    auto A = slate::BandMatrix<double>(m, n, kl, ku, nb, p, q, mpi_comm);

    auto AT = transpose( A );

    test_assert(AT.mt() == ceildiv(n, nb));
    test_assert(AT.nt() == ceildiv(m, nb));
    test_assert(AT.op() == slate::Op::Trans);
    test_assert(AT.lowerBandwidth() == ku);
    test_assert(AT.upperBandwidth() == kl);
    test_assert(AT.uplo() == slate::Uplo::General );
}

//------------------------------------------------------------------------------
void test_BandMatrix_conj_transpose()
{
    auto A = slate::BandMatrix<double>(m, n, kl, ku, nb, p, q, mpi_comm);

    auto AT = conj_transpose( A );

    test_assert(AT.mt() == ceildiv(n, nb));
    test_assert(AT.nt() == ceildiv(m, nb));
    test_assert(AT.op() == slate::Op::ConjTrans);
    test_assert(AT.lowerBandwidth() == ku);
    test_assert(AT.upperBandwidth() == kl);
    test_assert(AT.uplo() == slate::Uplo::General );
}

//------------------------------------------------------------------------------
void test_BandMatrix_swap()
{
    auto A = slate::BandMatrix<double>(m, n, kl, ku, nb, p, q, mpi_comm);

    int kl2 = kl + 1;
    int ku2 = ku + 1;
    auto B = slate::BandMatrix<double>(n, k, kl2, ku2, nb, p, q, mpi_comm);

    auto C = transpose( A );

    test_assert(C.mt() == ceildiv(n, nb));
    test_assert(C.nt() == ceildiv(m, nb));
    test_assert(C.op() == slate::Op::Trans);
    //test_assert(C(0,0).data() == Ad.data());
    test_assert(C.lowerBandwidth() == ku);
    test_assert(C.upperBandwidth() == kl);

    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(k, nb));
    test_assert(B.op() == slate::Op::NoTrans);
    //test_assert(B(0,0).data() == Bd.data());
    test_assert(B.lowerBandwidth() == kl2);
    test_assert(B.upperBandwidth() == ku2);

    swap(B, C);

    // swap(B, C) in asserts
    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(m, nb));
    test_assert(B.op() == slate::Op::Trans);
    //test_assert(B(0,0).data() == Ad.data());
    test_assert(B.lowerBandwidth() == ku);
    test_assert(B.upperBandwidth() == kl);

    test_assert(C.mt() == ceildiv(n, nb));
    test_assert(C.nt() == ceildiv(k, nb));
    test_assert(C.op() == slate::Op::NoTrans);
    //test_assert(C(0,0).data() == Bd.data());
    test_assert(C.lowerBandwidth() == kl2);
    test_assert(C.upperBandwidth() == ku2);
}

//------------------------------------------------------------------------------
void test_BandMatrix_tileInsert_new()
{
    auto A = slate::BandMatrix<double>(m, n, kl, ku, nb, p, q, mpi_comm);

    // Manually insert new tiles, which are allocated by SLATE.
    int jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if ((jj >= ii && jj - ii <= A.upperBandwidth()) ||
                (jj <= ii && ii - jj <= A.lowerBandwidth()))
            {
                int ib = std::min( nb, m - ii );
                int jb = std::min( nb, n - jj );

                auto T_ptr = A.tileInsert( i, j, HostNum );
                test_assert( T_ptr->mb() == ib );
                test_assert( T_ptr->nb() == jb );
                test_assert( T_ptr->op() == slate::Op::NoTrans );
                test_assert( T_ptr->uplo() == slate::Uplo::General );

                T_ptr->at(0, 0) = i + j / 10000.;
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    // Verify tiles.
    jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if ((jj >= ii && jj - ii <= A.upperBandwidth()) ||
                (jj <= ii && ii - jj <= A.lowerBandwidth()))
            {
                int ib = std::min( nb, m - ii );
                int jb = std::min( nb, n - jj );

                auto T = A(i, j);
                test_assert( T(0, 0) == i + j / 10000. );
                test_assert( T.mb() == ib );
                test_assert( T.nb() == jb );
                test_assert( T.op() == slate::Op::NoTrans );
                test_assert( T.uplo() == slate::Uplo::General );
            }
            else {
                // outside band, tiles don't exist
                test_assert_throw_std( A(i, j) );
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }
}

//------------------------------------------------------------------------------
void test_BandMatrix_tileInsert_data()
{
    auto A = slate::BandMatrix<double>( m, n, kl, ku, nb, p, q, mpi_comm );

    // Manually insert tiles, allocated in column-wise block packed fashion:
    // For kl_tiles = 1 and ku_tiles = 2,
    //     A = [ A00  A01  A02   -    -   ]
    //         [ A10  A11  A12  A13   -   ]
    //         [  -   A21  A22  A23  A24  ]
    //         [  -    -   A32  A33  A34  ]
    // is stored as
    //     [ A00  A10 | A01 A11 A21 | A02 A12 A22 A32 | A13 A23 A33 | A24 A34 ]
    int kl_tiles = slate::ceildiv( kl, nb );
    int ku_tiles = slate::ceildiv( ku, nb );
    // over-estimate of # tiles
    int ntiles = (kl_tiles + ku_tiles + 1) * std::min( A.mt(), A.nt() );

    std::vector<double> Ad( ntiles * nb * nb );

    // Manually insert new tiles inside band.
    int index = 0; // index in Ad storage
    int jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if ((ii == jj) ||
                (ii < jj && jj - (ii + A.tileMb(i) - 1) <= A.upperBandwidth()) ||
                (ii > jj && ii - (jj + A.tileNb(j) - 1) <= A.lowerBandwidth()))
            {
                int ib = std::min( nb, m - ii );
                int jb = std::min( nb, n - jj );

                auto T_ptr = A.tileInsert( i, j, &Ad[ index * nb * nb ], nb );
                index += 1;

                test_assert( T_ptr->mb() == ib );
                test_assert( T_ptr->nb() == jb );
                test_assert( T_ptr->stride() == nb );
                test_assert( T_ptr->op() == slate::Op::NoTrans );
                test_assert( T_ptr->uplo() == slate::Uplo::General );

                T_ptr->at(0, 0) = i + j / 10000.;
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    // Verify tiles.
    jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if ((ii == jj) ||
                (ii < jj && jj - (ii + A.tileMb(i) - 1) <= A.upperBandwidth()) ||
                (ii > jj && ii - (jj + A.tileNb(j) - 1) <= A.lowerBandwidth()))
            {
                // inside band
                int ib = std::min( nb, m - ii );
                int jb = std::min( nb, n - jj );

                auto T = A(i, j);
                test_assert( T(0, 0) == i + j / 10000. );
                test_assert( T.mb() == ib );
                test_assert( T.nb() == jb );
                test_assert( T.stride() == nb );
                test_assert( T.op() == slate::Op::NoTrans );
                test_assert( T.uplo() == slate::Uplo::General );
            }
            else {
                // outside band, tiles don't exist
                test_assert_throw_std( A(i, j) );
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }
}

//------------------------------------------------------------------------------
void test_BandMatrix_sub()
{
    auto A = slate::BandMatrix<double>( m, n, kl, ku, nb, p, q, mpi_comm );

    // Currently, A.sub returns a general Matrix, not a BandMatrix.
    // If sub is entirely within the band, then a general Matrix is best;
    // if sub is along diagonal (i1 == j1), then it could be a BandMatrix;
    // if sub is off-diagonal, it would be something weird.
    int i1 = rand() % A.mt();
    int i2 = rand() % A.mt();
    int j1 = rand() % A.nt();
    int j2 = rand() % A.nt();
    slate::Matrix<double> B = A.sub( i1, i2, j1, j2 );

    // todo: verify properties of B
}

//------------------------------------------------------------------------------
void test_BandMatrix_sub_trans()
{
    auto A = slate::BandMatrix<double>( m, n, kl, ku, nb, p, q, mpi_comm );

    auto AT = transpose( A );

    // Currently, A.sub returns a general Matrix, not a BandMatrix.
    // If sub is entirely within the band, then a general Matrix is best;
    // if sub is along diagonal (i1 == j1), then it could be a BandMatrix;
    // if sub is off-diagonal, it would be something weird.
    int i1 = rand() % AT.mt();
    int i2 = rand() % AT.mt();
    int j1 = rand() % AT.nt();
    int j2 = rand() % AT.nt();
    slate::Matrix<double> B = AT.sub( i1, i2, j1, j2 );

    // todo: verify properties of B
}

//------------------------------------------------------------------------------
void test_TriangularBandMatrix_gatherAll(slate::Uplo uplo)
{
    auto upper = uplo == slate::Uplo::Upper;
    auto kd = uplo == slate::Uplo::Upper ? ku : kl;

    auto A = slate::TriangularBandMatrix<double>(
                        uplo, slate::Diag::NonUnit,
                        m, kd, nb, p, q, mpi_comm);

    test_assert(A.mt() == A.nt());
    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.bandwidth() == kd);

    // Manually insert tiles, allocated in column-wise block packed fashion:
    // For kd_tiles = 2,
    //     A = [ A00  A01  A02   -   ]
    //         [  -   A11  A12  A13  ]
    //         [  -    -   A22  A23  ]
    //         [  -    -    -   A33  ]
    // is stored as
    //     [ A00 | A01 A11 | A02 A12 A22 | A13 A23 A33 ]
    int kd_tiles = slate::ceildiv( kd, nb );
    // over-estimate of # tiles
    int ntiles = (kd_tiles + 1) * A.mt();

    std::vector<double> Ad( ntiles * nb * nb );

    // Manually insert new tiles inside triangular band.
    int index = 0; // index in Ad storage
    int jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j) &&
                ((ii == jj) ||
                 ( upper && ii < jj && (jj - (ii + A.tileMb(i) - 1)) <= A.bandwidth()) ||
                 (!upper && ii > jj && (ii - (jj + A.tileNb(j) - 1)) <= A.bandwidth()) ) )
            {
                int ib = std::min( nb, m - ii );
                int jb = std::min( nb, m - jj );

                auto T_ptr = A.tileInsert( i, j, &Ad[ index * nb * nb ], nb );
                if (i == j)
                    T_ptr->uplo(uplo);
                index += 1;

                test_assert( T_ptr->mb() == ib );
                test_assert( T_ptr->nb() == jb );
                test_assert( T_ptr->stride() == nb );
                test_assert( T_ptr->op() == slate::Op::NoTrans );
                if (i == j)
                    test_assert( T_ptr->uplo() == uplo );
                else
                    test_assert( T_ptr->uplo() == slate::Uplo::General );

                T_ptr->set(i + j / 10.);
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }
    // if (mpi_rank == 0) {
    //     slate::Debug::printTiles( A, Field_MOSI );
    // }

    // Find the set of participating ranks.
    std::set<int> rank_set;
    A.getRanks(&rank_set);

    // gather A on each rank
    A.gatherAll(rank_set);
    // if (mpi_rank == 0) {
    //     slate::Debug::printTiles( A, Field_MOSI );
    // }

    // print_matrix( "A", A, 4, 1);

    jj = 0; // col index
    for (int j = 0; j < A.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < A.mt(); ++i) {
            if (((ii == jj) ||
                 ( upper && ii < jj && (jj - (ii + A.tileMb(i) - 1)) <= A.bandwidth()) ||
                 (!upper && ii > jj && (ii - (jj + A.tileNb(j) - 1)) <= A.bandwidth()) ) )
            {
                test_assert( A.tileExists(i, j) );
                if (i == j)
                    test_assert( A(i, j).uplo() == uplo );
                else
                    test_assert( A(i, j).uplo() == slate::Uplo::General );
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }
}

//------------------------------------------------------------------------------
void test_TriangularBandMatrix_gatherAll()
{
    auto uplo = slate::Uplo::Upper;
    test_TriangularBandMatrix_gatherAll(uplo);

    uplo = slate::Uplo::Lower;
    test_TriangularBandMatrix_gatherAll(uplo);
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_BandMatrix,            "BandMatrix()",           mpi_comm);

    if (mpi_rank == 0)
        printf("\nm-by-n, no data constructors\n");
    run_test(test_BandMatrix_empty,           "BandMatrix(m, n, ...)",                mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_BandMatrix_transpose,       "transpose",      mpi_comm);
    run_test(test_BandMatrix_conj_transpose,  "conj_transpose", mpi_comm);
    run_test(test_BandMatrix_swap,            "swap",           mpi_comm);
    run_test(test_BandMatrix_tileInsert_new,  "BandMatrix::tileInsert(i, j, dev) ", mpi_comm);
    run_test(test_BandMatrix_tileInsert_data, "BandMatrix::tileInsert(i, j, dev, data, lda)",  mpi_comm);
    run_test(test_BandMatrix_sub,             "BandMatrix::sub",       mpi_comm);
    run_test(test_BandMatrix_sub_trans,       "BandMatrix::sub(A^T)",  mpi_comm);
    run_test(test_TriangularBandMatrix_gatherAll, "TriangularBandMatrix::gatherAll()",  mpi_comm);
}

}  // namespace test

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace test;  // for globals mpi_rank, m, n, k, etc.

    MPI_Init(&argc, &argv);

    mpi_comm = MPI_COMM_WORLD;

    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    num_devices = blas::get_device_count();

    // globals
    m  = 200;
    n  = 100;
    k  = 75;
    kl = 10;
    ku = 20;
    nb = 16;
    p  = std::min(2, mpi_size);
    q  = mpi_size / p;
    unsigned seed = time( nullptr ) % 10000;  // 4 digit
    if (argc > 1) { m  = atoi(argv[1]); }
    if (argc > 2) { n  = atoi(argv[2]); }
    if (argc > 3) { k  = atoi(argv[3]); }
    if (argc > 4) { kl = atoi(argv[4]); }
    if (argc > 5) { ku = atoi(argv[5]); }
    if (argc > 6) { nb = atoi(argv[6]); }
    if (argc > 7) { p  = atoi(argv[7]); }
    if (argc > 8) { q  = atoi(argv[8]); }
    if (argc > 9) { seed = atoi(argv[9]); }
    if (mpi_rank == 0) {
        printf("Usage: %s %4s %4s %4s %4s %4s %4s %4s %4s %4s\n"
               "       %s %4d %4d %4d %4d %4d %4d %4d %4d %4u\n"
               "num_devices = %d\n",
               argv[0], "m", "n", "k", "kl", "ku", "nb", "p", "q", "seed",
               argv[0], m, n, k, kl, ku, nb, p, q, seed,
               num_devices);
    }

    MPI_Bcast( &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    srand( seed );

    int err = unit_test_main(mpi_comm);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
