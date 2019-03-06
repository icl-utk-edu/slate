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
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUT E GOODS OR SERVICES;
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

#include "slate_Matrix.hh"
#include "slate_HermitianMatrix.hh"
#include "slate_SymmetricMatrix.hh"
#include "slate_TrapezoidMatrix.hh"
#include "slate_TriangularMatrix.hh"
#include "slate/slate_util.hh"

#include "unit_test.hh"
#include "util_matrix.hh"

using slate::ceildiv;
using slate::roundup;

//------------------------------------------------------------------------------
// global variables
int m, n, k, nb, p, q;
int mpi_rank;
int mpi_size;
MPI_Comm mpi_comm;
int host_num, num_devices;
int verbose = 0;

//==============================================================================
// Constructors

//------------------------------------------------------------------------------
/// default constructor
/// Tests Matrix(), mt, nt, op.
void test_Matrix()
{
    slate::Matrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor
/// Tests Matrix(), mt, nt, op.
void test_Matrix_empty()
{
    slate::Matrix<double> A(m, n, nb, p, q, mpi_comm);

    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
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

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_lapack(A, i, j, nb, m, n, Ad.data(), lda);
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
        m, n, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_scalapack(A, i, j, nb, m, n, Ad.data(), lda);
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
        m, n, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    double** Aarray = new double*[ num_devices ];
    for (int dev = 0; dev < num_devices; ++dev) {
        int ntiles_local2, ntiles_dev, n_dev;
        get_cyclic_dimensions(num_devices, dev, n_local, nb,
                               ntiles_local2, ntiles_dev, n_dev);
        assert(ntiles_local == ntiles_local2);

        // cudaMalloc returns null if len = 0, so make it at least 1.
        size_t len = std::max(sizeof(double) * lda * n_dev, size_t(1));
        cudaMalloc((void**)&Aarray[dev], len);
        assert(Aarray[dev] != nullptr);
    }

    auto A = slate::Matrix<double>::fromDevices(
        m, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_device(A, i, j, nb, m, n, Aarray, lda);
        }
    }

    for (int dev = 0; dev < num_devices; ++dev) {
        cudaFree(Aarray[dev]);
    }
    delete[] Aarray;
}

//------------------------------------------------------------------------------
/// emptyLike
void test_Matrix_emptyLike()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);

    auto B = A.emptyLike();

    test_assert(B.m() == A.m());
    test_assert(B.n() == A.n());
    test_assert(B.mt() == A.mt());
    test_assert(B.nt() == A.nt());

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

//==============================================================================
// Methods

//------------------------------------------------------------------------------
/// Test transpose(A).
void test_Matrix_transpose()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto AT = transpose( A );

    test_assert(AT.mt() == ceildiv(n, nb));
    test_assert(AT.nt() == ceildiv(m, nb));
    test_assert(AT.op() == slate::Op::Trans);

    for (int j = 0; j < AT.nt(); ++j) {
        for (int i = 0; i < AT.mt(); ++i) {
            if (AT.tileIsLocal(i, j)) {
                int ib = std::min( nb, n - i*nb );
                int jb = std::min( nb, m - j*nb );
                test_assert(AT(i, j).data() == &Ad[j*nb + i*nb*lda]);
                test_assert(AT(i, j).op() == slate::Op::Trans);
                test_assert(AT(i, j).mb() == AT.tileMb(i));
                test_assert(AT(i, j).nb() == AT.tileNb(j));
                test_assert(AT(i, j).mb() == ib);
                test_assert(AT(i, j).nb() == jb);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test conj_transpose(A).
void test_Matrix_conj_transpose()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto AT = conj_transpose( A );

    test_assert(AT.mt() == ceildiv(n, nb));
    test_assert(AT.nt() == ceildiv(m, nb));
    test_assert(AT.op() == slate::Op::ConjTrans);

    for (int j = 0; j < AT.nt(); ++j) {
        for (int i = 0; i < AT.mt(); ++i) {
            if (AT.tileIsLocal(i, j)) {
                int ib = std::min( nb, n - i*nb );
                int jb = std::min( nb, m - j*nb );
                test_assert(AT(i, j).data() == &Ad[j*nb + i*nb*lda]);
                test_assert(AT(i, j).op() == slate::Op::ConjTrans);
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
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    int ldb = roundup(n, nb);
    std::vector<double> Bd( ldb*k );
    auto B = slate::Matrix<double>::fromLAPACK(
        n, k, Bd.data(), ldb, nb, p, q, mpi_comm );

    slate::Matrix<double> C = transpose( A );

    test_assert(C.mt() == ceildiv(n, nb));
    test_assert(C.nt() == ceildiv(m, nb));
    test_assert(C.op() == slate::Op::Trans);
    if (C.tileIsLocal(0, 0))
        test_assert(C(0, 0).data() == Ad.data());

    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(k, nb));
    test_assert(B.op() == slate::Op::NoTrans);
    if (C.tileIsLocal(0, 0))
        test_assert(B(0, 0).data() == Bd.data());

    swap(B, C);

    // swap(B, C) in asserts
    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(m, nb));
    test_assert(B.op() == slate::Op::Trans);
    if (C.tileIsLocal(0, 0))
        test_assert(B(0, 0).data() == Ad.data());

    test_assert(C.mt() == ceildiv(n, nb));
    test_assert(C.nt() == ceildiv(k, nb));
    test_assert(C.op() == slate::Op::NoTrans);
    if (C.tileIsLocal(0, 0))
        test_assert(C(0, 0).data() == Bd.data());
}

//------------------------------------------------------------------------------
/// Test tileInsert( i, j ).
void test_Matrix_tileInsert_new()
{
    auto A = slate::Matrix<double>( m, n, nb, p, q, mpi_comm );

    // Manually insert new tiles, which are allocated by SLATE.
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            int ib = std::min( nb, m - i*nb );
            int jb = std::min( nb, n - j*nb );

            auto T_ptr = A.tileInsert( i, j );  //, A.hostNum() );
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
            int ib = std::min( nb, m - i*nb );
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
    auto A = slate::Matrix<double>( m, n, nb, p, q, mpi_comm );

    // Manually insert tiles from a PLASMA-style tiled matrix.
    // Section A11 has full nb-by-nb tiles.
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
    int m2 = m % nb;
    int m1 = m - m2;
    int n2 = n % nb;
    int n1 = n - n2;

    std::vector<double> Ad( m*n );
    double *A11 = Ad.data();
    double *A21 = A11 + m1*n1;
    double *A12 = A21 + m2*n1;
    double *A22 = A12 + m1*n2;

    double *Td;
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            int ib = std::min( nb, m - i*nb );
            int jb = std::min( nb, n - j*nb );
            if (i*nb < m1) {
                if (j*nb < n1)
                    Td = A11 + i*nb*nb + j*m1*nb;
                else
                    Td = A12 + i*nb*n2;
            }
            else {
                if (j*nb < n1)
                    Td = A21 + j*m2*nb;
                else
                    Td = A22;
            }
            //auto T_ptr = A.tileInsert( i, j, A.hostNum(), Td, ib );
            auto T_ptr = A.tileInsert( i, j, Td, ib );
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
            int ib = std::min( nb, m - i*nb );
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
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    const int max_life = 4;
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            A.tileLife(i, j, max_life);
        }
    }

    for (int life = max_life; life > 0; --life) {
        for (int j = 0; j < A.nt(); ++j) {
            for (int i = 0; i < A.mt(); ++i) {
                if (! A.tileIsLocal(i, j)) {
                    // non-local tiles get decremented
                    test_assert( A.tileLife(i, j) == life );
                    A.tileTick(i, j);
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
    std::vector<double> Td( nb*nb );

    auto A = slate::Matrix<double>( m, n, nb, p, q, mpi_comm );
    slate::Tile<double> T;

    int i = rand() % A.mt();
    int j = rand() % A.nt();

    A.tileInsert( i, j, Td.data(), nb );  //A.hostNum()
    test_assert_no_throw( T = A( i, j ) );
    A.tileErase( i, j, A.hostNum() );
    test_assert_throw_std( T = A( i, j ) );

    // TODO: hard to tell if memory is actually deleted.
    A.tileInsert( i, j ); //A.hostNum()
    test_assert_no_throw( T = A( i, j ) );
    A.tileErase( i, j, A.hostNum() );
    test_assert_throw_std( T = A( i, j ) );
}

//------------------------------------------------------------------------------
/// Tests Matrix(), mt, nt, op, insertLocalTiles on host.
void test_Matrix_insertLocalTiles()
{
    slate::Matrix<double> A(m, n, nb, p, q, mpi_comm);

    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);

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
    slate::Matrix<double> A(m, n, nb, p, q, mpi_comm);

    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);

    A.insertLocalTiles( true );
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

//==============================================================================
// Sub-matrices and conversions

//------------------------------------------------------------------------------
/// Tests A.sub( i1, i2, j1, j2 ).
void test_Matrix_sub()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                A(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    auto Asub = A.sub( 0, 0, 0, 0 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::NoTrans );
    if (Asub.tileIsLocal(0, 0)) {
        test_assert( Asub(0, 0).op() == slate::Op::NoTrans );
        test_assert( Asub(0, 0).at(0, 0) == 0.0 );
    }

    // 1st column
    Asub = A.sub( 0, A.mt()-1, 0, 0 );
    test_assert( Asub.mt() == A.mt() );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::NoTrans );
    for (int i = 0; i < Asub.mt(); ++i) {
        if (Asub.tileIsLocal(i, 0)) {
            test_assert( Asub(i, 0).at(0, 0) == i );
            test_assert( Asub(i, 0).op() == slate::Op::NoTrans );
        }
    }

    // 1st row
    Asub = A.sub( 0, 0, 0, A.nt()-1 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == A.nt() );
    test_assert( Asub.op() == slate::Op::NoTrans );
    for (int j = 0; j < Asub.nt(); ++j) {
        if (Asub.tileIsLocal(0, j)) {
            test_assert( Asub(0, j).at(0, 0) == j / 10000. );
            test_assert( Asub(0, j).op() == slate::Op::NoTrans );
        }
    }

    // Arbitrary regions. At least 70% of time, set i1 <= i2, j1 <= j2.
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
        for (int j = 0; j < Asub.nt(); ++j) {
            for (int i = 0; i < Asub.mt(); ++i) {
                if (Asub.tileIsLocal(i, j)) {
                    test_assert( Asub(i, j).at(0, 0)
                            == (i1 + i) + (j1 + j) / 10000. );
                    test_assert( Asub(i, j).op() == slate::Op::NoTrans );
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
            for (int j = 0; j < Asub_b.nt(); ++j) {
                for (int i = 0; i < Asub_b.mt(); ++i) {
                    if (Asub_b.tileIsLocal(i, j)) {
                        test_assert( Asub_b(i, j).at(0, 0)
                                == (i1 + i1_b + i) + (j1 + j1_b + j) / 10000. );
                        test_assert( Asub_b(i, j).op() == slate::Op::NoTrans );
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
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

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
    if (Asub.tileIsLocal(0, 0)) {
        test_assert( Asub(0, 0).op() == slate::Op::Trans );
        test_assert( Asub(0, 0).at(0, 0) == 0.0 );
    }

    // 1st column
    Asub = AT.sub( 0, AT.mt()-1, 0, 0 );
    test_assert( Asub.mt() == AT.mt() );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::Trans );
    for (int i = 0; i < Asub.mt(); ++i) {
        if (Asub.tileIsLocal(i, 0)) {
            test_assert( Asub(i, 0).at(0, 0) == i / 10000. );
            test_assert( Asub(i, 0).op() == slate::Op::Trans );
        }
    }

    // 1st row
    Asub = AT.sub( 0, 0, 0, AT.nt()-1 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == AT.nt() );
    test_assert( Asub.op() == slate::Op::Trans );
    for (int j = 0; j < Asub.nt(); ++j) {
        if (Asub.tileIsLocal(0, j)) {
            test_assert( Asub(0, j).at(0, 0) == j );
            test_assert( Asub(0, j).op() == slate::Op::Trans );
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
        for (int j = 0; j < Asub.nt(); ++j) {
            for (int i = 0; i < Asub.mt(); ++i) {
                if (Asub.tileIsLocal(i, j)) {
                    test_assert( Asub(i, j).at(0, 0) == (j1 + j) + (i1 + i) / 10000. );
                    test_assert( Asub(i, j).op() == slate::Op::Trans );
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
            for (int j = 0; j < Asub_b.nt(); ++j) {
                for (int i = 0; i < Asub_b.mt(); ++i) {
                    if (Asub_b.tileIsLocal(i, j)) {
                        test_assert( Asub_b(i, j).at(0, 0) == (j1 + i1_b + i) + (i1 + j1_b + j) / 10000. );
                        test_assert( Asub_b(i, j).op() == slate::Op::NoTrans );
                    }
                }
            }
        }
    }
}

//==============================================================================
// To access BaseMatrix protected members, stick these in the slate::Debug class.
// Admittedly a hack, since this is different than the Debug class in slate_Debug.hh.
namespace slate {
class Debug {
public:

//------------------------------------------------------------------------------
// verify that B == op( A( row1 : row2, col1 : col2 ) )
static void verify_slice(
    slate::Matrix<double>& A,
    int row1, int row2, int col1, int col2,
    int nb, slate::Op trans )
{
    int ib = 0, jb = 0;
    if (trans == slate::Op::NoTrans) {
        test_assert( A.m() == row2 - row1 + 1 );
        test_assert( A.n() == col2 - col1 + 1 );
        test_assert( A.row0_offset() == row1 % nb );
        test_assert( A.col0_offset() == col1 % nb );
        test_assert( A.mt() == ceildiv( int(A.row0_offset() + row2 - row1 + 1), nb ) );
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
        test_assert( A.row0_offset() == row1 % nb );
        test_assert( A.col0_offset() == col1 % nb );
        test_assert( A.mt() == ceildiv( int(A.col0_offset() + col2 - col1 + 1), nb ) );
        test_assert( A.nt() == ceildiv( int(A.row0_offset() + row2 - row1 + 1), nb ) );
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
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                auto T = A(i, j);
                for (int jj = 0; jj < T.nb(); ++jj)
                    for (int ii = 0; ii < T.mb(); ++ii)
                        T.at(ii, jj) = i*nb + ii + (j*nb + jj)/100.;
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
        test_assert( B.row0_offset() == row1 % nb );
        test_assert( B.col0_offset() == col1 % nb );
        test_assert( B.ioffset() == int(row1 / nb) );
        test_assert( B.joffset() == int(col1 / nb) );
        test_assert( B.mt() == ceildiv( int(B.row0_offset() + B.m()), nb ) );
        test_assert( B.nt() == ceildiv( int(B.col0_offset() + B.n()), nb ) );

        verify_slice( B, row1, row2, col1, col2, nb, slate::Op::NoTrans );

        //printf( "BT = AT.slice( ... )\n" );
        auto BT = AT.slice( col1, col2, row1, row2 );

        test_assert( BT.op() == slate::Op::Trans );
        test_assert( BT.m() == n2 );  // trans
        test_assert( BT.n() == m2 );  // trans
        test_assert( BT.row0_offset() == row1 % nb );
        test_assert( BT.col0_offset() == col1 % nb );
        test_assert( BT.ioffset() == int(row1 / nb) );
        test_assert( BT.joffset() == int(col1 / nb) );
        // trans col, row
        test_assert( BT.mt() == ceildiv( int(BT.col0_offset() + BT.m()), nb ) );
        test_assert( BT.nt() == ceildiv( int(BT.row0_offset() + BT.n()), nb ) );

        verify_slice( BT, row1, row2, col1, col2, nb, slate::Op::Trans );

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
        test_assert( C.m() == m3 );
        test_assert( C.n() == n3 );
        test_assert( C.row0_offset() == (row1 + row3) % nb );
        test_assert( C.col0_offset() == (col1 + col3) % nb );
        test_assert( C.ioffset() == int((row1 + row3) / nb) );
        test_assert( C.joffset() == int((col1 + col3) / nb) );
        test_assert( C.mt() == ceildiv( int(C.row0_offset() + C.m()), nb ) );
        test_assert( C.nt() == ceildiv( int(C.col0_offset() + C.n()), nb ) );

        verify_slice( C, row1 + row3, row1 + row4, col1 + col3, col1 + col4,
                      nb, slate::Op::NoTrans );

        //printf( "CT = BT.slice( ... )\n" );
        auto CT = BT.slice( col3, col4, row3, row4 );

        test_assert( CT.op() == slate::Op::Trans );
        test_assert( CT.m() == n3 );  // trans
        test_assert( CT.n() == m3 );  // trans
        test_assert( CT.row0_offset() == (row1 + row3) % nb );
        test_assert( CT.col0_offset() == (col1 + col3) % nb );
        test_assert( CT.ioffset() == int((row1 + row3) / nb) );
        test_assert( CT.joffset() == int((col1 + col3) / nb) );
        // col, row trans
        test_assert( CT.mt() == ceildiv( int(CT.col0_offset() + CT.m()), nb ) );
        test_assert( CT.nt() == ceildiv( int(CT.row0_offset() + CT.n()), nb ) );

        verify_slice( CT, row1 + row3, row1 + row4, col1 + col3, col1 + col4,
                      nb, slate::Op::Trans );
    }
}

}; // class Debug
}  // namespace slate

//==============================================================================
// Conversions

//------------------------------------------------------------------------------
/// Tests Matrix( orig, i1, i2, j1, j2 ).
/// Does the same thing as A.sub( i1, i2, j1, j2 ), just more verbose.
void test_Matrix_to_Matrix()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

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

//------------------------------------------------------------------------------
/// Tests TrapezoidMatrix( uplo, diag, A ),
///       TrapezoidMatrix( uplo, diag, A, i1, i2, j1, j2 ).
void test_Matrix_to_Trapezoid()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // lower
    auto L = slate::TrapezoidMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, A );

    test_assert( L.mt() == A.mt() );
    test_assert( L.nt() == A.nt() );
    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            if (L.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( L(i, j).uplo() == slate::Uplo::Lower );
                else
                    test_assert( L(i, j).uplo() == slate::Uplo::General );
            }
        }
    }

    // upper
    auto U = slate::TrapezoidMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, A );

    test_assert( U.mt() == A.mt() );
    test_assert( U.nt() == A.nt() );
    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            if (U.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( U(i, j).uplo() == slate::Uplo::Upper );
                else
                    test_assert( U(i, j).uplo() == slate::Uplo::General );
            }
        }
    }

    // ----------
    // sub-matrix
    int i1 = rand() % A.mt();
    int i2 = rand() % A.mt();
    int j1 = rand() % A.nt();
    int j2 = rand() % A.nt();
    if (i1 > i2)
        std::swap( i1, i2 );
    if (j1 > j2)
        std::swap( j1, j2 );

    // lower, sub-matrix
    auto L2 = slate::TrapezoidMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, A, i1, i2, j1, j2 );

    test_assert( L2.mt() == i2 - i1 + 1 );
    test_assert( L2.nt() == j2 - j1 + 1);

    // upper, sub-matrix
    auto U2 = slate::TrapezoidMatrix<double>(
        slate::Uplo::Upper , slate::Diag::NonUnit, A, i1, i2, j1, j2 );
    test_assert( U2.mt() == i2 - i1 + 1 );
    test_assert( U2.nt() == j2 - j1 + 1 );
}

//------------------------------------------------------------------------------
/// Tests TriangularMatrix( uplo, diag, A ),
///       TriangularMatrix( uplo, diag, A, i1, i2, j1, j2 ).
void test_Matrix_to_Triangular()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // lower
    auto L = slate::TriangularMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, A );

    test_assert( L.mt() == std::min( A.mt(), A.nt() ) );
    test_assert( L.nt() == std::min( A.mt(), A.nt() ) );
    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            if (L.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( L(i, j).uplo() == slate::Uplo::Lower );
                else
                    test_assert( L(i, j).uplo() == slate::Uplo::General );
            }
        }
    }

    // upper
    auto U = slate::TriangularMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, A );

    test_assert( U.mt() == std::min( A.mt(), A.nt() ) );
    test_assert( U.nt() == std::min( A.mt(), A.nt() ) );
    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            if (U.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( U(i, j).uplo() == slate::Uplo::Upper );
                else
                    test_assert( U(i, j).uplo() == slate::Uplo::General );
            }
        }
    }

    // ----------
    // sub-matrix, must be square
    int i1 = rand() % A.mt();
    int j1 = rand() % A.nt();
    int n2 = rand() % std::min( A.mt() - i1, A.nt() - j1 );
    int i2 = i1 + n2 - 1;
    int j2 = j1 + n2 - 1;

    // lower, sub-matrix
    auto L2 = slate::TriangularMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, A, i1, i2, j1, j2 );

    test_assert( L2.mt() == i2 - i1 + 1 );
    test_assert( L2.nt() == j2 - j1 + 1 );

    // upper, sub-matrix
    auto U2 = slate::TriangularMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, A, i1, i2, j1, j2 );

    test_assert( U2.mt() == i2 - i1 + 1 );
    test_assert( U2.nt() == j2 - j1 + 1 );
}

//------------------------------------------------------------------------------
/// Tests SymmetricMatrix( uplo, diag, A ).
void test_Matrix_to_Symmetric()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // lower
    auto L = slate::SymmetricMatrix<double>(
        slate::Uplo::Lower, A );

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            if (L.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( L(i, j).uplo() == slate::Uplo::Lower );
                else
                    test_assert( L(i, j).uplo() == slate::Uplo::General );
            }
        }
    }

    // upper
    auto U = slate::SymmetricMatrix<double>(
        slate::Uplo::Upper, A );

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            if (U.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( U(i, j).uplo() == slate::Uplo::Upper );
                else
                    test_assert( U(i, j).uplo() == slate::Uplo::General );
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Tests HermitianMatrix( uplo, diag, A ).
void test_Matrix_to_Hermitian()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // lower
    auto L = slate::HermitianMatrix<double>(
        slate::Uplo::Lower, A );

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            if (L.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( L(i, j).uplo() == slate::Uplo::Lower );
                else
                    test_assert( L(i, j).uplo() == slate::Uplo::General );
            }
        }
    }

    // upper
    auto U = slate::HermitianMatrix<double>(
        slate::Uplo::Upper, A );

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            if (U.tileIsLocal(i, j)) {
                if (i == j)
                    test_assert( U(i, j).uplo() == slate::Uplo::Upper );
                else
                    test_assert( U(i, j).uplo() == slate::Uplo::General );
            }
        }
    }
}

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
                        A.tileRecv(i, j, dst);
                        test_assert( T(0, 0) == i + j/1000. + 1000*dst );
                    }
                    else if (mpi_rank == dst) {
                        //printf( "rank %d: recv A(%d, %d) from %d to %d\n",
                        //        mpi_rank, i, j, src, dst );

                        // Receive tile, update, then send updated tile back.
                        A.tileRecv(i, j, src);
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
//     cublas_handle
//     compute_stream
//     comm_stream
//
// Matrix
//     Matrix(orig, i1, i2, j1, j2)
// x   sub(i1, i2, j1, j2)
// x   swap
//     getMaxHostTiles
//     getMaxDeviceTiles
//     allocateBatchArrays
//     reserveHostWorkspace
//     reserveDeviceWorkspace
//     gather

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_Matrix,                 "Matrix()",              mpi_comm);
    run_test(test_Matrix_empty,           "Matrix(m, n, ...)",     mpi_comm);
    run_test(test_Matrix_fromLAPACK,      "Matrix::fromLAPACK",    mpi_comm);
    run_test(test_Matrix_fromScaLAPACK,   "Matrix::fromScaLAPACK", mpi_comm);
    run_test(test_Matrix_fromDevices,     "Matrix::fromDevices",   mpi_comm);
    run_test(test_Matrix_emptyLike,       "Matrix::emptyLike",     mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_Matrix_transpose,       "transpose",      mpi_comm);
    run_test(test_Matrix_conj_transpose,  "conj_transpose", mpi_comm);
    run_test(test_Matrix_swap,            "swap",           mpi_comm);
    run_test(test_Matrix_tileInsert_new,  "Matrix::tileInsert(i, j, dev) ", mpi_comm);
    run_test(test_Matrix_tileInsert_data, "Matrix::tileInsert(i, j, dev, data, lda)",  mpi_comm);
    run_test(test_Matrix_tileLife,        "Matrix::tileLife",  mpi_comm);
    run_test(test_Matrix_tileErase,       "Matrix::tileErase", mpi_comm);
    run_test(test_Matrix_insertLocalTiles,     "Matrix::insertLocalTiles()",           mpi_comm);
    run_test(test_Matrix_insertLocalTiles_dev, "Matrix::insertLocalTiles(on_devices)", mpi_comm);

    if (mpi_rank == 0)
        printf("\nSub-matrices and slices\n");
    run_test(test_Matrix_sub,             "Matrix::sub",       mpi_comm);
    run_test(test_Matrix_sub_trans,       "Matrix::sub(A^T)",  mpi_comm);
    run_test(slate::Debug::test_Matrix_slice, "Matrix::slice", mpi_comm);

    if (mpi_rank == 0)
        printf("\nConversions\n");
    run_test(test_Matrix_to_Matrix,       "Matrix => Matrix",  mpi_comm);
    run_test(test_Matrix_to_Trapezoid,    "Matrix => TrapezoidMatrix",  mpi_comm);
    run_test(test_Matrix_to_Triangular,   "Matrix => TriangularMatrix", mpi_comm);
    run_test(test_Matrix_to_Symmetric,    "Matrix => SymmetricMatrix",  mpi_comm);
    run_test(test_Matrix_to_Hermitian,    "Matrix => HermitianMatrix",  mpi_comm);

    if (mpi_rank == 0)
        printf("\nCommunication\n");
    run_test(test_tileSend_tileRecv, "tileSend, tileRecv", mpi_comm);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    mpi_comm = MPI_COMM_WORLD;

    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    cudaGetDeviceCount(&num_devices);
    host_num = -num_devices;

    // globals
    m  = 200;
    n  = 100;
    k  = 75;
    nb = 16;
    init_process_grid(mpi_size, &p, &q);
    unsigned seed = time( nullptr ) % 10000;  // 4 digit

    // parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i+1 < argc)
            m = atoi( argv[++i] );
        else if (arg == "-m" && i+1 < argc)
            m = atoi( argv[++i] );
        else if (arg == "-n" && i+1 < argc)
            n = atoi( argv[++i] );
        else if (arg == "-k" && i+1 < argc)
            k = atoi( argv[++i] );
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
