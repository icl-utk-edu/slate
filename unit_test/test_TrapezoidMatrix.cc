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

//------------------------------------------------------------------------------
// global variables
int m, n, k, nb, p, q;
int mpi_rank;
int mpi_size;
MPI_Comm mpi_comm;
int host_num = -1;
int num_devices = 0;

//==============================================================================
// Constructors

//------------------------------------------------------------------------------
/// default constructor
/// Tests TrapezoidMatrix(), mt, nt, op, uplo.
void test_TrapezoidMatrix()
{
    slate::TrapezoidMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::Lower);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor
/// Tests TrapezoidMatrix(), mt, nt, op, uplo, diag.
void test_TrapezoidMatrix_empty()
{
    slate::TrapezoidMatrix<double> L(
        blas::Uplo::Lower, blas::Diag::NonUnit, m, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    slate::TrapezoidMatrix<double> U(
        blas::Uplo::Upper, blas::Diag::Unit, m, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);
}

//------------------------------------------------------------------------------
/// fromLAPACK
/// Test TrapezoidMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_Matrix_fromLAPACK, but adds lower and upper.
void test_TrapezoidMatrix_fromLAPACK()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    //----------
    // lower
    auto L = slate::TrapezoidMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, m, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_lapack(L, i, j, nb, m, n, Ad.data(), lda);
        }
    }

    //----------
    // upper
    auto U = slate::TrapezoidMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, m, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_lapack(U, i, j, nb, m, n, Ad.data(), lda);
        }
    }
}

//------------------------------------------------------------------------------
/// fromScaLAPACK
/// Test TrapezoidMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_Matrix_fromScaLAPACK, but adds lower and upper.
void test_TrapezoidMatrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    //----------
    // lower
    auto L = slate::TrapezoidMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, m, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, m, n, Ad.data(), lda);
        }
    }

    //----------
    // upper
    auto U = slate::TrapezoidMatrix<double>::fromScaLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, m, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_scalapack(U, i, j, nb, m, n, Ad.data(), lda);
        }
    }
}

//------------------------------------------------------------------------------
/// fromDevices
/// Test TrapezoidMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_Matrix_fromDevices, but adds lower and upper.
void test_TrapezoidMatrix_fromDevices()
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

    double** Aarray = new double*[ num_devices ];
    for (int dev = 0; dev < num_devices; ++dev) {
        int ntiles_local2, ntiles_dev, n_dev;
        get_cyclic_dimensions(num_devices, dev, n_local, nb,
                               ntiles_local2, ntiles_dev, n_dev);
        assert(ntiles_local == ntiles_local2);

        // cudaMalloc returns null if len = 0, so make it at least 1.
        size_t len = std::max(sizeof(double) * lda * n_dev, size_t(1));
        slate_cuda_call( cudaMalloc((void**)&Aarray[dev], len) );
        assert(Aarray[dev] != nullptr);
    }

    //----------
    // lower
    auto L = slate::TrapezoidMatrix<double>::fromDevices(
        blas::Uplo::Lower, blas::Diag::NonUnit, m, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_device(L, i, j, nb, m, n, Aarray, lda);
        }
    }

    //----------
    // upper
    auto U = slate::TrapezoidMatrix<double>::fromDevices(
        blas::Uplo::Upper, blas::Diag::Unit, m, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_device(U, i, j, nb, m, n, Aarray, lda);
        }
    }

    for (int dev = 0; dev < num_devices; ++dev) {
        cudaFree(Aarray[dev]);
    }
    delete[] Aarray;
}

//==============================================================================
// Methods

//------------------------------------------------------------------------------
/// Tests insertLocalTiles on host.
void test_TrapezoidMatrix_insertLocalTiles()
{
    slate::TrapezoidMatrix<double> L(
        slate::Uplo::Lower, slate::Diag::NonUnit,
        m, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);

    L.insertLocalTiles();
    for (int j = 0; j < L.nt(); ++j) {
        for (int i = 0; i < L.mt(); ++i) {
            if (i < j) { // upper tiles don't exist
                test_assert(! L.tileExists(i, j));
            }
            else if (L.tileIsLocal(i, j)) {
                auto T = L(i, j);
                test_assert(T.mb() == L.tileMb(i));
                test_assert(T.nb() == L.tileNb(j));
                test_assert(T.stride() == L.tileMb(i));
            }
        }
    }

    //--------------------
    slate::TrapezoidMatrix<double> U(
        slate::Uplo::Upper, slate::Diag::NonUnit,
        m, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);

    U.insertLocalTiles();
    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i < U.mt(); ++i) {
            if (i > j) { // lower tiles don't exist
                test_assert(! U.tileExists(i, j));
            }
            else if (U.tileIsLocal(i, j)) {
                auto T = U(i, j);
                test_assert(T.mb() == U.tileMb(i));
                test_assert(T.nb() == U.tileNb(j));
                test_assert(T.stride() == U.tileMb(i));
            }
        }
    }
}

//==============================================================================
// Sub-matrices and conversions

//------------------------------------------------------------------------------
/// Tests A.sub( i1, i2, j1, j2 ).
void test_TrapezoidMatrix_sub()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    // --------------------
    // Lower
    auto A = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, slate::Diag::NonUnit,
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = j; i < A.mt(); ++i) { // lower
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

    // 1st row -- outside lower triangle, should throw error
    test_assert_throw( Asub = A.sub( 0, 0, 0, A.nt()-1 ), slate::Exception );

    // Arbitrary regions. 70% of time, set i1 <= i2, j1 <= j2, and i1 >= j2.
    // i1 > i2 or j1 > j2 are empty matrices.
    // i1 < j2 is invalid due to overlapping upper triangle.
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
            if (i1 < j2)
                j2 = i1;
        }
        //printf( "sub( %3d, %3d, %3d, %3d ) ", i1, i2, j1, j2 );
        if (i1 < j2) {
            //printf( "invalid\n" );
            test_assert_throw( Asub = A.sub( i1, i2, j1, j2 ), slate::Exception );
        }
        else {
            //if (i2 < i1 || j2 < j1)
            //    printf( "empty\n" );
            //else
            //    printf( "valid\n" );
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

    // --------------------
    // Upper
    A = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, slate::Diag::NonUnit,
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i <= j && i < A.mt(); ++i) { // upper
            if (A.tileIsLocal(i, j)) {
                A(i, j).at(0, 0) = i + j / 10000.;
            }
        }
    }

    // 1st tile
    Asub = A.sub( 0, 0, 0, 0 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::NoTrans );
    if (Asub.tileIsLocal(0, 0)) {
        test_assert( Asub(0, 0).op() == slate::Op::NoTrans );
        test_assert( Asub(0, 0).at(0, 0) == 0.0 );
    }

    // 1st column
    test_assert_throw( Asub = A.sub( 0, A.mt()-1, 0, 0 ), slate::Exception );

    // 1st row -- outside lower triangle, should throw error
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

    // Arbitrary regions. 70% of time, set i1 <= i2, j1 <= j2, and i2 <= j1.
    // i1 > i2 or j1 > j2 are empty matrices.
    // i2 > j1 is invalid due to overlapping lower triangle.
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
            if (i2 > j1)
                i2 = j1;
        }
        //printf( "sub( %3d, %3d, %3d, %3d ) ", i1, i2, j1, j2 );
        if (i2 > j1) {
            //printf( "invalid\n" );
            test_assert_throw( Asub = A.sub( i1, i2, j1, j2 ), slate::Exception );
        }
        else {
            //if (i2 < i1 || j2 < j1)
            //    printf( "empty\n" );
            //else
            //    printf( "valid\n" );
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
}

//------------------------------------------------------------------------------
/// Tests transpose( A ).sub( i1, i2, j1, j2 ).
void test_TrapezoidMatrix_sub_trans()
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

//------------------------------------------------------------------------------
void test_TrapezoidMatrix_to_Trapezoid()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // lower
    auto L = slate::TrapezoidMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, A );

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
void test_TrapezoidMatrix_to_Triangular()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // lower
    auto L = slate::TriangularMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, A );

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
void test_TrapezoidMatrix_to_Symmetric()
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
void test_TrapezoidMatrix_to_Hermitian()
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
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_TrapezoidMatrix,               "TrapezoidMatrix()",                mpi_comm);
    run_test(test_TrapezoidMatrix_empty,         "TrapezoidMatrix(uplo, m, n, ...)", mpi_comm);
    run_test(test_TrapezoidMatrix_fromLAPACK,    "TrapezoidMatrix::fromLAPACK",      mpi_comm);
    run_test(test_TrapezoidMatrix_fromScaLAPACK, "TrapezoidMatrix::fromScaLAPACK",   mpi_comm);
    run_test(test_TrapezoidMatrix_fromDevices,   "TrapezoidMatrix::fromDevices",     mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_TrapezoidMatrix_insertLocalTiles, "TrapezoidMatrix::insertLocalTiles", mpi_comm);

    if (mpi_rank == 0)
        printf("\nSub-matrices and conversions\n");
    run_test(test_TrapezoidMatrix_sub,           "TrapezoidMatrix::sub",                mpi_comm);
    run_test(test_TrapezoidMatrix_sub_trans,     "TrapezoidMatrix::sub(A^T)",           mpi_comm);
    run_test(test_TrapezoidMatrix_to_Trapezoid,  "TrapezoidMatrix => Matrix",           mpi_comm);
    run_test(test_TrapezoidMatrix_to_Trapezoid,  "TrapezoidMatrix => TrapezoidMatrix",  mpi_comm);
    run_test(test_TrapezoidMatrix_to_Triangular, "TrapezoidMatrix => TriangularMatrix", mpi_comm);
    run_test(test_TrapezoidMatrix_to_Symmetric,  "TrapezoidMatrix => SymmetricMatrix",  mpi_comm);
    run_test(test_TrapezoidMatrix_to_Hermitian,  "TrapezoidMatrix => HermitianMatrix",  mpi_comm);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    mpi_comm = MPI_COMM_WORLD;

    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    cudaGetDeviceCount(&num_devices);
    host_num = (num_devices == 0 ? -1 : -num_devices);

    // globals
    m  = 200;
    n  = 100;
    k  = 75;
    nb = 16;
    p  = std::min(2, mpi_size);
    q  = mpi_size / p;
    unsigned seed = time( nullptr ) % 10000;  // 4 digit
    if (argc > 1) { m  = atoi(argv[1]); }
    if (argc > 2) { n  = atoi(argv[2]); }
    if (argc > 3) { k  = atoi(argv[3]); }
    if (argc > 4) { nb = atoi(argv[4]); }
    if (argc > 5) { p  = atoi(argv[5]); }
    if (argc > 6) { q  = atoi(argv[6]); }
    if (argc > 7) { seed = atoi(argv[7]); }
    if (mpi_rank == 0) {
        printf("Usage: %s %4s %4s %4s %4s %4s %4s %4s\n"
               "       %s %4d %4d %4d %4d %4d %4d %4u\n"
               "num_devices = %d\n",
               argv[0], "m", "n", "k", "nb", "p", "q", "seed",
               argv[0], m, n, k, nb, p, q, seed,
               num_devices);
    }

    MPI_Bcast( &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    srand( seed );

    int err = unit_test_main(mpi_comm);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
