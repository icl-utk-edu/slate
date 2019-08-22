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
int host_num = slate::HostNum;
int num_devices = 0;

//==============================================================================
// Constructors

//------------------------------------------------------------------------------
/// default constructor
/// Tests HermitianMatrix(), mt, nt, op, uplo.
void test_HermitianMatrix()
{
    slate::HermitianMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::Lower);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor
/// Tests HermitianMatrix(), mt, nt, op, uplo.
void test_HermitianMatrix_empty()
{
    slate::HermitianMatrix<double> L(blas::Uplo::Lower, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    slate::HermitianMatrix<double> U(blas::Uplo::Upper, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    test_assert_throw(
        slate::HermitianMatrix<double> A(blas::Uplo::General, n, nb, p, q, mpi_comm),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromLAPACK
/// Test HermitianMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromLAPACK.
void test_HermitianMatrix_fromLAPACK()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    //----------
    // lower
    auto L = slate::HermitianMatrix<double>::fromLAPACK(
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
    auto U = slate::HermitianMatrix<double>::fromLAPACK(
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
    // general
    test_assert_throw(
        slate::HermitianMatrix<double>::fromLAPACK(
            blas::Uplo::General, n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromScaLAPACK
/// Test HermitianMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromScaLAPACK.
void test_HermitianMatrix_fromScaLAPACK()
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
    auto L = slate::HermitianMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // upper
    auto U = slate::HermitianMatrix<double>::fromScaLAPACK(
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
    // general
    test_assert_throw(
        slate::HermitianMatrix<double>::fromScaLAPACK(
            blas::Uplo::General, n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromDevices
/// Test HermitianMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromDevices.
void test_HermitianMatrix_fromDevices()
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
    auto L = slate::HermitianMatrix<double>::fromDevices(
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
    auto U = slate::HermitianMatrix<double>::fromDevices(
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
        cudaFree(Aarray[dev]);
    }
    delete[] Aarray;

    //----------
    // general
    test_assert_throw(
        slate::HermitianMatrix<double>::fromDevices(
            blas::Uplo::General, n, Aarray, num_devices, lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//==============================================================================
// Methods

//==============================================================================
// Sub-matrices and conversions

//------------------------------------------------------------------------------
/// Tests A.sub( i1, i2 ).
///
void test_Hermitian_sub()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::HermitianMatrix<double>::fromLAPACK(
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
void test_Hermitian_sub_trans()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::HermitianMatrix<double>::fromLAPACK(
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
void test_Hermitian_slice()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Mark entries so they're identifiable.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Ad[ i + j*lda ] = i + j / 10000.;

    // Arbitrary regions.
    // Currently, enforce i2 >= i1.
    // todo: allow i2 < i1 for empty matrix, as in test_Hermitian_sub?
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
void test_Hermitian_slice_offdiag()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    auto L = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Mark entries so they're identifiable.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Ad[ i + j*lda ] = i + j / 10000.;

    // Arbitrary regions.
    // For upper: row1 <= row2 <= col1 <= col2.
    // todo: allow row2 > row1, col2 > col1 for empty matrix, as in test_Hermitian_sub?
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

//------------------------------------------------------------------------------
void test_Hermitian_to_Triangular()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    // ----- Lower
    auto A = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // okay: square, lower [ 1:mt-1, 0:nt-2 ]
    auto L1 = slate::TriangularMatrix<double>(
        slate::Diag::NonUnit, A,   1, A.mt()-1,   0, A.nt()-2 );

    // fail: non-square [ 0:mt-1, 0:nt-2 ]
    test_assert_throw_std(
    auto L2 = slate::TriangularMatrix<double>(
        slate::Diag::NonUnit, A,   0, A.mt()-1,   0, A.nt()-2 ));

    // fail: top-left (0, 1) is upper
    test_assert_throw_std(
    auto L3 = slate::TriangularMatrix<double>(
        slate::Diag::NonUnit, A,   0, A.mt()-2,   1, A.nt()-1 ));

    // ----- Upper
    auto B = slate::HermitianMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // okay: square, upper
    auto U1 = slate::TriangularMatrix<double>(
        slate::Diag::NonUnit, B,   0, B.mt()-2,   1, B.nt()-1 );

    // fail: non-square
    test_assert_throw_std(
    auto U2 = slate::TriangularMatrix<double>(
        slate::Diag::NonUnit, B,   0, B.mt()-1,   1, B.nt()-1 ));

    // fail: top-left (1, 0) is lower
    test_assert_throw_std(
    auto U3 = slate::TriangularMatrix<double>(
        slate::Diag::NonUnit, B,   1, B.mt()-1,   0, B.nt()-2 ));
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_HermitianMatrix,               "HermitianMatrix()",              mpi_comm);
    run_test(test_HermitianMatrix_empty,         "HermitianMatrix(uplo, n, ...)",  mpi_comm);
    run_test(test_HermitianMatrix_fromLAPACK,    "HermitianMatrix::fromLAPACK",    mpi_comm);
    run_test(test_HermitianMatrix_fromScaLAPACK, "HermitianMatrix::fromScaLAPACK", mpi_comm);
    run_test(test_HermitianMatrix_fromDevices,   "HermitianMatrix::fromDevices",   mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");

    if (mpi_rank == 0)
        printf("\nSub-matrices and conversions\n");
    run_test(test_Hermitian_sub,           "HermitianMatrix::sub",                   mpi_comm);
    run_test(test_Hermitian_sub_trans,     "HermitianMatrix::sub(A^T)",              mpi_comm);
    run_test(test_Hermitian_slice,         "HermitianMatrix::slice(i1, i2)",         mpi_comm);
    run_test(test_Hermitian_slice_offdiag, "HermitianMatrix::slice(i1, i2, j1, j2)", mpi_comm);
    run_test(test_Hermitian_to_Triangular, "Hermitian => Triangular", mpi_comm);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    mpi_comm = MPI_COMM_WORLD;

    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    cudaGetDeviceCount(&num_devices);
    host_num = slate::HostNum;

    // globals
    m  = 200;
    n  = 100;
    k  = 75;
    nb = 16;
    init_process_grid(mpi_size, &p, &q);
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
