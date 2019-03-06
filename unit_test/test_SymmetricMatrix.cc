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

//==============================================================================
// Constructors

//------------------------------------------------------------------------------
/// default constructor
/// Tests SymmetricMatrix(), mt, nt, op, uplo.
void test_SymmetricMatrix()
{
    slate::SymmetricMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::Lower);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor
/// Tests SymmetricMatrix(), mt, nt, op, uplo.
void test_SymmetricMatrix_empty()
{
    slate::SymmetricMatrix<double> L(blas::Uplo::Lower, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    slate::SymmetricMatrix<double> U(blas::Uplo::Upper, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
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
        n, n, nb, // square
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
        n, n, nb, // square
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
        cudaFree(Aarray[dev]);
    }
    delete[] Aarray;
}

//==============================================================================
// Methods

//==============================================================================
// Sub-matrices and conversions

//------------------------------------------------------------------------------
void test_Symmetric_to_Triangular()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    // ----- Lower
    auto A = slate::SymmetricMatrix<double>::fromLAPACK(
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
    auto B = slate::SymmetricMatrix<double>::fromLAPACK(
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
        printf("\nDefault constructors\n");
    run_test(test_SymmetricMatrix,               "SymmetricMatrix()",              mpi_comm);
    run_test(test_SymmetricMatrix_empty,         "SymmetricMatrix(uplo, n, ...)",  mpi_comm);
    run_test(test_SymmetricMatrix_fromLAPACK,    "SymmetricMatrix::fromLAPACK",    mpi_comm);
    run_test(test_SymmetricMatrix_fromScaLAPACK, "SymmetricMatrix::fromScaLAPACK", mpi_comm);
    run_test(test_SymmetricMatrix_fromDevices,   "SymmetricMatrix::fromDevices",   mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");

    if (mpi_rank == 0)
        printf("\nSub-matrices and conversions\n");
    run_test(test_Symmetric_to_Triangular, "Symmetric => Triangular", mpi_comm);
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
