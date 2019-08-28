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
int m, n, k, mb, nb, p, q;
int mpi_rank;
int mpi_size;
MPI_Comm mpi_comm;
int host_num = slate::HostNum;
int num_devices = 0;
int verbose = 0;

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
/// n-by-n, no-data constructor
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

    test_assert_throw(
        slate::SymmetricMatrix<double> A(blas::Uplo::General, n, nb, p, q, mpi_comm),
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
    // general
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
    // general
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

    //----------
    // general
    test_assert_throw(
        slate::SymmetricMatrix<double>::fromDevices(
            blas::Uplo::General, n, Aarray, num_devices, lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//==============================================================================
// Methods

//==============================================================================
// Sub-matrices

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
            auto L = slate::SymmetricMatrix<double>(
                slate::Uplo::Lower, A ),
            slate::Exception);

        test_assert_throw(
            auto U = slate::SymmetricMatrix<double>(
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
            auto L = slate::SymmetricMatrix<double>( L0rect ),
            slate::Exception);

        test_assert_throw(
            auto U = slate::SymmetricMatrix<double>( U0rect ),
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
    run_test(test_SymmetricMatrix,               "SymmetricMatrix()",              mpi_comm);
    run_test(test_SymmetricMatrix_empty,         "SymmetricMatrix(uplo, n, nb, ...)",     mpi_comm);
    run_test(test_SymmetricMatrix_lambda,        "SymmetricMatrix(uplo, n, tileNb, ...)", mpi_comm);
    run_test(test_SymmetricMatrix_fromLAPACK,    "SymmetricMatrix::fromLAPACK",    mpi_comm);
    run_test(test_SymmetricMatrix_fromScaLAPACK, "SymmetricMatrix::fromScaLAPACK", mpi_comm);
    run_test(test_SymmetricMatrix_fromDevices,   "SymmetricMatrix::fromDevices",   mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");

    if (mpi_rank == 0)
        printf("\nSub-matrices\n");

    if (mpi_rank == 0)
        printf("\nConversion to Symmetric\n");
    run_test(test_Symmetric_from_Matrix,     "SymmetricMatrix( uplo, Matrix )",     mpi_comm);
    run_test(test_Symmetric_from_Hermitian,  "SymmetricMatrix( HermitianMatrix )",  mpi_comm);
    run_test(test_Symmetric_from_Trapezoid,  "SymmetricMatrix( TrapezoidMatrix )",  mpi_comm);
    run_test(test_Symmetric_from_Triangular, "SymmetricMatrix( TriangularMatrix )", mpi_comm);
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
