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
#include "slate/TriangularBandMatrix.hh"
#include "slate/internal/util.hh"

#include "unit_test.hh"
#include "util_matrix.hh"

using slate::ceildiv;
using slate::roundup;

//------------------------------------------------------------------------------
// global variables
int m, n, k, mb, nb, kd, p, q;
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
/// Tests TriangularBandMatrix(), mt, nt, op, uplo.
void test_TriangularBandMatrix()
{
    slate::TriangularBandMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.bandwidth() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::Lower);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor
/// Tests TriangularBandMatrix(), mt, nt, op, uplo, diag.
void test_TriangularBandMatrix_empty()
{
    // ----------
    // lower
    slate::TriangularBandMatrix<double> L(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, kd, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.bandwidth() == kd);
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    // ----------
    // upper
    slate::TriangularBandMatrix<double> U(
        blas::Uplo::Upper, blas::Diag::Unit, n, kd, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.bandwidth() == kd);
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::TriangularBandMatrix<double> A(
            blas::Uplo::General, blas::Diag::NonUnit, n, kd, nb, p, q, mpi_comm),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor
/// Tests TriangularBandMatrix(), mt, nt, op, uplo, diag.
void test_TriangularBandMatrix_trans()
{
    // ----------
    // lower
    slate::TriangularBandMatrix<double> Ln(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, kd, nb, p, q, mpi_comm);

    verify_TriangularBand( slate::Uplo::Lower, slate::Diag::NonUnit, slate::Op::NoTrans,
                           ceildiv(n, nb), n, kd, Ln );

    auto Lnt = transpose(Ln);

    verify_TriangularBand( slate::Uplo::Upper, slate::Diag::NonUnit, slate::Op::Trans,
                           ceildiv(n, nb), n, kd, Lnt );

    // ----------
    // upper
    slate::TriangularBandMatrix<double> Uu(
        slate::Uplo::Upper, slate::Diag::Unit, n, kd, nb, p, q, mpi_comm);

    verify_TriangularBand( slate::Uplo::Upper, slate::Diag::Unit, slate::Op::NoTrans,
                           ceildiv(n, nb), n, kd, Uu );

    auto Uut = transpose(Uu);

    verify_TriangularBand( slate::Uplo::Lower, slate::Diag::Unit, slate::Op::Trans,
                           ceildiv(n, nb), n, kd, Uut );
}

//------------------------------------------------------------------------------
/// n-by-n, no-data constructor,
/// using lambda functions for tileNb, tileRank, tileDevice.
/// Tests TriangularBandMatrix(uplo, n, tileNb, ...), m, n, mt, nt, op.
void test_TriangularBandMatrix_lambda()
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
    slate::TriangularBandMatrix<double> L(
        slate::Uplo::Lower, blas::Diag::NonUnit, n, kd, tileNb,
        tileRank, tileDevice, mpi_comm);

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
    test_assert(L.bandwidth() == kd);
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == slate::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    // unit diag
    slate::TriangularBandMatrix<double> Lu(
        slate::Uplo::Lower, blas::Diag::Unit, n, kd, tileNb,
        tileRank, tileDevice, mpi_comm);

    test_assert(Lu.m() == n);
    test_assert(Lu.n() == n);
    test_assert(Lu.bandwidth() == kd);
    test_assert(Lu.op() == blas::Op::NoTrans);
    test_assert(Lu.uplo() == slate::Uplo::Lower);
    test_assert(Lu.diag() == blas::Diag::Unit);

    // ----------
    // upper
    slate::TriangularBandMatrix<double> U(
        slate::Uplo::Upper, blas::Diag::Unit, n, kd, tileNb,
        tileRank, tileDevice, mpi_comm);

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
    test_assert(U.bandwidth() == kd);
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == slate::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    // unit diag
    slate::TriangularBandMatrix<double> Uu(
        slate::Uplo::Upper, blas::Diag::Unit, n, kd, tileNb,
        tileRank, tileDevice, mpi_comm);

    test_assert(Uu.m() == n);
    test_assert(Uu.n() == n);
    test_assert(Uu.bandwidth() == kd);
    test_assert(Uu.op() == blas::Op::NoTrans);
    test_assert(Uu.uplo() == slate::Uplo::Upper);
    test_assert(Uu.diag() == blas::Diag::Unit);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::TriangularBandMatrix<double> A(
            blas::Uplo::General, blas::Diag::NonUnit, n, kd, tileNb, tileRank, tileDevice, mpi_comm),
        slate::Exception);
}

//==============================================================================
// Sub-matrices

//==============================================================================
// Conversion to Triangular

//------------------------------------------------------------------------------
/// Tests TriangularMatrix( uplo, diag, Matrix A ).
///
void test_TriangularBand_from_BandMatrix()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto mt = ceildiv(m, nb);
    auto nt = ceildiv(n, nb);

    auto op = blas::Op::NoTrans;

    auto A = slate::BandMatrix<double>( m, n, kd, kd, nb, p, q, mpi_comm );
    verify_Band( op, mt, nt, m, n, kd, kd, A );

    int64_t min_mt_nt = std::min( A.mt(), A.nt() );
    int64_t min_mn = std::min( A.m(), A.n() );

    // Make square A.
    auto Asquare = A.slice( 0, min_mn-1, 0, min_mn-1 );
    verify_Band( op, min_mt_nt, min_mt_nt, min_mn, min_mn, kd, kd, Asquare );

    // ----------
    // lower, non-unit and unit
    auto Ln = slate::TriangularBandMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, Asquare );
    verify_TriangularBand( slate::Uplo::Lower, slate::Diag::NonUnit, op,
                           min_mt_nt, min_mn, kd, Ln );

    auto Lu = slate::TriangularBandMatrix<double>(
        slate::Uplo::Lower, slate::Diag::Unit, Asquare );
    verify_TriangularBand( slate::Uplo::Lower, slate::Diag::Unit, op,
                           min_mt_nt, min_mn, kd, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TriangularBandMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, Asquare );
    verify_TriangularBand( slate::Uplo::Upper, slate::Diag::NonUnit, op,
                           min_mt_nt, min_mn, kd, Un );

    auto Uu = slate::TriangularBandMatrix<double>(
        slate::Uplo::Upper, slate::Diag::Unit, Asquare );
    verify_TriangularBand( slate::Uplo::Upper, slate::Diag::Unit, op,
                           min_mt_nt, min_mn, kd, Uu );

}

//==============================================================================
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_TriangularBandMatrix,               "TriangularBandMatrix()",               mpi_comm);
    run_test(test_TriangularBandMatrix_empty,         "TriangularBandMatrix(uplo, n, nb, ...)",     mpi_comm);
    run_test(test_TriangularBandMatrix_lambda,        "TriangularBandMatrix(uplo, n, tileNb, ...)", mpi_comm);

    if (mpi_rank == 0)
        printf("\nTrans\n");
    run_test(test_TriangularBandMatrix_trans,        "transpose(TriangularBandMatrix(...))", mpi_comm);


    if (mpi_rank == 0)
        printf("\nConversion to Triangular\n");
    run_test(test_TriangularBand_from_BandMatrix,      "TriangularBandMatrix( uplo, diag, BandMatrix )",    mpi_comm);
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
    kd = 10;
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
        else if (arg == "-kd" && i+1 < argc)
            kd = atoi( argv[++i] );
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
        printf("Usage: %s [-m %d] [-n %d] [-k %d] [-kd %d] [-nb %d] [-p %d] [-q %d] [-seed %d] [-v]\n"
               "num_devices = %d\n",
               argv[0], m, n, k, kd, nb, p, q, seed,
               num_devices);
    }

    MPI_Bcast( &seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
    srand( seed );

    int err = unit_test_main(mpi_comm);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
