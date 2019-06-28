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
/// Tests TriangularMatrix(), mt, nt, op, uplo.
void test_TriangularMatrix()
{
    slate::TriangularMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::Lower);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor
/// Tests TriangularMatrix(), mt, nt, op, uplo, diag.
void test_TriangularMatrix_empty()
{
    // ----------
    // lower
    slate::TriangularMatrix<double> L(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    // ----------
    // upper
    slate::TriangularMatrix<double> U(
        blas::Uplo::Upper, blas::Diag::Unit, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::TriangularMatrix<double> A(
            blas::Uplo::General, blas::Diag::NonUnit, n, nb, p, q, mpi_comm),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// n-by-n, no-data constructor,
/// using lambda functions for tileNb, tileRank, tileDevice.
/// Tests TriangularMatrix(uplo, n, tileNb, ...), m, n, mt, nt, op.
void test_TriangularMatrix_lambda()
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
    slate::TriangularMatrix<double> L(
        slate::Uplo::Lower, blas::Diag::NonUnit, n, tileNb,
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
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == slate::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    // unit diag
    slate::TriangularMatrix<double> Lu(
        slate::Uplo::Lower, blas::Diag::Unit, n, tileNb,
        tileRank, tileDevice, mpi_comm);

    test_assert(Lu.m() == n);
    test_assert(Lu.n() == n);
    test_assert(Lu.op() == blas::Op::NoTrans);
    test_assert(Lu.uplo() == slate::Uplo::Lower);
    test_assert(Lu.diag() == blas::Diag::Unit);

    // ----------
    // upper
    slate::TriangularMatrix<double> U(
        slate::Uplo::Upper, blas::Diag::Unit, n, tileNb,
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
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == slate::Uplo::Upper);
    test_assert(L.diag() == blas::Diag::NonUnit);
    
    // unit diag
    slate::TriangularMatrix<double> Uu(
        slate::Uplo::Upper, blas::Diag::Unit, n, tileNb,
        tileRank, tileDevice, mpi_comm);

    test_assert(Uu.m() == n);
    test_assert(Uu.n() == n);
    test_assert(Uu.op() == blas::Op::NoTrans);
    test_assert(Uu.uplo() == slate::Uplo::Upper);
    test_assert(Uu.diag() == blas::Diag::Unit);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::TriangularMatrix<double> A(
            blas::Uplo::General, blas::Diag::NonUnit, n, tileNb, tileRank, tileDevice, mpi_comm),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromLAPACK
/// Test TriangularMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromLAPACK, but uses n-by-n matrix.
void test_TriangularMatrix_fromLAPACK()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );

    //----------
    // lower
    auto L = slate::TriangularMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_lapack(L, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // upper
    auto U = slate::TriangularMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_lapack(U, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::TriangularMatrix<double>::fromLAPACK(
            blas::Uplo::General, blas::Diag::Unit,
            n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromScaLAPACK
/// Test TriangularMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromScaLAPACK, but uses n-by-n matrix.
void test_TriangularMatrix_fromScaLAPACK()
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
    auto L = slate::TriangularMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // upper
    auto U = slate::TriangularMatrix<double>::fromScaLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, n, Ad.data(), lda, nb,
        p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_scalapack(U, i, j, nb, n, n, Ad.data(), lda);
        }
    }

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::TriangularMatrix<double>::fromScaLAPACK(
            blas::Uplo::General, blas::Diag::Unit,
            n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// fromDevices
/// Test TriangularMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromDevices, but uses n-by-n matrix.
void test_TriangularMatrix_fromDevices()
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
    auto L = slate::TriangularMatrix<double>::fromDevices(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_device(L, i, j, nb, n, n, Aarray, lda);
        }
    }

    //----------
    // upper
    auto U = slate::TriangularMatrix<double>::fromDevices(
        blas::Uplo::Upper, blas::Diag::Unit, n, Aarray, num_devices, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

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
    // uplo=General fails
    test_assert_throw(
        slate::TriangularMatrix<double>::fromDevices(
            blas::Uplo::General, blas::Diag::Unit,
            n, Aarray, num_devices, lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//==============================================================================
// Methods

//------------------------------------------------------------------------------
/// emptyLike
void test_TriangularMatrix_emptyLike()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb,  // square
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::TriangularMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, blas::Diag::Unit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

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
void test_TriangularMatrix_emptyLikeMbNb()
{
    using llong = long long;

    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb,  // square
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::TriangularMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, blas::Diag::Unit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

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
void test_TriangularMatrix_emptyLikeOp()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::TriangularMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, blas::Diag::Unit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto Asub = A.sub( 1, 3 );
    auto Asub_trans = transpose( Asub );

    for (int nb2: std::vector<int>({ 0, 5 })) {
        // ----- no trans
        auto B = Asub.emptyLike( nb2, slate::Op::Trans );

        // just like test_TriangularMatrix_emptyLikeMbNb,
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

//==============================================================================
// Conversion to Triangular

//------------------------------------------------------------------------------
/// Tests TriangularMatrix( uplo, diag, Matrix A ).
///
void test_Triangular_from_Matrix()
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
    // lower, non-unit and unit
    auto Ln = slate::TriangularMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, Asquare );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::NonUnit,
                       min_mt_nt, min_mn, Ln );

    auto Lu = slate::TriangularMatrix<double>(
        slate::Uplo::Lower, slate::Diag::Unit, Asquare );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::Unit,
                       min_mt_nt, min_mn, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TriangularMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, Asquare );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::NonUnit,
                       min_mt_nt, min_mn, Un );

    auto Uu = slate::TriangularMatrix<double>(
        slate::Uplo::Upper, slate::Diag::Unit, Asquare );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::Unit,
                       min_mt_nt, min_mn, Uu );

    // ----------
    // Rectangular tiles (even with square A) should fail.
    if (mb != nb) {
        auto Arect = slate::Matrix<double>::fromLAPACK(
            min_mn, min_mn, Ad.data(), lda, mb, nb, p, q, mpi_comm );

        test_assert_throw(
            auto Lrect = slate::TriangularMatrix<double>(
                slate::Uplo::Lower, slate::Diag::NonUnit, Arect ),
            slate::Exception);

        test_assert_throw(
            auto Urect = slate::TriangularMatrix<double>(
                slate::Uplo::Upper, slate::Diag::NonUnit, Arect ),
            slate::Exception);
    }
}

//------------------------------------------------------------------------------
/// Tests TriangularMatrix( diag, Hermitian A ).
///
void test_Triangular_from_Hermitian()
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
    // lower, non-unit and unit
    auto Ln = slate::TriangularMatrix<double>( slate::Diag::NonUnit, L0 );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::NonUnit,
                       L0.nt(), n, Ln );

    auto Lu = slate::TriangularMatrix<double>( slate::Diag::Unit, L0 );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::Unit,
                       L0.nt(), n, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TriangularMatrix<double>( slate::Diag::NonUnit, U0 );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::NonUnit,
                       U0.nt(), n, Un );

    auto Uu = slate::TriangularMatrix<double>( slate::Diag::Unit, U0 );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::Unit,
                       U0.nt(), n, Uu );
}

//------------------------------------------------------------------------------
/// Tests TriangularMatrix( diag, Symmetric A ).
///
void test_Triangular_from_Symmetric()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );
    auto L0 = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Lower,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U0 = slate::SymmetricMatrix<double>::fromLAPACK(
        slate::Uplo::Upper,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    // ----------
    // lower, non-unit and unit
    auto Ln = slate::TriangularMatrix<double>( slate::Diag::NonUnit, L0 );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::NonUnit,
                       L0.nt(), n, Ln );

    auto Lu = slate::TriangularMatrix<double>( slate::Diag::Unit, L0 );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::Unit,
                       L0.nt(), n, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TriangularMatrix<double>( slate::Diag::NonUnit, U0 );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::NonUnit,
                       U0.nt(), n, Un );

    auto Uu = slate::TriangularMatrix<double>( slate::Diag::Unit, U0 );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::Unit,
                       U0.nt(), n, Uu );
}

//------------------------------------------------------------------------------
/// Tests TriangularMatrix( Trapezoid A ).
///
void test_Triangular_from_Trapezoid()
{
    // todo: when Trapezoid has slice, use it as in test_Triangular_from_Matrix.
    // For now, create as square.
    int64_t min_mn = std::min( m, n );
    
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto L0 = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, slate::Diag::NonUnit,
        min_mn, min_mn, Ad.data(), lda, nb, p, q, mpi_comm );

    auto L1 = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, slate::Diag::Unit,
        min_mn, min_mn, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U0 = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, slate::Diag::NonUnit,
        min_mn, min_mn, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U1 = slate::TrapezoidMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, slate::Diag::Unit,
        min_mn, min_mn, Ad.data(), lda, nb, p, q, mpi_comm );

    int64_t min_mt_nt = std::min( L0.mt(), L0.nt() );

    // ----------
    // lower, non-unit and unit
    auto Ln = slate::TriangularMatrix<double>( L0 );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::NonUnit,
                       min_mt_nt, min_mn, Ln );

    auto Lu = slate::TriangularMatrix<double>( L1 );
    verify_Triangular( slate::Uplo::Lower, slate::Diag::Unit,
                       min_mt_nt, min_mn, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TriangularMatrix<double>( U0 );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::NonUnit,
                       min_mt_nt, min_mn, Un );

    auto Uu = slate::TriangularMatrix<double>( U1 );
    verify_Triangular( slate::Uplo::Upper, slate::Diag::Unit,
                       min_mt_nt, min_mn, Uu );
}

//==============================================================================
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_TriangularMatrix,               "TriangularMatrix()",               mpi_comm);
    run_test(test_TriangularMatrix_empty,         "TriangularMatrix(uplo, n, nb, ...)",     mpi_comm);
    run_test(test_TriangularMatrix_lambda,        "TriangularMatrix(uplo, n, tileNb, ...)", mpi_comm);
    run_test(test_TriangularMatrix_fromLAPACK,    "TriangularMatrix::fromLAPACK",     mpi_comm);
    run_test(test_TriangularMatrix_fromScaLAPACK, "TriangularMatrix::fromScaLAPACK",  mpi_comm);
    run_test(test_TriangularMatrix_fromDevices,   "TriangularMatrix::fromDevices",    mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_TriangularMatrix_emptyLike,     "TriangularMatrix::emptyLike()",           mpi_comm);
    run_test(test_TriangularMatrix_emptyLikeMbNb, "TriangularMatrix::emptyLike(nb)",         mpi_comm);
    run_test(test_TriangularMatrix_emptyLikeOp,   "TriangularMatrix::emptyLike(..., Trans)", mpi_comm);

    if (mpi_rank == 0)
        printf("\nSub-matrices\n");

    if (mpi_rank == 0)
        printf("\nConversion to Triangular\n");
    run_test(test_Triangular_from_Matrix,      "TriangularMatrix( uplo, diag, Matrix )",    mpi_comm);
    run_test(test_Triangular_from_Hermitian,   "TriangularMatrix( diag, HermitianMatrix )", mpi_comm);
    run_test(test_Triangular_from_Symmetric,   "TriangularMatrix( diag, SymmetricMatrix )", mpi_comm);
    run_test(test_Triangular_from_Trapezoid,   "TriangularMatrix( TrapezoidMatrix )",       mpi_comm);
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
