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
    // ----------
    // lower
    slate::TrapezoidMatrix<double> L(
        blas::Uplo::Lower, blas::Diag::NonUnit, m, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    // ----------
    // upper
    slate::TrapezoidMatrix<double> U(
        blas::Uplo::Upper, blas::Diag::Unit, m, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::TrapezoidMatrix<double> A(
            blas::Uplo::General, blas::Diag::NonUnit, m, n, nb, p, q, mpi_comm),
        slate::Exception);
}

//------------------------------------------------------------------------------
/// m-by-n, no-data constructor,
/// using lambda functions for tileNb, tileRank, tileDevice.
/// Tests TrapezoidMatrix(uplo, n, tileNb, ...), m, n, mt, nt, op.
void test_TrapezoidMatrix_lambda()
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
    slate::TrapezoidMatrix<double> L(
        slate::Uplo::Lower, blas::Diag::NonUnit, m, n, tileNb,
        tileRank, tileDevice, mpi_comm);

    // verify mt, tileMb(i), and sum tileMb(i) == m
    int ii = 0;
    for (int i = 0; i < L.mt(); ++i) {
        test_assert( L.tileMb(i) == blas::min( tileNb(i), m - ii ) );
        ii += L.tileMb(i);
    }
    test_assert( ii == m );

    // verify nt, tileNb(j), and sum tileNb(j) == n
    int jj = 0;
    for (int j = 0; j < L.nt(); ++j) {
        test_assert( L.tileNb(j) == blas::min( tileNb(j), n - jj ) );
        jj += L.tileNb(j);
    }
    test_assert( jj == n );

    test_assert(L.m() == m);
    test_assert(L.n() == n);
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == slate::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    // unit diag
    slate::TrapezoidMatrix<double> Lu(
        slate::Uplo::Lower, blas::Diag::Unit, m, n, tileNb,
        tileRank, tileDevice, mpi_comm);

    test_assert(Lu.m() == m);
    test_assert(Lu.n() == n);
    test_assert(Lu.op() == blas::Op::NoTrans);
    test_assert(Lu.uplo() == slate::Uplo::Lower);
    test_assert(Lu.diag() == blas::Diag::Unit);

    // ----------
    // upper
    slate::TrapezoidMatrix<double> U(
        slate::Uplo::Upper, blas::Diag::NonUnit, m, n, tileNb,
        tileRank, tileDevice, mpi_comm);

    // verify mt, tileNb(i), and sum tileNb(i) == n
    ii = 0;
    for (int i = 0; i < U.mt(); ++i) {
        test_assert( U.tileMb(i) == blas::min( tileNb(i), m - ii ) );
        ii += U.tileMb(i);
    }
    test_assert( ii == m );

    // verify nt, tileNb(j), and sum tileNb(j) == n
    jj = 0;
    for (int j = 0; j < U.nt(); ++j) {
        test_assert( U.tileNb(j) == blas::min( tileNb(j), n - jj ) );
        jj += U.tileNb(j);
    }
    test_assert( jj == n );

    test_assert(U.m() == m);
    test_assert(U.n() == n);
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == slate::Uplo::Upper);
    test_assert(L.diag() == blas::Diag::NonUnit);
    
    // unit diag
    slate::TrapezoidMatrix<double> Uu(
        slate::Uplo::Upper, blas::Diag::Unit, m, n, tileNb,
        tileRank, tileDevice, mpi_comm);

    test_assert(Uu.m() == m);
    test_assert(Uu.n() == n);
    test_assert(Uu.op() == blas::Op::NoTrans);
    test_assert(Uu.uplo() == slate::Uplo::Upper);
    test_assert(Uu.diag() == blas::Diag::Unit);

    // ----------
    // uplo=General fails
    test_assert_throw(
        slate::TrapezoidMatrix<double> A(
            blas::Uplo::General, blas::Diag::NonUnit,
            m, n, tileNb, tileRank, tileDevice, mpi_comm),
        slate::Exception);
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

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::TrapezoidMatrix<double>::fromLAPACK(
            blas::Uplo::General, blas::Diag::Unit,
            m, n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
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

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::TrapezoidMatrix<double>::fromScaLAPACK(
            blas::Uplo::General, blas::Diag::Unit,
            m, n, Ad.data(), lda, nb, p, q, mpi_comm ),
        slate::Exception);
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

    //----------
    // uplo=General fails
    test_assert_throw(
        slate::TrapezoidMatrix<double>::fromDevices(
            blas::Uplo::General, blas::Diag::Unit,
            m, n, Aarray, num_devices, lda, nb, p, q, mpi_comm ),
        slate::Exception);
}

//==============================================================================
// Methods

//------------------------------------------------------------------------------
/// emptyLike
void test_TrapezoidMatrix_emptyLike()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::TrapezoidMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, blas::Diag::Unit,
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

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
/// emptyLike with nb overriding size.
void test_TrapezoidMatrix_emptyLikeMbNb()
{
    using llong = long long;

    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb, nb,  // square
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::TrapezoidMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, blas::Diag::Unit,
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

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
/// emptyLike with nb overriding size, and op to deep transpose.
void test_TrapezoidMatrix_emptyLikeOp()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n, nb, nb,
        mtiles, mtiles_local, m_local,
        ntiles, ntiles_local, n_local, lda );

    std::vector<double> Ad( lda*n_local );

    auto A = slate::TrapezoidMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, blas::Diag::Unit,
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto Asub = A.sub( 1, 3 );
    auto Asub_trans = transpose( Asub );

    for (int nb2: std::vector<int>({ 0, 5 })) {
        // ----- no trans
        auto B = Asub.emptyLike( nb2, slate::Op::Trans );

        // just like test_TrapezoidMatrix_emptyLikeMbNb,
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

//------------------------------------------------------------------------------
/// Tests insertLocalTiles on host.
void test_TrapezoidMatrix_insertLocalTiles()
{
    //--------------------
    // lower
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
    // upper
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

//------------------------------------------------------------------------------
/// Test allocateBatchArrays, clearBatchArrays, batchArraySize.
///
void test_TrapezoidMatrix_allocateBatchArrays()
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );

    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, Ad.size(), Ad.data() );

    auto L = slate::TrapezoidMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit,
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U = slate::TrapezoidMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit,
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // initially, batch arrays are null
    test_assert( L.batchArraySize() == 0 );
    test_assert( U.batchArraySize() == 0 );
    for (int device = 0; device < num_devices; ++device) {
        test_assert( L.a_array_host(device) == nullptr );
        test_assert( L.b_array_host(device) == nullptr );
        test_assert( L.c_array_host(device) == nullptr );

        test_assert( L.a_array_device(device) == nullptr );
        test_assert( L.b_array_device(device) == nullptr );
        test_assert( L.c_array_device(device) == nullptr );

        // -----
        test_assert( U.a_array_host(device) == nullptr );
        test_assert( U.b_array_host(device) == nullptr );
        test_assert( U.c_array_host(device) == nullptr );

        test_assert( U.a_array_device(device) == nullptr );
        test_assert( U.b_array_device(device) == nullptr );
        test_assert( U.c_array_device(device) == nullptr );
    }

    // allocate size 10
    L.allocateBatchArrays( 10 );
    U.allocateBatchArrays( 10 );
    test_assert( L.batchArraySize() == 10 );
    test_assert( U.batchArraySize() == 10 );
    for (int device = 0; device < num_devices; ++device) {
        test_assert( L.a_array_host(device) != nullptr );
        test_assert( L.b_array_host(device) != nullptr );
        test_assert( L.c_array_host(device) != nullptr );

        test_assert( L.a_array_device(device) != nullptr );
        test_assert( L.b_array_device(device) != nullptr );
        test_assert( L.c_array_device(device) != nullptr );

        // -----
        test_assert( U.a_array_host(device) != nullptr );
        test_assert( U.b_array_host(device) != nullptr );
        test_assert( U.c_array_host(device) != nullptr );

        test_assert( U.a_array_device(device) != nullptr );
        test_assert( U.b_array_device(device) != nullptr );
        test_assert( U.c_array_device(device) != nullptr );
    }

    // increase to size 20
    L.allocateBatchArrays( 20 );
    U.allocateBatchArrays( 20 );
    test_assert( L.batchArraySize() == 20 );
    test_assert( U.batchArraySize() == 20 );

    // requesting 15 should leave it at 20
    L.allocateBatchArrays( 15 );
    U.allocateBatchArrays( 15 );
    test_assert( L.batchArraySize() == 20 );
    test_assert( U.batchArraySize() == 20 );

    int numL = 0;
    int numU = 0;
    for (int device = 0; device < num_devices; ++device) {
        numL = blas::max( numL, L.getMaxDeviceTiles( device ) );
        numU = blas::max( numU, U.getMaxDeviceTiles( device ) );
    }

    // request enough for local tiles
    L.allocateBatchArrays();
    U.allocateBatchArrays();
    test_assert( L.batchArraySize() == blas::max( numL, 20 ) );
    test_assert( U.batchArraySize() == blas::max( numU, 20 ) );

    // clear should free arrays
    L.clearBatchArrays();
    U.clearBatchArrays();
    test_assert( L.batchArraySize() == 0 );
    test_assert( U.batchArraySize() == 0 );
    for (int device = 0; device < num_devices; ++device) {
        test_assert( L.a_array_host(device) == nullptr );
        test_assert( L.b_array_host(device) == nullptr );
        test_assert( L.c_array_host(device) == nullptr );

        test_assert( L.a_array_device(device) == nullptr );
        test_assert( L.b_array_device(device) == nullptr );
        test_assert( L.c_array_device(device) == nullptr );

        // -----
        test_assert( U.a_array_host(device) == nullptr );
        test_assert( U.b_array_host(device) == nullptr );
        test_assert( U.c_array_host(device) == nullptr );

        test_assert( U.a_array_device(device) == nullptr );
        test_assert( U.b_array_device(device) == nullptr );
        test_assert( U.c_array_device(device) == nullptr );
    }
}

//==============================================================================
// Sub-matrices

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

//==============================================================================
// Conversion to Trapezoid

//------------------------------------------------------------------------------
/// Tests TrapezoidMatrix( uplo, diag, Matrix A ).
///
void test_Trapezoid_from_Matrix()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // Take sub-matrix, offset by 1 tile.
    A = A.sub( 0, A.mt()-1, 1, A.nt()-1 );
    int64_t mt = A.mt();
    int64_t nt = A.nt();
    int64_t m_ = A.m();
    int64_t n_ = A.n();

    // ----------
    // lower, non-unit and unit
    auto Ln = slate::TrapezoidMatrix<double>(
        slate::Uplo::Lower, slate::Diag::NonUnit, A );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::NonUnit,
                      mt, nt, m_, n_, Ln );

    auto Lu = slate::TrapezoidMatrix<double>(
        slate::Uplo::Lower, slate::Diag::Unit, A );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::Unit,
                      mt, nt, m_, n_, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TrapezoidMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, A );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::NonUnit,
                      mt, nt, m_, n_, Un );

    auto Uu = slate::TrapezoidMatrix<double>(
        slate::Uplo::Upper, slate::Diag::Unit, A );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::Unit,
                      mt, nt, m_, n_, Uu );

    // ----------
    // Rectangular tiles should fail.
    if (mb != nb) {
        auto Arect = slate::Matrix<double>::fromLAPACK(
            m, n, Ad.data(), lda, mb, nb, p, q, mpi_comm );

        test_assert_throw(
            auto Lrect = slate::TrapezoidMatrix<double>(
                slate::Uplo::Lower, slate::Diag::NonUnit, Arect ),
            slate::Exception);

        test_assert_throw(
            auto Urect = slate::TrapezoidMatrix<double>(
                slate::Uplo::Upper, slate::Diag::NonUnit, Arect ),
            slate::Exception);
    }
}

//------------------------------------------------------------------------------
/// Tests TrapezoidMatrix( diag, Hermitian A ).
///
void test_Trapezoid_from_Hermitian()
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
    auto Ln = slate::TrapezoidMatrix<double>( slate::Diag::NonUnit, L0 );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::NonUnit,
                      L0.mt(), L0.nt(), n, n, Ln );

    auto Lu = slate::TrapezoidMatrix<double>( slate::Diag::Unit, L0 );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::Unit,
                      L0.mt(), L0.nt(), n, n, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TrapezoidMatrix<double>( slate::Diag::NonUnit, U0 );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::NonUnit,
                      U0.mt(), U0.nt(), n, n, Un );

    auto Uu = slate::TrapezoidMatrix<double>( slate::Diag::Unit, U0 );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::Unit,
                      U0.mt(), U0.nt(), n, n, Uu );
}

//------------------------------------------------------------------------------
/// Tests TrapezoidMatrix( diag, Symmetric A ).
///
void test_Trapezoid_from_Symmetric()
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
    auto Ln = slate::TrapezoidMatrix<double>( slate::Diag::NonUnit, L0 );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::NonUnit,
                      L0.mt(), L0.nt(), n, n, Ln );

    auto Lu = slate::TrapezoidMatrix<double>( slate::Diag::Unit, L0 );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::Unit,
                      L0.mt(), L0.nt(), n, n, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TrapezoidMatrix<double>( slate::Diag::NonUnit, U0 );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::NonUnit,
                      U0.mt(), U0.nt(), n, n, Un );

    auto Uu = slate::TrapezoidMatrix<double>( slate::Diag::Unit, U0 );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::Unit,
                      U0.mt(), U0.nt(), n, n, Uu );
}

//------------------------------------------------------------------------------
/// Tests TrapezoidMatrix( Triangular A ).
///
void test_Trapezoid_from_Triangular()
{
    int lda = roundup(n, nb);
    std::vector<double> Ad( lda*n );
    auto L0 = slate::TriangularMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, slate::Diag::NonUnit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto L1 = slate::TriangularMatrix<double>::fromLAPACK(
        slate::Uplo::Lower, slate::Diag::Unit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U0 = slate::TriangularMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, slate::Diag::NonUnit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    auto U1 = slate::TriangularMatrix<double>::fromLAPACK(
        slate::Uplo::Upper, slate::Diag::Unit,
        n, Ad.data(), lda, nb, p, q, mpi_comm );

    // ----------
    // lower, non-unit and unit
    auto Ln = slate::TrapezoidMatrix<double>( L0 );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::NonUnit,
                      L0.mt(), L0.nt(), n, n, Ln );

    auto Lu = slate::TrapezoidMatrix<double>( L1 );
    verify_Trapezoid( slate::Uplo::Lower, slate::Diag::Unit,
                      L1.mt(), L1.nt(), n, n, Lu );

    // ----------
    // upper, non-unit and unit
    auto Un = slate::TrapezoidMatrix<double>( U0 );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::NonUnit,
                      U0.mt(), U0.nt(), n, n, Un );

    auto Uu = slate::TrapezoidMatrix<double>( U1 );
    verify_Trapezoid( slate::Uplo::Upper, slate::Diag::Unit,
                      U1.mt(), U1.nt(), n, n, Uu );
}

//==============================================================================
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nConstructors\n");
    run_test(test_TrapezoidMatrix,               "TrapezoidMatrix()",                mpi_comm);
    run_test(test_TrapezoidMatrix_empty,         "TrapezoidMatrix(uplo, m, n, nb, ...)",     mpi_comm);
    run_test(test_TrapezoidMatrix_lambda,        "TrapezoidMatrix(uplo, m, n, tileNb, ...)", mpi_comm);
    run_test(test_TrapezoidMatrix_fromLAPACK,    "TrapezoidMatrix::fromLAPACK",      mpi_comm);
    run_test(test_TrapezoidMatrix_fromScaLAPACK, "TrapezoidMatrix::fromScaLAPACK",   mpi_comm);
    run_test(test_TrapezoidMatrix_fromDevices,   "TrapezoidMatrix::fromDevices",     mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_TrapezoidMatrix_emptyLike,           "TrapezoidMatrix::emptyLike()",           mpi_comm);
    run_test(test_TrapezoidMatrix_emptyLikeMbNb,       "TrapezoidMatrix::emptyLike(nb)",         mpi_comm);
    run_test(test_TrapezoidMatrix_emptyLikeOp,         "TrapezoidMatrix::emptyLike(..., Trans)", mpi_comm);
    run_test(test_TrapezoidMatrix_insertLocalTiles,    "TrapezoidMatrix::insertLocalTiles",      mpi_comm);
    run_test(test_TrapezoidMatrix_allocateBatchArrays, "TrapezoidMatrix::allocateBatchArrays",   mpi_comm);

    if (mpi_rank == 0)
        printf("\nSub-matrices\n");
    run_test(test_TrapezoidMatrix_sub,           "TrapezoidMatrix::sub",                mpi_comm);
    run_test(test_TrapezoidMatrix_sub_trans,     "TrapezoidMatrix::sub(A^T)",           mpi_comm);

    if (mpi_rank == 0)
        printf("\nConversion to Trapezoid\n");
    run_test(test_Trapezoid_from_Matrix,       "TrapezoidMatrix( uplo, diag, Matrix )",    mpi_comm);
    run_test(test_Trapezoid_from_Hermitian,    "TrapezoidMatrix( diag, HermitianMatrix )", mpi_comm);
    run_test(test_Trapezoid_from_Symmetric,    "TrapezoidMatrix( diag, SymmetricMatrix )", mpi_comm);
    run_test(test_Trapezoid_from_Triangular,   "TrapezoidMatrix( TriangularMatrix )",      mpi_comm);
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
