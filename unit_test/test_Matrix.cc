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
#include "slate_util.hh"

#include "unit_test.hh"

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
// default constructor

//------------------------------------------------------------------------------
/// Tests Matrix(), mt, nt, op.
void test_Matrix()
{
    slate::Matrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
}

//------------------------------------------------------------------------------
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
/// Tests HermitianMatrix(), mt, nt, op, uplo.
void test_HermitianMatrix()
{
    slate::HermitianMatrix<double> A;

    test_assert(A.mt() == 0);
    test_assert(A.nt() == 0);
    test_assert(A.op() == blas::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::Lower);
}

//==============================================================================
// m-by-n, no-data constructor

//------------------------------------------------------------------------------
/// Tests Matrix(), mt, nt, op.
void test_Matrix_empty()
{
    slate::Matrix<double> A(m, n, nb, p, q, mpi_comm);

    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);
}

//------------------------------------------------------------------------------
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
/// Tests TriangularMatrix(), mt, nt, op, uplo, diag.
void test_TriangularMatrix_empty()
{
    slate::TriangularMatrix<double> L(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, nb, p, q, mpi_comm);

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    slate::TriangularMatrix<double> U(
        blas::Uplo::Upper, blas::Diag::Unit, n, nb, p, q, mpi_comm);

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);
}

//------------------------------------------------------------------------------
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
}

//==============================================================================
// fromLAPACK

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original LAPACK-style matrix that A is created from.
void verify_tile_lapack(
    slate::BaseMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ i*nb + j*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        test_assert(tile.uplo()   == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original LAPACK-style matrix that A is created from.
void verify_tile_lapack(
    slate::BaseTrapezoidMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ i*nb + j*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        if (i == j)
            test_assert(tile.uplo() == A.uplo());
        else
            test_assert(tile.uplo() == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
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
/// Test TrapezoidMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TrapezoidMatrix_fromLAPACK, but uses n-by-n matrix.
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
}

//------------------------------------------------------------------------------
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
}

//==============================================================================
// fromScaLAPACK

//------------------------------------------------------------------------------
/// Computes local dimensions for a block-cyclic distribution.
void get_cyclic_dimensions(
    int num_ranks, int rank, int n, int nb,
    int& ntiles, int& ntiles_local, int& n_local )
{
    assert(num_ranks > 0);

    // full tiles
    ntiles = int64_t(n / nb);
    ntiles_local = int64_t(ntiles / num_ranks);
    if (rank < ntiles % num_ranks) {
        ntiles_local += 1;
    }
    n_local = ntiles_local * nb;

    // partial tile
    if (n % nb > 0) {
        if (rank == ntiles % num_ranks) {
            ntiles_local += 1;
            n_local += n % nb;
        }
        ntiles += 1;
    }
}

//------------------------------------------------------------------------------
/// Computes local dimensions:
/// mtiles, mtiles_local, m_local,
/// ntiles, ntiles_local, n_local,
/// lda.
void get_2d_cyclic_dimensions(
    int m, int n,
    int& mtiles, int& mtiles_local, int& m_local,
    int& ntiles, int& ntiles_local, int& n_local,
    int& lda )
{
    int err;
    int mpi_rank;
    err = MPI_Comm_rank(mpi_comm, &mpi_rank);
    assert(err == 0);

    assert(p > 0 && q > 0);
    bool columnwise = true;
    int p_rank = (columnwise ? mpi_rank % p : mpi_rank / q);
    int q_rank = (columnwise ? mpi_rank / p : mpi_rank % q);

    get_cyclic_dimensions(p, p_rank, m, nb, mtiles, mtiles_local, m_local);
    get_cyclic_dimensions(q, q_rank, n, nb, ntiles, ntiles_local, n_local);
    lda = roundup(m_local, nb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original ScaLAPACK-style matrix that A is created from.
/// Similar to verify_tile_lapack, with different formula for Ad[ i, j ].
void verify_tile_scalapack(
    slate::BaseMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ int(i/p)*nb + int(j/q)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        test_assert(tile.uplo()   == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original ScaLAPACK-style matrix that A is created from.
void verify_tile_scalapack(
    slate::BaseTrapezoidMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ int(i/p)*nb + int(j/q)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        if (i == j)
            test_assert(tile.uplo() == A.uplo());
        else
            test_assert(tile.uplo() == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Test Matrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
void test_Matrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n,
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
/// Test TrapezoidMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_Matrix_fromScaLAPACK, but adds lower and upper.
void test_TrapezoidMatrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n,
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
/// Test TriangularMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TrapezoidMatrix_fromScaLAPACK, but uses n-by-n matrix.
void test_TriangularMatrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, // square
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
}

//------------------------------------------------------------------------------
/// Test SymmetricMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromScaLAPACK.
void test_SymmetricMatrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, // square
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
/// Test HermitianMatrix::fromScaLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromScaLAPACK.
void test_HermitianMatrix_fromScaLAPACK()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, // square
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
}

//==============================================================================
// fromDevices

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Aarray is the original array of device matrices that A is created from.
/// Similar to verify_tile_lapack, but with Aarray[ dev ][ i, j ]
/// and device().
void verify_tile_device(
    slate::BaseMatrix<double>& A, int i, int j, int nb,
    int m, int n, double** Aarray, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int dev = (j / p) % num_devices;
    test_assert(A.tileDevice(i, j) == dev);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j, dev);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   ==
            &Aarray[ dev ][ int(i/p)*nb + int(int(j/q)/num_devices)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        test_assert(tile.uplo()   == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == dev);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j, dev);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Aarray is the original array of device matrices that A is created from.
/// Similar to verify_tile_lapack, but with Aarray[ dev ][ i, j ].
void verify_tile_device(
    slate::BaseTrapezoidMatrix<double>& A, int i, int j, int nb,
    int m, int n, double** Aarray, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int dev = (j / p) % num_devices;
    test_assert(A.tileDevice(i, j) == dev);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j, dev);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   ==
            &Aarray[ dev ][ int(i/p)*nb + int(int(j/q)/num_devices)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        if (i == j)
            test_assert(tile.uplo() == A.uplo());
        else
            test_assert(tile.uplo() == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == dev);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j, dev);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Test Matrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
void test_Matrix_fromDevices()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n,
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
/// Test TrapezoidMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_Matrix_fromDevices, but adds lower and upper.
void test_TrapezoidMatrix_fromDevices()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        m, n,
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

//------------------------------------------------------------------------------
/// Test TriangularMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TrapezoidMatrix_fromDevices, but uses n-by-n matrix.
void test_TriangularMatrix_fromDevices()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, // square
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
}

//------------------------------------------------------------------------------
/// Test SymmetricMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromDevices.
void test_SymmetricMatrix_fromDevices()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, // square
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

//------------------------------------------------------------------------------
/// Test HermitianMatrix::fromDevices, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromDevices.
void test_HermitianMatrix_fromDevices()
{
    int mtiles, mtiles_local, m_local, lda;
    int ntiles, ntiles_local, n_local;
    get_2d_cyclic_dimensions(
        n, n, // square
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
}

//==============================================================================
// Methods

//------------------------------------------------------------------------------
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
                test_assert(AT(i,j).data() == &Ad[j*nb + i*nb*lda]);
                test_assert(AT(i,j).op() == slate::Op::Trans);
                test_assert(AT(i,j).mb() == AT.tileMb(i));
                test_assert(AT(i,j).nb() == AT.tileNb(j));
                test_assert(AT(i,j).mb() == ib);
                test_assert(AT(i,j).nb() == jb);
            }
        }
    }
}

//------------------------------------------------------------------------------
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
                test_assert(AT(i,j).data() == &Ad[j*nb + i*nb*lda]);
                test_assert(AT(i,j).op() == slate::Op::ConjTrans);
                test_assert(AT(i,j).mb() == AT.tileMb(i));
                test_assert(AT(i,j).nb() == AT.tileNb(j));
                test_assert(AT(i,j).mb() == ib);
                test_assert(AT(i,j).nb() == jb);
            }
        }
    }
}

//------------------------------------------------------------------------------
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
    test_assert(C(0,0).data() == Ad.data());

    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(k, nb));
    test_assert(B.op() == slate::Op::NoTrans);
    test_assert(B(0,0).data() == Bd.data());

    swap(B, C);

    // swap(B, C) in asserts
    test_assert(B.mt() == ceildiv(n, nb));
    test_assert(B.nt() == ceildiv(m, nb));
    test_assert(B.op() == slate::Op::Trans);
    test_assert(B(0,0).data() == Ad.data());

    test_assert(C.mt() == ceildiv(n, nb));
    test_assert(C.nt() == ceildiv(k, nb));
    test_assert(C.op() == slate::Op::NoTrans);
    test_assert(C(0,0).data() == Bd.data());
}

//------------------------------------------------------------------------------
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
void test_Matrix_sub()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            A(i, j).at(0, 0) = i + j / 10000.;
        }
    }

    auto Asub = A.sub( 0, 0, 0, 0 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::NoTrans );
    test_assert( Asub(0, 0).op() == slate::Op::NoTrans );
    test_assert( Asub(0, 0).at(0, 0) == 0.0 );

    // 1st column
    Asub = A.sub( 0, A.mt()-1, 0, 0 );
    test_assert( Asub.mt() == A.mt() );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::NoTrans );
    for (int i = 0; i < Asub.mt(); ++i) {
        test_assert( Asub(i, 0).at(0, 0) == i );
        test_assert( Asub(i, 0).op() == slate::Op::NoTrans );
    }

    // 1st row
    Asub = A.sub( 0, 0, 0, A.nt()-1 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == A.nt() );
    test_assert( Asub.op() == slate::Op::NoTrans );
    for (int j = 0; j < Asub.nt(); ++j) {
        test_assert( Asub(0, j).at(0, 0) == j / 10000. );
        test_assert( Asub(0, j).op() == slate::Op::NoTrans );
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
                test_assert( Asub(i, j).at(0, 0) == (i1 + i) + (j1 + j) / 10000. );
                test_assert( Asub(i, j).op() == slate::Op::NoTrans );
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
                    test_assert( Asub_b(i, j).at(0, 0) == (i1 + i1_b + i) + (j1 + j1_b + j) / 10000. );
                    test_assert( Asub_b(i, j).op() == slate::Op::NoTrans );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
void test_Matrix_sub_trans()
{
    int lda = roundup(m, nb);
    std::vector<double> Ad( lda*n );
    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad.data(), lda, nb, p, q, mpi_comm );

    // mark tiles so they're identifiable
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            A(i, j).at(0, 0) = i + j / 10000.;
        }
    }

    auto AT = transpose( A );

    auto Asub = AT.sub( 0, 0, 0, 0 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::Trans );
    test_assert( Asub(0, 0).op() == slate::Op::Trans );
    test_assert( Asub(0, 0).at(0, 0) == 0.0 );

    // 1st column
    Asub = AT.sub( 0, AT.mt()-1, 0, 0 );
    test_assert( Asub.mt() == AT.mt() );
    test_assert( Asub.nt() == 1 );
    test_assert( Asub.op() == slate::Op::Trans );
    for (int i = 0; i < Asub.mt(); ++i) {
        test_assert( Asub(i, 0).at(0, 0) == i / 10000. );
        test_assert( Asub(i, 0).op() == slate::Op::Trans );
    }

    // 1st row
    Asub = AT.sub( 0, 0, 0, AT.nt()-1 );
    test_assert( Asub.mt() == 1 );
    test_assert( Asub.nt() == AT.nt() );
    test_assert( Asub.op() == slate::Op::Trans );
    for (int j = 0; j < Asub.nt(); ++j) {
        test_assert( Asub(0, j).at(0, 0) == j );
        test_assert( Asub(0, j).op() == slate::Op::Trans );
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
                test_assert( Asub(i, j).at(0, 0) == (j1 + j) + (i1 + i) / 10000. );
                test_assert( Asub(i, j).op() == slate::Op::Trans );
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
                    test_assert( Asub_b(i, j).at(0, 0) == (j1 + i1_b + i) + (i1 + j1_b + j) / 10000. );
                    test_assert( Asub_b(i, j).op() == slate::Op::NoTrans );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
void test_TrapezoidMatrix_conversion()
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
            if (i == j)
                test_assert( L(i, j).uplo() == slate::Uplo::Lower );
            else
                test_assert( L(i, j).uplo() == slate::Uplo::General );
        }
    }

    // upper
    auto U = slate::TrapezoidMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, A );

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            if (i == j)
                test_assert( U(i, j).uplo() == slate::Uplo::Upper );
            else
                test_assert( U(i, j).uplo() == slate::Uplo::General );
        }
    }
}

//------------------------------------------------------------------------------
void test_TriangularMatrix_conversion()
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
            if (i == j)
                test_assert( L(i, j).uplo() == slate::Uplo::Lower );
            else
                test_assert( L(i, j).uplo() == slate::Uplo::General );
        }
    }

    // upper
    auto U = slate::TriangularMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, A );

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            if (i == j)
                test_assert( U(i, j).uplo() == slate::Uplo::Upper );
            else
                test_assert( U(i, j).uplo() == slate::Uplo::General );
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
//
// Trapezoid
//     Trapezoid(BaseTrapezoid)  // conversion
//     Trapezoid(uplo, Matrix)   // conversion
//     Trapezoid(orig, i1, i2, j1, j2)
//     Trapezoid(uplo, orig, i1, i2, j1, j2)
//     sub(i1, i2)          // diag
//     sub(i1, i2, j1, j2)  // off-diag
//     swap

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0)
        printf("\nDefault constructors\n");
    run_test(test_Matrix,            "Matrix()",           mpi_comm);
    run_test(test_TrapezoidMatrix,   "TrapezoidMatrix()",  mpi_comm);
    run_test(test_TriangularMatrix,  "TriangularMatrix()", mpi_comm);
    run_test(test_SymmetricMatrix,   "SymmetricMatrix()",  mpi_comm);
    run_test(test_HermitianMatrix,   "HermitianMatrix()",  mpi_comm);

    if (mpi_rank == 0)
        printf("\nm-by-n, no data constructors\n");
    run_test(test_Matrix_empty,           "Matrix(m, n, ...)",                mpi_comm);
    run_test(test_TrapezoidMatrix_empty,  "TrapezoidMatrix(uplo, m, n, ...)", mpi_comm);
    run_test(test_TriangularMatrix_empty, "TriangularMatrix(uplo, n, ...)",   mpi_comm);
    run_test(test_SymmetricMatrix_empty,  "SymmetricMatrix(uplo, n, ...)",    mpi_comm);
    run_test(test_HermitianMatrix_empty,  "HermitianMatrix(uplo, n, ...)",    mpi_comm);

    if (mpi_rank == 0)
        printf("\nLAPACK constructors\n");
    run_test(test_Matrix_fromLAPACK,           "Matrix::fromLAPACK",           mpi_comm);
    run_test(test_TrapezoidMatrix_fromLAPACK,  "TrapezoidMatrix::fromLAPACK",  mpi_comm);
    run_test(test_TriangularMatrix_fromLAPACK, "TriangularMatrix::fromLAPACK", mpi_comm);
    run_test(test_SymmetricMatrix_fromLAPACK,  "SymmetricMatrix::fromLAPACK",  mpi_comm);
    run_test(test_HermitianMatrix_fromLAPACK,  "HermitianMatrix::fromLAPACK",  mpi_comm);

    if (mpi_rank == 0)
        printf("\nScaLAPACK constructors\n");
    run_test(test_Matrix_fromScaLAPACK,           "Matrix::fromScaLAPACK",           mpi_comm);
    run_test(test_TrapezoidMatrix_fromScaLAPACK,  "TrapezoidMatrix::fromScaLAPACK",  mpi_comm);
    run_test(test_TriangularMatrix_fromScaLAPACK, "TriangularMatrix::fromScaLAPACK", mpi_comm);
    run_test(test_SymmetricMatrix_fromScaLAPACK,  "SymmetricMatrix::fromScaLAPACK",  mpi_comm);
    run_test(test_HermitianMatrix_fromScaLAPACK,  "HermitianMatrix::fromScaLAPACK",  mpi_comm);

    if (mpi_rank == 0)
        printf("\nDevice constructors\n");
    run_test(test_Matrix_fromDevices,           "Matrix::fromDevices",           mpi_comm);
    run_test(test_TrapezoidMatrix_fromDevices,  "TrapezoidMatrix::fromDevices",  mpi_comm);
    run_test(test_TriangularMatrix_fromDevices, "TriangularMatrix::fromDevices", mpi_comm);
    run_test(test_SymmetricMatrix_fromDevices,  "SymmetricMatrix::fromDevices",  mpi_comm);
    run_test(test_HermitianMatrix_fromDevices,  "HermitianMatrix::fromDevices",  mpi_comm);

    if (mpi_rank == 0)
        printf("\nMethods\n");
    run_test(test_Matrix_transpose,       "transpose",      mpi_comm);
    run_test(test_Matrix_conj_transpose,  "conj_transpose", mpi_comm);
    run_test(test_Matrix_swap,            "swap",           mpi_comm);
    run_test(test_Matrix_tileInsert_new,  "Matrix::tileInsert(i, j, dev) ", mpi_comm);
    run_test(test_Matrix_tileInsert_data, "Matrix::tileInsert(i, j, dev, data, lda)",  mpi_comm);
    run_test(test_Matrix_tileLife,        "Matrix::tileLife",  mpi_comm);
    run_test(test_Matrix_tileErase,       "Matrix::tileErase", mpi_comm);
    run_test(test_Matrix_sub,             "Matrix::sub",       mpi_comm);
    run_test(test_Matrix_sub_trans,       "Matrix::sub(A^T)",  mpi_comm);
    run_test(test_TrapezoidMatrix_conversion, "TrapezoidMatrix conversion", mpi_comm);
    run_test(test_TriangularMatrix_conversion, "TriangularMatrix conversion", mpi_comm);
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
