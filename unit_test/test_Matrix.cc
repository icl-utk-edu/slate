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
int m, n, nb, p, q;
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
    double* Ad = new double[ lda*n ];

    auto A = slate::Matrix<double>::fromLAPACK(
        m, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == ceildiv(m, nb));
    test_assert(A.nt() == ceildiv(n, nb));
    test_assert(A.op() == blas::Op::NoTrans);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_lapack(A, i, j, nb, m, n, Ad, lda);
        }
    }

    delete[] Ad;
}

//------------------------------------------------------------------------------
/// Test TrapezoidMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_Matrix_fromLAPACK, but adds lower and upper.
void test_TrapezoidMatrix_fromLAPACK()
{
    int lda = roundup(m, nb);
    double* Ad = new double[ lda*n ];

    //----------
    // lower
    auto L = slate::TrapezoidMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, m, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_lapack(L, i, j, nb, m, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::TrapezoidMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, m, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_lapack(U, i, j, nb, m, n, Ad, lda);
        }
    }

    delete[] Ad;
}

//------------------------------------------------------------------------------
/// Test TrapezoidMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TrapezoidMatrix_fromLAPACK, but uses n-by-n matrix.
void test_TriangularMatrix_fromLAPACK()
{
    int lda = roundup(n, nb);
    double* Ad = new double[ lda*n ];

    //----------
    // lower
    auto L = slate::TriangularMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_lapack(L, i, j, nb, n, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::TriangularMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_lapack(U, i, j, nb, n, n, Ad, lda);
        }
    }

    delete[] Ad;
}

//------------------------------------------------------------------------------
/// Test SymmetricMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromLAPACK.
void test_SymmetricMatrix_fromLAPACK()
{
    int lda = roundup(n, nb);
    double* Ad = new double[ lda*n ];

    //----------
    // lower
    auto L = slate::SymmetricMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_lapack(L, i, j, nb, n, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::SymmetricMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_lapack(U, i, j, nb, n, n, Ad, lda);
        }
    }

    delete[] Ad;
}

//------------------------------------------------------------------------------
/// Test HermitianMatrix::fromLAPACK, A(i, j), tileIsLocal, tileMb, tileNb.
/// Similar to test_TriangularMatrix_fromLAPACK.
void test_HermitianMatrix_fromLAPACK()
{
    int lda = roundup(n, nb);
    double* Ad = new double[ lda*n ];

    //----------
    // lower
    auto L = slate::HermitianMatrix<double>::fromLAPACK(
        blas::Uplo::Lower, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_lapack(L, i, j, nb, n, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::HermitianMatrix<double>::fromLAPACK(
        blas::Uplo::Upper, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_lapack(U, i, j, nb, n, n, Ad, lda);
        }
    }

    delete[] Ad;
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

    double* Ad = new double[ lda*n_local ];

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(A.mt() == mtiles);
    test_assert(A.nt() == ntiles);
    test_assert(A.op() == blas::Op::NoTrans);

    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            verify_tile_scalapack(A, i, j, nb, m, n, Ad, lda);
        }
    }

    delete[] Ad;
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

    double* Ad = new double[ lda*n_local ];

    //----------
    // lower
    auto L = slate::TrapezoidMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, m, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(m, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, m, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::TrapezoidMatrix<double>::fromScaLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, m, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(m, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_scalapack(U, i, j, nb, m, n, Ad, lda);
        }
    }

    delete[] Ad;
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

    double* Ad = new double[ lda*n_local ];

    //----------
    // lower
    auto L = slate::TriangularMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, blas::Diag::NonUnit, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);
    test_assert(L.diag() == blas::Diag::NonUnit);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, n, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::TriangularMatrix<double>::fromScaLAPACK(
        blas::Uplo::Upper, blas::Diag::Unit, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);
    test_assert(U.diag() == blas::Diag::Unit);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_scalapack(U, i, j, nb, n, n, Ad, lda);
        }
    }

    delete[] Ad;
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

    double* Ad = new double[ lda*n_local ];

    //----------
    // lower
    auto L = slate::SymmetricMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, n, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::SymmetricMatrix<double>::fromScaLAPACK(
        blas::Uplo::Upper, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_scalapack(U, i, j, nb, n, n, Ad, lda);
        }
    }

    delete[] Ad;
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

    double* Ad = new double[ lda*n_local ];

    //----------
    // lower
    auto L = slate::HermitianMatrix<double>::fromScaLAPACK(
        blas::Uplo::Lower, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(L.mt() == ceildiv(n, nb));
    test_assert(L.nt() == ceildiv(n, nb));
    test_assert(L.op() == blas::Op::NoTrans);
    test_assert(L.uplo() == blas::Uplo::Lower);

    for (int j = 0; j < L.nt(); ++j) {
        for (int i = j; i < L.mt(); ++i) {  // lower
            verify_tile_scalapack(L, i, j, nb, n, n, Ad, lda);
        }
    }

    //----------
    // upper
    auto U = slate::HermitianMatrix<double>::fromScaLAPACK(
        blas::Uplo::Upper, n, Ad, lda, nb, p, q, mpi_comm );

    test_assert(U.mt() == ceildiv(n, nb));
    test_assert(U.nt() == ceildiv(n, nb));
    test_assert(U.op() == blas::Op::NoTrans);
    test_assert(U.uplo() == blas::Uplo::Upper);

    for (int j = 0; j < U.nt(); ++j) {
        for (int i = 0; i <= j && i < U.mt(); ++i) {  // upper
            verify_tile_scalapack(U, i, j, nb, n, n, Ad, lda);
        }
    }

    delete[] Ad;
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
//     swap
//     transpose
//     conj_transpose
//     num_devices
//     m, n
//     tileDevice
//     tileInsert (2)
//     tileLife (2)
//     tileTick
//     tileErase
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
//     sub(i1, i2, j1, j2)
//     swap
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
    nb = 16;
    p  = std::min(2, mpi_size);
    q  = mpi_size / p;
    if (argc > 1) { m  = atoi(argv[1]); }
    if (argc > 2) { n  = atoi(argv[2]); }
    if (argc > 3) { nb = atoi(argv[3]); }
    if (argc > 4) { p  = atoi(argv[4]); }
    if (argc > 5) { q  = atoi(argv[5]); }
    if (mpi_rank == 0) {
        printf("m %d, n %d, nb %d, p %d, q %d, num_devices %d\n",
               m, n, nb, p, q, num_devices);
    }

    int err = unit_test_main(mpi_comm);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
