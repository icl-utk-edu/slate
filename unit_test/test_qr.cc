// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "internal/Tile_tpqrt.hh"
#include "internal/Tile_tpmqrt.hh"

#include "unit_test.hh"
#include "print_tile.hh"
#include "../test/print_matrix.hh"

using slate::HostNum;

namespace test {

//------------------------------------------------------------------------------
// globals
int      g_argc      = 0;
char**   g_argv      = nullptr;
int      verbose     = 0;
int      debug       = 0;
int      mpi_rank    = -1;
int      mpi_size    = 0;
int      num_devices = 0;
MPI_Comm mpi_comm;

//------------------------------------------------------------------------------
/// Test tpqrt and tpmqrt, QR of 2 tiles and multiply by Q.
///
template <typename scalar_t>
void test_tpqrt_work( int m, int n, int l, int cn, int ib )
{
    if (verbose) {
        printf( "%s( m=%3d, n=%3d, l=%3d, cn=%3d, ib=%3d )",  // no \n
                __func__, m, n, l, cn, ib );
    }

    using real_t = blas::real_type<scalar_t>;
    using lapack::Norm;
    using lapack::Uplo;
    using lapack::Diag;
    using lapack::MatrixType;

    // Unused entries will be set to nan.
    scalar_t nan_ = nan("");
    scalar_t zero = 0;

    int lda1 = n;
    int lda2 = m;
    std::vector< scalar_t > A1data( lda1*n, nan_ );
    std::vector< scalar_t > A2data( lda2*n, zero ); // not nan, for lange
    std::vector< scalar_t >  Tdata( ib*n,   nan_ );
    slate::Tile< scalar_t >
        A1( n,  n, A1data.data(), lda1, HostNum, slate::TileKind::UserOwned ),
        A2( m,  n, A2data.data(), lda2, HostNum, slate::TileKind::UserOwned ),
        T(  ib, n, Tdata.data(),  ib,   HostNum, slate::TileKind::UserOwned );

    // A1 is upper triangular, n-by-n.
    srand( 1234 );
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j && i < n; ++i)  // upper
            A1.at( i, j ) = rand() / real_t(RAND_MAX);

    // A2 is upper pentagonal, m-by-n.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j + (m - l) && i < m; ++i)  // upper pent.
            A2.at( i, j ) = rand() / real_t(RAND_MAX);

    // || A1 ||_1 + || A2 ||_1, bound on || A ||_1.
    auto Anorm = lapack::lantr( Norm::One, Uplo::Upper,
                                Diag::NonUnit, n, n, A1.data(), A1.stride() )
               + lapack::lange( Norm::One, m, n, A2.data(), A2.stride() );

    auto A1save = A1data;
    auto A2save = A2data;

    if (verbose > 1) {
        printf( "\n" );
        print( "A1", A1 );
        print( "A2", A2 );
        print( "T",   T );
    }

    slate::tpqrt( l, A1, A2, T );

    if (verbose > 1) {
        print( "post A1", A1 );
        print( "post A2", A2 );
        print( "post T",   T );
    }

    //---------------------
    // Error check || Q R - A ||_1 / || A ||_1 < tol.
    std::vector< scalar_t > R2data( lda2*n, zero );
    slate::Tile< scalar_t >
        R2( m, n, R2data.data(), lda2, HostNum, slate::TileKind::UserOwned );

    // Zero out R1 (A1) below diag.
    lapack::laset( MatrixType::Lower, n-1, n-1, zero, zero,
                   &A1.at(1, 0), A1.stride() );

    // Form Ahat = Q R, where Q = I - V T V^H, V = [ I  ],  R = [ A1 ]
    //                                             [ V2 ]       [ R2 ]
    slate::tpmqrt( slate::Side::Left, slate::Op::NoTrans, l, A2, T, A1, R2 );
    if (verbose > 1) {
        print( "Q R1", A1 );
        print( "Q R2", R2 );
    }

    // || Q R1 - A1 ||_1 + || Q R2 - A2 ||_1, bound on || Q R - A ||_1.
    for (size_t i = 0; i < A1data.size(); ++i)
        A1data[i] -= A1save[i];
    for (size_t i = 0; i < A2data.size(); ++i)
        R2data[i] -= A2save[i];
    auto err = lapack::lantr( Norm::One, Uplo::Upper,
                              Diag::NonUnit, n, n, A1.data(), A1.stride() )
             + lapack::lange( Norm::One, m, n, R2.data(), R2.stride() );
    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = 50*eps;
    if (verbose) {
        printf( " err %8.2e, Anorm %8.2e, err/Anorm %8.2e, %s\n",
                err, Anorm, err/Anorm, (err/Anorm < tol ? "pass" : "FAILED") );
    }
    test_assert( err/Anorm < tol );

    //---------------------
    // Left: C = op(Q) C
    // todo: check results
    // This only runs tpmlqt routine, so errors like wrong dimensions and
    // segfaults are caught, but doesn't yet check numerical error.
    // Could allocate A1, A2 as one LAPACK matrix, and C1, C2 as one LAPACK
    // matrix, then compare with LAPACK's unmqr output?
    int ldc1 = n;
    int ldc2 = m;
    std::vector< scalar_t > C1data( ldc1*cn );  // n-by-cn
    std::vector< scalar_t > C2data( ldc2*cn );  // m-by-cn
    slate::Tile< scalar_t >
        C1( n, cn, C1data.data(), ldc1, HostNum, slate::TileKind::UserOwned ),
        C2( m, cn, C2data.data(), ldc2, HostNum, slate::TileKind::UserOwned );

    for (int j = 0; j < cn; ++j)
        for (int i = 0; i < n; ++i)
            C1.at( i, j ) = rand() / real_t(RAND_MAX);

    for (int j = 0; j < cn; ++j)
        for (int i = 0; i < m; ++i)
            C2.at( i, j ) = rand() / real_t(RAND_MAX);

    if (verbose > 1) {
        print( "C1", C1 );
        print( "C2", C2 );
    }

    slate::tpmqrt( slate::Side::Left, slate::Op::NoTrans, l, A2, T, C1, C2 );
    if (verbose > 1) {
        print( "Q C1", C1 );
        print( "Q C2", C2 );
    }

    slate::tpmqrt( slate::Side::Left, slate::Op::ConjTrans, l, A2, T, C1, C2 );
    if (verbose > 1) {
        print( "Q^H C1", C1 );
        print( "Q^H C2", C2 );
    }

    if (! blas::is_complex< scalar_t >::value) {
        slate::tpmqrt( slate::Side::Left, slate::Op::Trans, l, A2, T, C1, C2 );
        if (verbose > 1) {
            print( "Q^T C1", C1 );
            print( "Q^T C2", C2 );
        }
    }
    else {
        test_assert_throw_std(
            slate::tpmqrt( slate::Side::Left, slate::Op::Trans, l, A2, T, C1, C2 ));
    }

    //---------------------
    // Right: C = C op(Q)
    int ldd = cn;
    std::vector< scalar_t > D1data( ldd*n );  // cn-by-n
    std::vector< scalar_t > D2data( ldd*m );  // cn-by-m
    slate::Tile< scalar_t >
        D1( cn, n, D1data.data(), ldd, HostNum, slate::TileKind::UserOwned ),
        D2( cn, m, D2data.data(), ldd, HostNum, slate::TileKind::UserOwned );

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < cn; ++i)
            D1.at( i, j ) = rand() / real_t(RAND_MAX);

    for (int j = 0; j < m; ++j)
        for (int i = 0; i < cn; ++i)
            D2.at( i, j ) = rand() / real_t(RAND_MAX);

    if (verbose > 1) {
        print( "D1", D1 );
        print( "D2", D2 );
    }

    slate::tpmqrt( slate::Side::Right, slate::Op::NoTrans, l, A2, T, D1, D2 );
    if (verbose > 1) {
        print( "D1 Q", D1 );
        print( "D2 Q", D2 );
    }

    slate::tpmqrt( slate::Side::Right, slate::Op::ConjTrans, l, A2, T, D1, D2 );
    if (verbose > 1) {
        print( "D1 Q^H", D1 );
        print( "D2 Q^H", D2 );
    }

    if (! blas::is_complex< scalar_t >::value) {
        slate::tpmqrt( slate::Side::Right, slate::Op::Trans, l, A2, T, D1, D2 );
        if (verbose > 1) {
            print( "D1 Q^T", D1 );
            print( "D2 Q^T", D2 );
        }
    }
    else {
        test_assert_throw_std(
            slate::tpmqrt( slate::Side::Right, slate::Op::Trans, l, A2, T, D1, D2 ));
    }
}

template <typename scalar_t>
void test_tpqrt_scalar()
{
    int cn = 13;
    // tplqt requires ib <= n.
    for (int ib = 4; ib <= 8; ++ib) {
        // ts triangle-square (rectangle) kernel cases, l == 0
        test_tpqrt_work< scalar_t >( 12, 12, 0, cn, ib );  // m == n (square)
        test_tpqrt_work< scalar_t >(  8, 12, 0, cn, ib );  // m <  n (wide rectangle)
        test_tpqrt_work< scalar_t >( 16, 12, 0, cn, ib );  // m >  n (tall rectangle)

        // tt triangle-triangle (trapezoid) kernel cases, l == min(m, n)
        test_tpqrt_work< scalar_t >( 12, 12, 12, cn, ib );  // m == n (triangle)
        test_tpqrt_work< scalar_t >(  8, 12,  8, cn, ib );  // m <  n (wide trapezoid)
        test_tpqrt_work< scalar_t >( 16, 12, 12, cn, ib );  // m >  n (tall trapezoid)

        // tp triangle-pentagonal cases, l < min(m, n)
        test_tpqrt_work< scalar_t >( 12, 12, 6, cn, ib );  // m == n
        test_tpqrt_work< scalar_t >(  8, 12, 6, cn, ib );  // m <  n
        test_tpqrt_work< scalar_t >( 16, 12, 6, cn, ib );  // m >  n
    }
}

void test_tpqrt()
{
    test_tpqrt_scalar< float >();
    test_tpqrt_scalar< double >();
    test_tpqrt_scalar< std::complex<float> >();
    test_tpqrt_scalar< std::complex<double> >();
}



//------------------------------------------------------------------------------
/// Test ttqrt and ttmqr.
/// todo: test ttmqr for other sides and ops.
///
template <typename scalar_t>
void test_ttqrt_work( int m, int n, int nb, int ib, int p, int q )
{
    if (verbose && mpi_rank == 0) {
        printf( "\nrank %2d, %s( m=%3d, n=%3d, nb=%3d, ib=%3d, p=%d, q=%d\n",
                mpi_rank, __func__, m, n, nb, ib, p, q );
    }

    using real_t = blas::real_type<scalar_t>;

    scalar_t one  = 1;
    scalar_t zero = 0;

    if (debug) printf( "rank %2d, init\n", mpi_rank );
    slate::Matrix<scalar_t> A( m, n, nb, p, q, mpi_comm );
    A.insertLocalTiles();

    if (debug) printf( "rank %2d, rand\n", mpi_rank );
    // Initialize random, different on each rank.
    srand( time(nullptr) * (mpi_rank + 1) );
    for (int j = 0; j < A.nt(); ++j) {
        for (int i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                auto t = A(i, j);
                int t_mb = t.mb();
                int t_nb = t.nb();
                for (int jj = 0; jj < t_nb; ++jj) {
                    for (int ii = 0; ii < t_mb; ++ii) {
                        t.at(ii, jj) = rand() / (0.5 * RAND_MAX) - 1;
                    }
                }
            }
        }
    }

    if (debug) printf( "rank %2d, copy\n", mpi_rank );
    // Copy A.
    auto A0 = A.emptyLike();
    A0.insertLocalTiles();
    copy( A, A0 );

    if (debug) printf( "rank %2d, copy TT\n", mpi_rank );
    // Copy just the triangle-triangle pieces.
    auto TT = A.emptyLike();
    TT.insertLocalTiles();
    set( zero, TT );
    for (int i = 0; i < blas::min( A.mt(), p ); ++i) {
        if (A.tileIsLocal( i, 0 )) {
            auto Ai0  =  A( i, 0 );
            auto TTi0 = TT( i, 0 );
            Ai0 .uplo( slate::Uplo::Upper );
            TTi0.uplo( slate::Uplo::Upper );
            slate::tile::tzcopy( Ai0, TTi0 );
        }
    }

    if (debug) printf( "rank %2d, T\n", mpi_rank );
    // Matrix of T matrices for block Householder reflectors.
    auto T = A.emptyLike( ib, 0 );  // ib-by-nb tiles

    if (verbose > 1) {
        slate::print( "A_init", A );
        slate::print( "TT_init", TT );
        slate::print( "T_init", T );
    }

    //----------
    // Triangle-triangle factor of A.
    if (debug) printf( "rank %2d, ttqrt\n", mpi_rank );
    auto A_panel = A.sub( 0, A.mt()-1, 0, 0 );
    auto T_panel = T.sub( 0, A.mt()-1, 0, 0 );
    slate::internal::ttqrt(
        std::move( A_panel ),
        std::move( T_panel ));

    if (verbose > 1) {
        slate::print( "A", A );
        slate::print( "T", T );
    }

    // Copy top tile as R.
    if (debug) printf( "rank %2d, copy R\n", mpi_rank );
    auto R = A.emptyLike();
    R.insertLocalTiles();
    set( zero, R );
    if (A.tileIsLocal( 0, 0 )) {
        auto R00 = R( 0, 0 );
        auto A00 = A( 0, 0 );
        R00.uplo( slate::Uplo::Upper );
        A00.uplo( slate::Uplo::Upper );
        slate::tile::tzcopy( A00, R00 );
    }
    if (verbose > 1) {
        slate::print( "R", R );
    }

    //----------
    // Multiply R = QR with Q from triangle-triangle factorization.
    // First broadcast V and T along rows, for ranks after 0.
    if (debug) printf( "rank %2d, ttmqr bcast\n", mpi_rank );
    for (int i = 1; i < blas::min( A.mt(), p ); ++i) {
        A_panel.tileBcast( i, 0, A.sub( i, i, 0, A.nt()-1 ), slate::Layout::ColMajor );
        T_panel.tileBcast( i, 0, T.sub( i, i, 0, T.nt()-1 ), slate::Layout::ColMajor );
    }
    if (debug) printf( "rank %2d, ttmqr\n", mpi_rank );
    slate::internal::ttmqr(
        slate::Side::Left, slate::Op::NoTrans,
        std::move( A_panel ),
        std::move( T ),
        std::move( R ));
    if (verbose > 1) {
        slate::print( "QR", R );
    }

    // Error check || QR - A || / || A ||, where A is the original
    // triangle-triangle data copied in TT.
    if (debug) printf( "rank %2d, error norms\n", mpi_rank );
    real_t Anorm = slate::norm( slate::Norm::Fro, TT );
    add( -one, TT, one, R );
    if (verbose > 1) {
        slate::print( "QR - A", R );
    }
    real_t QR_Anorm = slate::norm( slate::Norm::Fro, R );
    real_t error = QR_Anorm / Anorm;
    real_t eps = std::numeric_limits<real_t>::epsilon();
    real_t tol = 50 * eps;
    if (verbose > 0 && mpi_rank == 0) {
        printf( "rank %2d, QR_Anorm %.2e / Anorm %.2e = error %.2e\n",
                A.mpiRank(), QR_Anorm, Anorm, error );
    }
    test_assert( error < tol );

    // A0 - A should be 0 outside the triangle-triangle regions;
    // inside the TT regions, the values are meaningless.
    if (verbose > 1) {
        add( -one, A, one, A0 );
        slate::print( "Ainit - Ahat", A0 );
    }
}

//------------------------------------------------------------------------------
void test_ttqrt()
{
    if (false) {
        // Test a specific size for debugging.
        int p  = mpi_size;
        int q  = 1;
        int ib = 4;   // 3;
        int nb = 8;   // 6;
        int m  = 13;  // 2 * q * nb;
        int n  = 4;   // nb;
        test_ttqrt_work<float>( m, n, nb, ib, p, q );
    }
    else {
        // Try all valid combinations of p*q.
        for (int q = 1; q <= mpi_size; ++q) {
            int p = mpi_size / q;
            if (p*q != mpi_size)
                continue;

            for (int ib = 4; ib <= 8; ib += 2) {
                for (int nb = ib; nb <= 2*ib; nb += 4) {

                    // Try single block-col matrices.
                    for (int m = nb; m <= 2*p*nb; ++m) {
                        for (int n = ib; n <= nb; ++n) { // Requires n >= ib.
                            test_ttqrt_work<float>( m, n, nb, ib, p, q );
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Test unmqr.
///
template <typename scalar_t>
void test_unmqr_work( slate::Side side, slate::Op op, int m, int n, int k )
{
    using real_t = blas::real_type<scalar_t>;

    if (verbose && mpi_rank == 0) {
        printf( "%s( side=%c, op=%c, m=%3d, n=%3d, k=%3d ) ",
                __func__, char(side), char(op), m, n, k );
    }

    assert( m >= k ); // assuming left

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };

    // For left,  C = op(Q) C = (I - V op(T) V^H) C.
    // For right, C = C op(Q) = C (I - V^H op(T) V).
    // V is m-by-k (on left) or n-by-k (on right).
    // Random unitary -- each col is made into a Householder vector,
    // without updating the trailing matrix.
    int64_t Vm = (side == slate::Side::Left ? m : n);
    int64_t ldv = Vm;
    std::vector< scalar_t > Vdata( ldv * k );
    std::vector< scalar_t > tau( k );
    lapack::larnv( idist, iseed, Vdata.size(), Vdata.data() );
    for (int64_t j = 0; j < k; ++j) {
        lapack::larfg( Vm - j, &Vdata[ j     + j*ldv ],
                               &Vdata[ (j+1) + j*ldv ], 1, &tau[j] );
    }

    // T is k-by-k. Generate from V and tau.
    int64_t ldt = k;
    std::vector< scalar_t > Tdata( ldt * k );
    lapack::larft( lapack::Direction::Forward, lapack::StoreV::Columnwise, Vm, k,
                   Vdata.data(), ldv, tau.data(), Tdata.data(), ldt );

    // C is m-by-n.
    int64_t ldc = m;
    std::vector< scalar_t > Cdata( ldc * n );
    lapack::larnv( idist, iseed, Cdata.size(), Cdata.data() );
    std::vector< scalar_t > Cref_data = Cdata;

    // Use nb = k.
    auto V = slate::Matrix<scalar_t>::fromLAPACK( Vm,  k, Vdata.data(), ldv, k, 1, 1, mpi_comm );
    auto T = slate::Matrix<scalar_t>::fromLAPACK(  k,  k, Tdata.data(), ldt, k, 1, 1, mpi_comm );
    auto C = slate::Matrix<scalar_t>::fromLAPACK(  m,  n, Cdata.data(), ldc, k, 1, 1, mpi_comm );
    auto Cref = slate::Matrix<scalar_t>::fromLAPACK( m, n, Cref_data.data(), ldc, k, 1, 1, mpi_comm );

    if (verbose >= 2) {
        printf( "\n" );
        print( "tau", 1, k, tau.data(), 1, 1, 1, mpi_comm );
        slate::print( "V", V );
        slate::print( "T", T );
    }

    //----------
    // Perform operations with LAPACK.
    if (verbose >= 2) {
        slate::print( "Cref", Cref );
    }
    lapack::unmqr( side, op, m, n, k,
                   Vdata.data(), ldv, tau.data(),
                   Cref_data.data(), ldc );
    if (verbose >= 2) {
        slate::print( "Cref_after", Cref );
    }

    //----------
    // Perform operations with SLATE.
    if (verbose >= 2) {
        slate::print( "C",  C );
    }
    auto W = C.emptyLike();
    slate::internal::unmqr( side, op,
                            std::move(V), std::move(T),
                            std::move(C), std::move(W) );
    if (verbose >= 2) {
        slate::print( "C_after",  C );
    }

    //----------
    // Relative error check.
    real_t Cnorm = slate::norm( slate::Norm::One, Cref );
    scalar_t one = 1;
    slate::add( -one, Cref, one, C );
    real_t error = slate::norm( slate::Norm::One, C ) / Cnorm;
    if (verbose >= 2) {
        slate::print( "C - Cref",  C );
    }
    if (verbose > 0 && mpi_rank == 0) {
        printf( "error %.2e, Cnorm %.2e\n", error, Cnorm );
    }
    real_t eps = std::numeric_limits<real_t>::epsilon();
    real_t tol = 50 * eps;
    test_assert( error < tol );
}

void test_unmqr()
{
    if (false) {
        // Test a specific size for debugging.
        test_unmqr_work<float>( slate::Side::Left, slate::Op::NoTrans, 8, 8, 4 );
    }
    else {
        // Try various sizes.
        for (int m = 8; m <= 16; ++m) {
            for (int n = 8; n <= 16; ++n) {
                for (int k = 4; k < 16; k += 4) {
                    if (m >= k) {
                        test_unmqr_work<float>( slate::Side::Left, slate::Op::NoTrans,   m, n, k );
                        test_unmqr_work<float>( slate::Side::Left, slate::Op::ConjTrans, m, n, k );

                        test_unmqr_work< double >( slate::Side::Left, slate::Op::NoTrans,   m, n, k );
                        test_unmqr_work< double >( slate::Side::Left, slate::Op::ConjTrans, m, n, k );

                        test_unmqr_work< std::complex<float> >( slate::Side::Left, slate::Op::NoTrans,   m, n, k );
                        test_unmqr_work< std::complex<float> >( slate::Side::Left, slate::Op::ConjTrans, m, n, k );

                        test_unmqr_work< std::complex<double> >( slate::Side::Left, slate::Op::NoTrans,   m, n, k );
                        test_unmqr_work< std::complex<double> >( slate::Side::Left, slate::Op::ConjTrans, m, n, k );
                    }
                    // todo: side = Right not implemented.
                    //if (n >= k) {
                    //    test_unmqr_work<float>( slate::Side::Right, slate::Op::NoTrans,   m, n, k );
                    //    test_unmqr_work<float>( slate::Side::Right, slate::Op::ConjTrans, m, n, k );
                    //}
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Similar routine list to testsweeper. No params yet.
typedef void (*test_func_ptr)();

typedef struct {
    const char* name;
    test_func_ptr func;
    int section;
} routines_t;

//------------------------------------------------------------------------------
enum Section {
    newline = 0,  // zero flag forces newline
    qr,
};

//------------------------------------------------------------------------------
std::vector< routines_t > routines = {
    { "tpqrt",  test_tpqrt,  Section::qr },
    { "ttqrt",  test_ttqrt,  Section::qr },
    { "unmqr",  test_unmqr,  Section::qr },
    { "",       nullptr,     Section::newline },
};

//------------------------------------------------------------------------------
// todo: usage as in testsweeper.
void usage()
{
    printf("Usage: %s [routines]\n", g_argv[0]);
    int col = 0;
    int last_section = routines[0].section;
    for (size_t j = 0; j < routines.size(); ++j) {
        if (routines[j].section != Section::newline &&
            routines[j].section != last_section)
        {
            last_section = routines[j].section;
            col = 0;
            printf("\n");
        }
        if (routines[j].name)
            printf("    %-20s", routines[j].name);
        col += 1;
        if (col == 3 || routines[j].section == Section::newline) {
            col = 0;
            printf("\n");
        }
    }
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    // Parse global options -h, -v, -d.
    int i;
    for (i = 1; i < g_argc; ++i) {
        std::string arg = g_argv[i];
        if (arg == "-h" || arg == "--help") {
            usage();
            return;
        }
        else if (arg == "-v" || arg == "--verbose") {
            ++verbose;
            continue;
        }
        else if (arg == "-d" || arg == "--debug") {
            ++debug;
            continue;
        }
        else {
            break;
        }
    }

    // Remaining options are tests to run.
    if (i == g_argc) {
        // Run all tests.
        for (size_t j = 0; j < routines.size(); ++j)
            if (routines[j].func != nullptr)
                run_test(routines[j].func, routines[j].name, MPI_COMM_WORLD);
    }
    else {
        // Run tests mentioned on command line.
        for (/* continued */; i < g_argc; ++i) {
            std::string arg = g_argv[i];
            bool found = false;
            for (size_t j = 0; j < routines.size(); ++j) {
                if (arg == routines[j].name) {
                    run_test(routines[j].func, routines[j].name, MPI_COMM_WORLD);
                    found = true;
                }
            }
            if (! found) {
                usage();
                printf("Unknown routine: %s\n", g_argv[i]);
            }
        }
    }
}

}  // namespace test

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace test;  // for globals mpi_rank, etc.

    g_argc = argc;
    g_argv = argv;
    MPI_Init(&argc, &argv);
    mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    num_devices = blas::get_device_count();

    int err = unit_test_main(mpi_comm);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
