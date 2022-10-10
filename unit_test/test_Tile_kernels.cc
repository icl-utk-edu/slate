// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "internal/Tile_lapack.hh"
#include "slate/internal/device.hh"

#include "unit_test.hh"
#include "print_tile.hh"

using slate::HostNum;

namespace test {

//------------------------------------------------------------------------------
// globals
int      g_argc      = 0;
char**   g_argv      = nullptr;
int      verbose     = 0;
int      mpi_rank    = -1;
int      mpi_size    = 0;
int      num_devices = 0;
MPI_Comm mpi_comm;

//------------------------------------------------------------------------------
// arrays of options to loop over in tests
blas::Uplo uplos[] = {
    blas::Uplo::Lower,
    blas::Uplo::Upper
};

blas::Op ops[] = {
    blas::Op::NoTrans,
    blas::Op::Trans,
    blas::Op::ConjTrans
};

blas::Side sides[] = {
    blas::Side::Left,
    blas::Side::Right
};

blas::Diag diags[] = {
    blas::Diag::NonUnit,
    blas::Diag::Unit
};

lapack::Norm norms[] = {
    lapack::Norm::Max,
    lapack::Norm::One,
    lapack::Norm::Inf,
    lapack::Norm::Fro
};

//------------------------------------------------------------------------------
// conjugates the matrix A, in-place.
template <typename scalar_t>
void conjugate(int m, int n, scalar_t* A, int lda)
{
    using blas::conj;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[ i + j*lda ] = conj(A[ i + j*lda ]);
}

//------------------------------------------------------------------------------
// conjugates the tile A, in-place.
template <typename scalar_t>
void conjugate(slate::Tile< scalar_t >& A)
{
    using blas::conj;
    for (int j = 0; j < A.nb(); ++j)
        for (int i = 0; i < A.mb(); ++i)
            A.at(i,j) = conj( A.at(i,j) );
}

//------------------------------------------------------------------------------
// copy op(A) to opAref
template <typename scalar_t>
void copy( slate::Tile< scalar_t > const& A, scalar_t* opAref, int lda )
{
    using blas::conj;
    for (int j = 0; j < A.nb(); ++j) {
        for (int i = 0; i < A.mb(); ++i) {
            opAref[ i + j*lda ] = A(i,j);
        }
    }
}

//------------------------------------------------------------------------------
// check op(A) == B, within absolute or relative tolerance.
// Assert aborts on failure.
template <typename scalar_t>
void test_assert_equal(
    slate::Tile< scalar_t > const& A,
    scalar_t const* B, int ldb,
    blas::real_type<scalar_t> abs_tol=0,
    blas::real_type<scalar_t> rel_tol=0 )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::conj;

    // whether op(A) is general, lower, or upper
    bool general = (A.uplo() == blas::Uplo::General);
    bool lower = (A.uplo() == blas::Uplo::Lower);
    bool upper = (A.uplo() == blas::Uplo::Upper);
    assert( general || lower || upper );

    for (int j = 0; j < A.nb(); ++j) {
        for (int i = 0; i < A.mb(); ++i) {
            if (general || (lower && i >= j) || (upper && i <= j)) {
                real_t abs_error;
                abs_error = std::abs( A(i,j) - B[ i + j*ldb ] );
                real_t rel_error = abs_error / std::abs( A(i,j) );

                // print elements if assert will fail
                if (! (abs_error <= abs_tol || rel_error <= rel_tol)) {
                    printf( "A(%3d, %3d) %8.4f + %8.4fi\n"
                            "B           %8.4f + %8.4fi, abs_error %8.2e, rel_error %8.2e\n",
                            i, j, real( A(i,j) ), imag( A(i,j) ),
                                  real( B[ i + j*ldb ] ), imag( B[ i + j*ldb ] ),
                            abs_error, rel_error );
                }

                test_assert( abs_error <= abs_tol || rel_error <= rel_tol );
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_gemm()
{
    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    real_t eps = std::numeric_limits< real_t >::epsilon();
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int m = 50;
    int n = 40;
    int k = 30;

    scalar_t alpha, beta;
    lapack::larnv( 1, iseed, 1, &alpha );
    lapack::larnv( 1, iseed, 1, &beta  );
    if (verbose) {
        printf( "alpha = %.4f + %.4fi;\n"
                "beta  = %.4f + %.4fi;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
    }

    // test all combinations of op(C), op(B), op(A)
    for (int ic = 0; ic < 3; ++ic) {
    for (int ib = 0; ib < 3; ++ib) {
    for (int ia = 0; ia < 3; ++ia) {
        // setup C such that op(C) is m-by-n
        int Cm = (ic == 0 ? m : n);
        int Cn = (ic == 0 ? n : m);
        int ldc = Cm + 1;
        std::vector< scalar_t > Cdata( ldc*Cn );
        lapack::larnv( 1, iseed, Cdata.size(), Cdata.data() );
        slate::Tile< scalar_t > C( Cm, Cn, Cdata.data(), ldc, HostNum,
                                   slate::TileKind::UserOwned );
        C.op( ops[ic] );
        assert( C.mb() == m );
        assert( C.nb() == n );

        // opCref = op(C) is m-by-n
        int ldopc = m + 1;
        std::vector< scalar_t > opCref( ldopc*n );
        copy( C, opCref.data(), ldopc );

        // setup B such that op(B) is k-by-n
        int Bm = (ib == 0 ? k : n);
        int Bn = (ib == 0 ? n : k);
        int ldb = Bm + 1;
        std::vector< scalar_t > Bdata( ldb*Bn );
        lapack::larnv( 1, iseed, Bdata.size(), Bdata.data() );
        slate::Tile< scalar_t > B( Bm, Bn, Bdata.data(), ldb, HostNum,
                                   slate::TileKind::UserOwned );
        B.op( ops[ib] );
        assert( B.mb() == k );
        assert( B.nb() == n );

        // setup A such that op(A) is m-by-k
        int Am = (ia == 0 ? m : k);
        int An = (ia == 0 ? k : m);
        int lda = Am + 1;
        std::vector< scalar_t > Adata( lda*An );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( Am, An, Adata.data(), lda, HostNum,
                                   slate::TileKind::UserOwned );
        A.op( ops[ia] );
        assert( A.mb() == m );
        assert( A.nb() == k );

        if (verbose) {
            printf( "gemm( opA=%c, opB=%c, opC=%c )\n",
                    char(A.op()), char(B.op()), char(C.op()) );
        }

        //if (verbose) {
        //    print( "A", A );
        //    print( "B", B );
        //    print( "C", C );
        //}

        // run test
        try {
            slate::tile::gemm( alpha, A, B, beta, C );

            // It should throw error if and only if
            // C is complex and
            // ((C is transposed and either A or B is conj-transposed) or
            //  (C is conj-transposed and either A or B is tranpsosed)).
            assert( ! (slate::is_complex< scalar_t >::value &&
                       ((ic == 1 && (ia == 2 || ib == 2)) ||
                        (ic == 2 && (ia == 1 || ib == 1)))) );
        }
        catch (std::exception& e) {
            //printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    ((ic == 1 && (ia == 2 || ib == 2)) ||
                     (ic == 2 && (ia == 1 || ib == 1))) );
            continue;
        }

        //if (verbose) {
        //    print( "Chat", C );
        //    print( "Aref", Am, An, Adata.data(), lda );
        //    print( "Bref", Bm, Bn, Bdata.data(), ldb );
        //    print( "Cref", m, n, opCref.data(), ldopc );
        //}

        // reference solution
        blas::gemm( blas::Layout::ColMajor, A.op(), B.op(), m, n, k,
                    alpha, Adata.data(), lda,
                           Bdata.data(), ldb,
                    beta, opCref.data(), ldopc );

        //if (verbose) {
        //    print( "Chat_ref", m, n, opCref.data(), ldopc );
        //}

        test_assert_equal( C, opCref.data(), ldopc, 3*sqrt(k)*eps, 3*sqrt(k)*eps );
    }}}
}

void test_gemm()
{
    test_gemm< float  >();
    test_gemm< double >();
    test_gemm< std::complex<float>  >();
    test_gemm< std::complex<double> >();
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_syrk()
{
    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    real_t eps = std::numeric_limits< real_t >::epsilon();
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int n = 50;
    int k = 30;

    scalar_t alpha, beta;
    lapack::larnv( 1, iseed, 1, &alpha );
    lapack::larnv( 1, iseed, 1, &beta  );
    if (verbose) {
        printf( "alpha = %.4f + %.4fi;\n"
                "beta  = %.4f + %.4fi;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
    }

    // test all combinations of op(C), op(A), uplo
    for (int ic = 0; ic < 3; ++ic) {
    for (int ia = 0; ia < 3; ++ia) {
    for (int iu = 0; iu < 2; ++iu) {
        blas::Uplo uplo = uplos[iu];

        // setup C such that op(C) is n-by-n
        int ldc = n + 1;
        std::vector< scalar_t > Cdata( ldc*n );
        lapack::larnv( 1, iseed, Cdata.size(), Cdata.data() );
        slate::Tile< scalar_t > C( n, n, Cdata.data(), ldc, HostNum,
                                   slate::TileKind::UserOwned );
        C.uplo( uplo );
        C.op( ops[ic] );
        assert( C.mb() == n );
        assert( C.nb() == n );

        // set unused data to nan
        scalar_t nan_ = nan("");
        if (uplo == blas::Uplo::Lower) {
            lapack::laset( lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                           &Cdata[ 0 + 1*ldc ], ldc );
        }
        else {
            lapack::laset( lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                           &Cdata[ 1 + 0*ldc ], ldc );
        }

        // opCref = op(C) is n-by-n
        std::vector< scalar_t > opCref( ldc*n );
        copy( C, opCref.data(), ldc );

        // setup A such that op(A) is n-by-k
        int Am = (ia == 0 ? n : k);
        int An = (ia == 0 ? k : n);
        int lda = Am + 1;
        std::vector< scalar_t > Adata( lda*An );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( Am, An, Adata.data(), lda, HostNum,
                                   slate::TileKind::UserOwned );
        A.op( ops[ia] );
        assert( A.mb() == n );
        assert( A.nb() == k );

        if (verbose) {
            printf( "syrk( uplo=%c, opA=%c, opC=%c )\n",
                    char(C.uplo()), char(A.op()), char(C.op()) );
        }

        //if (verbose) {
        //    print( "A", A );
        //    print( "C", C );
        //}

        // run test
        try {
            if (C.op() == blas::Op::ConjTrans)  // TODO
                conjugate( C );
            slate::tile::syrk( alpha, A, beta, C );
            if (C.op() == blas::Op::ConjTrans)  // TODO
                conjugate( C );

            // It should throw error if and only if
            // C is complex and
            // C is conj-transposed or A is conj-transposed.
            assert( ! (slate::is_complex< scalar_t >::value &&
                       (ic == 2 || ia == 2)) );
        }
        catch (std::exception& e) {
            //printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    (ic == 2 || ia == 2) );
            continue;
        }

        //if (verbose) {
        //    print( "Chat", C );
        //    print( "Aref", Am, An, Adata.data(), lda );
        //    print( "Cref", n, n, opCref.data(), ldc );
        //}

        // reference solution
        // transpose flips uplo
        blas::Uplo op_uplo = uplo;
        if (C.op() != blas::Op::NoTrans) {
            op_uplo = (op_uplo == blas::Uplo::Lower ? blas::Uplo::Upper
                                                     : blas::Uplo::Lower);
        }
        blas::syrk( blas::Layout::ColMajor, op_uplo, A.op(), n, k,
                    alpha, Adata.data(), lda,
                    beta, opCref.data(), ldc );

        //if (verbose) {
        //    print( "Chat_ref", n, n, opCref.data(), ldc );
        //}

        test_assert_equal( C, opCref.data(), ldc, 3*sqrt(k)*eps, 3*sqrt(k)*eps );
    }}}
}

void test_syrk()
{
    test_syrk< float  >();
    test_syrk< double >();
    test_syrk< std::complex<float>  >();
    test_syrk< std::complex<double> >();
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_herk()
{
    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    real_t eps = std::numeric_limits< real_t >::epsilon();
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int n = 50;
    int k = 30;

    real_t alpha, beta;
    lapack::larnv( 1, iseed, 1, &alpha );
    lapack::larnv( 1, iseed, 1, &beta  );
    if (verbose) {
        printf( "alpha = %.4f;\n"
                "beta  = %.4f;\n",
                alpha, beta );
    }

    // test all combinations of op(C), op(A), uplo
    for (int ic = 0; ic < 3; ++ic) {
    for (int ia = 0; ia < 3; ++ia) {
    for (int iu = 0; iu < 2; ++iu) {
        blas::Uplo uplo = uplos[iu];

        // setup C such that op(C) is n-by-n
        int ldc = n + 1;
        std::vector< scalar_t > Cdata( ldc*n );
        lapack::larnv( 1, iseed, Cdata.size(), Cdata.data() );
        slate::Tile< scalar_t > C( n, n, Cdata.data(), ldc, HostNum,
                                   slate::TileKind::UserOwned );
        C.uplo( uplo );
        C.op( ops[ic] );
        assert( C.mb() == n );
        assert( C.nb() == n );

        // set unused data to nan
        scalar_t nan_ = nan("");
        if (uplo == blas::Uplo::Lower) {
            lapack::laset( lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                           &Cdata[ 0 + 1*ldc ], ldc );
        }
        else {
            lapack::laset( lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                           &Cdata[ 1 + 0*ldc ], ldc );
        }

        // opCref = op(C) is n-by-n
        std::vector< scalar_t > opCref( ldc*n );
        copy( C, opCref.data(), ldc );

        // setup A such that op(A) is n-by-k
        int Am = (ia == 0 ? n : k);
        int An = (ia == 0 ? k : n);
        int lda = Am + 1;
        std::vector< scalar_t > Adata( lda*An );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( Am, An, Adata.data(), lda, HostNum,
                                   slate::TileKind::UserOwned );
        A.op( ops[ia] );
        assert( A.mb() == n );
        assert( A.nb() == k );

        if (verbose) {
            printf( "herk( uplo=%c, opA=%c, opC=%c )\n",
                    char(C.uplo()), char(A.op()), char(C.op()) );
        }

        //if (verbose) {
        //    print( "A", A );
        //    print( "C", C );
        //}

        // run test
        try {
            if (C.op() == blas::Op::Trans)  // TODO
                conjugate( C );
            slate::tile::herk( alpha, A, beta, C );
            if (C.op() == blas::Op::Trans)  // TODO
                conjugate( C );

            // It should throw error if and only if
            // C is complex and
            // (C or A is transposed).
            assert( ! (slate::is_complex< scalar_t >::value &&
                       (ic == 1 || ia == 1)) );
        }
        catch (std::exception& e) {
            //printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    (ic == 1 || ia == 1) );
            continue;
        }

        //if (verbose) {
        //    print( "Chat", C );
        //    print( "Aref", Am, An, Adata.data(), lda );
        //    print( "Cref", n, n, opCref.data(), ldc );
        //}

        // reference solution
        // transpose flips uplo
        blas::Uplo op_uplo = uplo;
        if (C.op() != blas::Op::NoTrans) {
            op_uplo = (op_uplo == blas::Uplo::Lower ? blas::Uplo::Upper
                                                     : blas::Uplo::Lower);
        }
        blas::herk( blas::Layout::ColMajor, op_uplo, A.op(), n, k,
                    alpha, Adata.data(), lda,
                    beta, opCref.data(), ldc );

        //if (verbose) {
        //    print( "Chat_ref", n, n, opCref.data(), ldc );
        //}

        test_assert_equal( C, opCref.data(), ldc, 3*sqrt(k)*eps, 3*sqrt(k)*eps );
    }}}
}

void test_herk()
{
    test_herk< float  >();
    test_herk< double >();
    test_herk< std::complex<float>  >();
    test_herk< std::complex<double> >();
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_trsm()
{
    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    real_t eps = std::numeric_limits< real_t >::epsilon();
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int m = 50;
    int n = 30;

    scalar_t alpha;
    lapack::larnv( 1, iseed, 1, &alpha );
    if (verbose) {
        printf( "alpha = %.4f + %.4fi;\n",
                real(alpha), imag(alpha) );
    }

    // test all combinations of op(A), op(B), side, uplo, diag
    for (int ia = 0; ia < 3; ++ia) {
    for (int ib = 0; ib < 3; ++ib) {
    for (int is = 0; is < 2; ++is) {
    for (int iu = 0; iu < 2; ++iu) {
    for (int id = 0; id < 2; ++id) {
        blas::Side side = sides[is];
        blas::Uplo uplo = uplos[iu];
        blas::Diag diag = diags[id];

        // setup A such that op(A) is either m-by-m (left) or n-by-n (right)
        int An = (is == 0 ? m : n);
        int lda = An + 1;
        std::vector< scalar_t > Adata( lda*An );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( An, An, Adata.data(), lda, HostNum,
                                   slate::TileKind::UserOwned );
        A.uplo( uplo );
        A.op( ops[ia] );

        // set unused data to nan
        scalar_t nan_ = nan("");
        if (uplo == blas::Uplo::Lower) {
            lapack::laset( lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                           &Adata[ 0 + 1*lda ], lda );
        }
        else {
            lapack::laset( lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                           &Adata[ 1 + 0*lda ], lda );
        }

        // brute force positive definiteness
        for (int j = 0; j < An; ++j)
            Adata[ j + j*lda ] += An;

        // factor to get well-conditioned triangle
        int info = lapack::potrf( A.uploPhysical(), An, Adata.data(), lda );
        assert( info == 0 );
        SLATE_UNUSED( info );

        // setup B such that op(B) is m-by-n
        int Bm = (ib == 0 ? m : n);
        int Bn = (ib == 0 ? n : m);
        int ldb = Bm + 1;
        std::vector< scalar_t > Bdata( ldb*Bn );
        lapack::larnv( 1, iseed, Bdata.size(), Bdata.data() );
        slate::Tile< scalar_t > B( Bm, Bn, Bdata.data(), ldb, HostNum,
                                   slate::TileKind::UserOwned );
        B.op( ops[ib] );
        assert( B.mb() == m );
        assert( B.nb() == n );

        // opBref = op(B) is m-by-n
        int ldopb = m + 1;
        std::vector< scalar_t > opBref( ldopb*n );
        copy( B, opBref.data(), ldopb );

        if (verbose) {
            printf( "trsm( side=%c, uplo=%c, opA=%c, diag=%c, opB=%c )\n",
                    char(side), char(A.uplo()), char(A.op()), char(diag),
                    char(B.op()) );
        }

        //if (verbose) {
        //    print( "A", A );
        //    print( "B", B );
        //}

        // run test
        try {
            slate::tile::trsm( side, diag, alpha, A, B );

            // It should throw error if and only if
            // B is complex and
            // ((B is transposed and A is conj-transposed) or
            //  (B is conj-transposed and A is tranpsosed)).
            assert( ! (slate::is_complex< scalar_t >::value &&
                       ((ib == 1 && ia == 2) ||
                        (ib == 2 && ia == 1))) );
        }
        catch (std::exception& e) {
            //printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    ((ib == 1 && ia == 2) ||
                     (ib == 2 && ia == 1)) );
            continue;
        }

        //if (verbose) {
        //    print( "Bhat", B );
        //    print( "Aref", An, An, Adata.data(), lda );
        //    print( "Bref", m, n, opBref.data(), ldopb );
        //}

        // reference solution
        blas::trsm( blas::Layout::ColMajor, side, A.uploPhysical(), A.op(), diag, m, n,
                    alpha, Adata.data(), lda,
                           opBref.data(), ldopb );

        //if (verbose) {
        //    print( "Bhat_ref", m, n, opBref.data(), ldopb );
        //}

        test_assert_equal( B, opBref.data(), ldopb, 3*eps, 3*eps );
    }}}}}
}

void test_trsm()
{
    test_trsm< float  >();
    test_trsm< double >();
    test_trsm< std::complex<float>  >();
    test_trsm< std::complex<double> >();
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_potrf()
{
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    real_t eps = std::numeric_limits< real_t >::epsilon();
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int n = 50;

    // test all combinations of op(A), uplo
    for (int ia = 0; ia < 3; ++ia) {
    for (int iu = 0; iu < 2; ++iu) {
        blas::Uplo uplo = uplos[iu];

        // setup A such that op(A) is n-by-n
        int lda = n + 1;
        std::vector< scalar_t > Adata(  lda*n );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( n, n, Adata.data(), lda, HostNum,
                                   slate::TileKind::UserOwned );
        A.uplo( uplo );
        A.op( ops[ia] );

        // set unused data to nan
        scalar_t nan_ = nan("");
        if (uplo == blas::Uplo::Lower) {
            lapack::laset( lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                           &Adata[ 0 + 1*lda ], lda );
        }
        else {
            lapack::laset( lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                           &Adata[ 1 + 0*lda ], lda );
        }

        // brute force positive definiteness
        for (int j = 0; j < n; ++j)
            Adata[ j + j*lda ] += n;

        // opAref = op(A) is n-by-n
        std::vector< scalar_t > opAref( lda*n );
        copy( A, opAref.data(), lda );

        if (verbose) {
            printf( "potrf( op=%c, uplo=%c )\n",
                    char(A.op()), char(A.uplo()) );
        }

        //if (verbose) {
        //    print( "A", A );
        //}

        // run test
        int info = potrf( A );
        test_assert( info == 0 );

        //if (verbose) {
        //    print( "Ahat", A );
        //    print( "opA", n, n, opAref.data(), lda );
        //}

        // reference solution
        // transpose flips uplo
        blas::Uplo op_uplo = uplo;
        if (A.op() != blas::Op::NoTrans) {
            op_uplo = (op_uplo == blas::Uplo::Lower ? blas::Uplo::Upper
                                                    : blas::Uplo::Lower);
        }
        info = lapack::potrf( op_uplo, n, opAref.data(), lda );
        test_assert( info == 0 );

        //if (verbose) {
        //    print( "opAhat", n, n, opAref.data(), lda );
        //}

        test_assert_equal( A, opAref.data(), lda, 3*eps, 3*eps );
    }}
}

void test_potrf()
{
    test_potrf< float  >();
    test_potrf< double >();
    test_potrf< std::complex<float>  >();
    test_potrf< std::complex<double> >();
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_genorm()
{
    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    ///real_t eps = std::numeric_limits< real_t >::epsilon();
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int m = 50;
    int n = 30;

    // test all combinations of norm
    for (int in = 0; in < 4; ++in) {
        lapack::Norm norm = norms[in];

        // setup A
        int lda = m + 1;
        std::vector< scalar_t > Adata( lda*n );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( m, n, Adata.data(), lda, HostNum,
                                   slate::TileKind::UserOwned );
        A.at( 3, 5 ) *= 1e6;

        if (verbose) {
            printf( "genorm( norm=%c )\n",
                    char(norm) );
        }

        //if (verbose) {
        //    print( "A", A );
        //}

        // run test
        std::vector<real_t> values;
        real_t result = -1;
        try {
            // setup values
            if (norm == lapack::Norm::Max) {
                values.resize( 1 );
            }
            else if (norm == lapack::Norm::One) {
                values.resize( A.nb() );
            }
            else if (norm == lapack::Norm::Inf) {
                values.resize( A.mb() );
            }
            else if (norm == lapack::Norm::Fro) {
                values.resize( 2 );
            }

            //---------------------
            // call kernel
            slate::genorm( norm, slate::NormScope::Matrix, A, values.data() );

            // post-process result
            if (norm == lapack::Norm::Max) {
                result = values[0];
            }
            else if (norm == lapack::Norm::One) {
                result = 0;
                for (int j = 0; j < n; ++j) {
                    result = std::max( result, values[j] );
                }
            }
            else if (norm == lapack::Norm::Inf) {
                result = 0;
                for (int i = 0; i < m; ++i) {
                    result = std::max( result, values[i] );
                }
            }
            else if (norm == lapack::Norm::Fro) {
                result = values[0] * sqrt( values[1] );
            }
        }
        catch (const std::exception& ex) {
            printf( "caught unexpected error: %s\n", ex.what() );
            continue;
        }

        //if (verbose) {
        //    print( "values", 1, values.size(), values.data(), 1 );
        //    printf( "result    %.4f\n", result );
        //}

        real_t ref = lapack::lange( norm, m, n, A.data(), A.stride() );
        SLATE_UNUSED(ref);
        //if (verbose) {
        //    printf( "reference %.4f\n", ref );
        //}

        ///test_assert_equal( B, opBref.data(), ldopb, 3*eps, 3*eps );
    }
}

void test_genorm()
{
    test_genorm< float  >();
    test_genorm< double >();
    test_genorm< std::complex<float>  >();
    test_genorm< std::complex<double> >();
}

//------------------------------------------------------------------------------
// square tiles
template <typename scalar_t>
void test_convert_layout(int n)
{
    //printf( "%s\n", __PRETTY_FUNCTION__ );
    using slate::Layout;

    int lda = n + 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    std::vector<scalar_t> Adata( lda*n );
    lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
    std::vector<scalar_t> Bdata = Adata;
    slate::Tile<scalar_t> A( n, n, Adata.data(), lda, HostNum,
                             slate::TileKind::UserOwned );
    slate::Tile<scalar_t> B( n, n, Bdata.data(), lda, HostNum,
                             slate::TileKind::UserOwned );

    test_assert(A.layout() == Layout::ColMajor);
    test_assert(B.layout() == Layout::ColMajor);

    //-----------------------------------------
    // Run kernel.
    A.layoutConvert();

    // Verify layout of A changed.
    test_assert(A.layout() == Layout::RowMajor);
    test_assert(B.layout() == Layout::ColMajor);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            // Check that actual data is transposed.
            test_assert(Adata[ j + i*lda ] == Bdata[ i + j*lda ]);
            // A(i, j) takes col/row-major into account.
            test_assert(A(i, j) == B(i, j));
        }
    }
}

// rectangular tiles
template <typename scalar_t>
void test_convert_layout(int m, int n)
{
    //printf( "%s\n", __PRETTY_FUNCTION__ );
    using slate::Layout;

    int lda = m + 1;
    int ldae = n;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    std::vector<scalar_t> Adata( lda*n );
    std::vector<scalar_t> A_extData( lda*n );
    lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
    std::vector<scalar_t> Bdata = Adata;
    slate::Tile<scalar_t> A( m, n, Adata.data(), lda, HostNum,
                             slate::TileKind::UserOwned );
    slate::Tile<scalar_t> B( m, n, Bdata.data(), lda, HostNum,
                             slate::TileKind::UserOwned );

    test_assert(A.layout() == Layout::ColMajor);
    test_assert(B.layout() == Layout::ColMajor);

    if (m != n) {
        A.makeTransposable(A_extData.data());
    }

    //-----------------------------------------
    // Run kernel.
    double time = omp_get_wtime();
    A.layoutConvert();
    time = omp_get_wtime() - time;
    if (verbose)
        printf( "m %d, n %d, time %.6f\n", m, n, time);

    // Verify layout of A changed.
    test_assert(A.layout() == Layout::RowMajor);
    test_assert(B.layout() == Layout::ColMajor);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            // Check that actual data is transposed.
            if (m != n)
                test_assert(A_extData[ j + i*ldae ] == Bdata[ i + j*lda ]);
            else
                test_assert(Adata[ j + i*lda ] == Bdata[ i + j*lda ]);
            // A(i, j) takes col/row-major into account.
            test_assert(A(i, j) == B(i, j));
        }
    }

    // convert back
    //-----------------------------------------
    // Run kernel.
    A.layoutConvert();

    // Verify layout of A changed.
    test_assert(A.layout() == Layout::ColMajor);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            // Check that actual data is transposed back.
            test_assert(Adata[ i + j*lda ] == Bdata[ i + j*lda ]);
            // A(i, j) takes col/row-major into account.
            test_assert(A(i, j) == B(i, j));
        }
    }
}

void test_convert_layout()
{
    int n = 256;
    test_convert_layout< float  >(n);
    test_convert_layout< double >(n);
    test_convert_layout< std::complex<float>  >(n);
    test_convert_layout< std::complex<double> >(n);

    int m = 256;
    test_convert_layout< float  >(m, n);
    test_convert_layout< double >(m, n);
    test_convert_layout< std::complex<float>  >(m, n);
    test_convert_layout< std::complex<double> >(m, n);

    m = 128;
    test_convert_layout< float  >(m, n);
    test_convert_layout< double >(m, n);
    test_convert_layout< std::complex<float>  >(m, n);
    test_convert_layout< std::complex<double> >(m, n);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_device_convert_layout(int m, int n)
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    using blas::real;

    int batch_count = 100;
    int lda = m;
    int ldat = n;
    int repeat = 1;
    int device = 0;

    // setup batch A and copy B on CPU
    int64_t iseed[4] = { 0, 1, 2, 3 };
    std::vector< scalar_t > Adata( lda * n * batch_count );
    lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
    std::vector< scalar_t > Bdata = Adata;
    std::vector< slate::Tile<scalar_t> > Atiles( batch_count );
    std::vector< slate::Tile<scalar_t> > Btiles( batch_count );
    for (int k = 0; k < batch_count; ++k) {
        Atiles[k] = slate::Tile<scalar_t>( m, n, &Adata[ k*lda*n ], lda, HostNum, slate::TileKind::UserOwned );
        Btiles[k] = slate::Tile<scalar_t>( m, n, &Bdata[ k*lda*n ], lda, HostNum, slate::TileKind::UserOwned );
    }

    // copy batch A to GPU
    scalar_t* Adata_dev;
    const int batch_arrays_index = 0;
    blas::Queue queue(device, batch_arrays_index);

    Adata_dev = blas::device_malloc<scalar_t>(Adata.size(), queue);
    blas::device_memcpy<scalar_t>(Adata_dev, Adata.data(),
                        Adata.size(),
                        blas::MemcpyKind::HostToDevice,
                        queue);

    scalar_t* Adata_dev_ext;
    Adata_dev_ext = blas::device_malloc<scalar_t>(Adata.size(), queue);

    std::vector< slate::Tile<scalar_t> > Atiles_dev( batch_count );
    std::vector< scalar_t* > Aarray( batch_count );
    std::vector< scalar_t* > Aarray_ext( batch_count );
    for (int k = 0; k < batch_count; ++k) {
        Atiles_dev[k] = slate::Tile<scalar_t>( m, n, &Adata_dev[ k*lda*n ], lda, device, slate::TileKind::UserOwned );
        Aarray[k] = &Adata_dev[ k*lda*n ];
        Aarray_ext[k] = &Adata_dev_ext[ k*lda*n ];
    }
    scalar_t** Aarray_dev;
    Aarray_dev = blas::device_malloc<scalar_t*>(Aarray.size(), queue);
    blas::device_memcpy<scalar_t*>(Aarray_dev, Aarray.data(),
                        Aarray.size(),
                        blas::MemcpyKind::HostToDevice,
                        queue);

    scalar_t** Aarray_dev_ext;
    Aarray_dev_ext = blas::device_malloc<scalar_t*>(Aarray_ext.size(), queue);
    blas::device_memcpy<scalar_t*>(Aarray_dev_ext, Aarray_ext.data(),
                        Aarray_ext.size(),
                        blas::MemcpyKind::HostToDevice,
                        queue);

    if (verbose > 1) {
        printf("A = [\n");
        for (int k = 0; k < batch_count; ++k) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    printf(" %5.2f", real(Adata[ i + j*lda + k*lda*n ]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("];\n");
    }

    //-----------------------------------------
    // Run kernel.
    for (int i = 0; i < repeat; ++i) {
        queue.sync();
        double time = omp_get_wtime();

        if (m == n)
            slate::device::transpose_batch(n, Aarray_dev, lda, batch_count, queue);
        else
            slate::device::transpose_batch(m, n, Aarray_dev, lda, Aarray_dev_ext, ldat, batch_count, queue);

        queue.sync();
        time = omp_get_wtime() - time;
        if (verbose) {
            printf( "batch_count %d, m %d, n %d, time %.6f, GB/s (read & write) %.4f batch\n",
                    batch_count, m, n, time, 2 * Adata.size() * sizeof(scalar_t) * 1e-9 / time);
        }
    }
    if (verbose)
        printf( "\n" );
    blas::device_memcpy<scalar_t>(Adata.data(),
                        (m == n ? Adata_dev : Adata_dev_ext),
                        Adata.size(),
                        blas::MemcpyKind::DeviceToHost,
                        queue);

    //-----------------------------------------
    // Run kernel.
    for (int i = 0; i < repeat; ++i) {
        queue.sync();
        double time = omp_get_wtime();

        if (m == n) {
            for (int k = 0; k < batch_count; ++k) {
                slate::device::transpose(n, Aarray[k], lda, queue);
            }
        }
        else {
            for (int k = 0; k < batch_count; ++k) {
                slate::device::transpose(m, n, Aarray[k], lda, Aarray_ext[k], ldat, queue);
                // Atiles[k].stride(ldat);
            }
        }

        queue.sync();
        time = omp_get_wtime() - time;
        if (verbose) {
            printf( "batch_count %d, m %d, n %d, time %.6f, GB/s (read & write) %.4f 1-by-1\n",
                    batch_count, m, n, time, 2 * Adata.size() * sizeof(scalar_t) * 1e-9 / time);
        }
    }
    if (verbose)
        printf( "\n" );

    //-----------------------------------------
    // Run kernel.
    for (int i = 0; i < repeat; ++i) {
        queue.sync();
        double time = omp_get_wtime();

        for (int k = 0; k < batch_count; ++k) {
            if (m == n)
                Atiles_dev[k].layoutConvert(queue);
            else
                Atiles_dev[k].layoutConvert(Aarray_ext[k], queue);
        }

        queue.sync();
        time = omp_get_wtime() - time;
        if (verbose) {
            printf( "batch_count %d, m %d, n %d, time %.6f, GB/s (read & write) %.4f 1-by-1\n",
                    batch_count, m, n, time, 2 * Adata.size() * sizeof(scalar_t) * 1e-9 / time);
        }
    }
    if (verbose)
        printf( "\n" );
    blas::device_memcpy<scalar_t>(Adata.data(),
                        (m == n ? Adata_dev : Adata_dev_ext),
                        Adata.size(),
                        blas::MemcpyKind::DeviceToHost,
                        queue);

    if (verbose > 1) {
        printf("AT = [\n");
        for (int k = 0; k < batch_count; ++k) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    printf(" %5.2f", real(Adata[ i + j*lda + k*lda*n ]));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("];\n");
    }

    // Verify layout of A changed.
    for (int k = 0; k < batch_count; ++k) {
        test_assert(Atiles_dev[k].layout() == slate::Layout::RowMajor);
        Atiles[k].layout(slate::Layout::RowMajor);
        Atiles[k].stride(ldat);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                // A(i, j) takes col/row-major into account.
                // Check that actual data is transposed.
                int Adata_lda;
                if (m == n)
                    Adata_lda = lda;
                else
                    Adata_lda = ldat;
                if (Adata[ j + i*Adata_lda + k*lda*n ] != Bdata[ i + j*lda + k*lda*n ]) {
                    printf( "Adata[ j(%d) + i(%d)*lda + k(%d)*lda*n ] %5.2f\n"
                            "Bdata[ i(%d) + j(%d)*lda + k(%d)*lda*n ] %5.2f\n",
                            j, i, k, real(Adata[ j + i*Adata_lda + k*lda*n ]),
                            i, j, k, real(Bdata[ i + j*lda + k*lda*n ]) );
                }
                test_assert(Adata[ j + i*Adata_lda + k*lda*n ] == Bdata[ i + j*lda + k*lda*n ]);
                test_assert(Atiles[k](i, j) == Btiles[k](i, j));
            }
        }
    }

    blas::device_free(Adata_dev, queue);
    blas::device_free(Adata_dev_ext, queue);
    blas::device_free(Aarray_dev, queue);
}

void test_device_convert_layout()
{
    int m = 256, n = 256;
    test_device_convert_layout< float  >(m, n);
    test_device_convert_layout< double >(m, n);
    test_device_convert_layout< std::complex<float>  >(m, n);
    test_device_convert_layout< std::complex<double> >(m, n);

    m = 128;
    test_device_convert_layout< float  >(m, n);
    test_device_convert_layout< double >(m, n);
    test_device_convert_layout< std::complex<float>  >(m, n);
    test_device_convert_layout< std::complex<double> >(m, n);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_deepTranspose_work(int m, int n)
{
    if (verbose)
        printf( "%s< %s >( m=%3d, n=%3d )\n", __func__, type_name<scalar_t>().c_str(), m, n );

    using blas::conj;

    int lda  = slate::roundup( m, 16 );
    int ldat = slate::roundup( n, 16 );
    std::vector<scalar_t> data( lda*n );
    std::vector<scalar_t> dataT( m*ldat );

    int64_t idist = 3;
    int64_t iseed[4] = { 1, 2, 3, 5 };
    lapack::larnv( idist, iseed, data.size(), data.data() );
    lapack::larnv( idist, iseed, dataT.size(), dataT.data() );

    slate::Tile< scalar_t > A( m, n, data.data(), lda, HostNum,
                               slate::TileKind::UserOwned );
    slate::Tile< scalar_t > AT( n, m, dataT.data(), ldat, HostNum,
                                slate::TileKind::UserOwned );

    slate::tile::deepTranspose( std::move( A ), std::move( AT ) );
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            test_assert( A(i, j) == AT(j, i) );

    // deepTranspose( std::move(A), std::move(A) );  // error

    if (m == n) {
        slate::tile::deepTranspose( std::move( A ) );
        // Check that A == A^T.
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                test_assert( A(i, j) == AT(i, j) );
    }
    else {
        // deepTranspose( std::move(A) );  // error
    }
}

void test_deepTranspose()
{
    for (int m = 8; m <= 16; m += 2) {
        for (int n = 8; n <= 16; n += 2) {
            test_deepTranspose_work< float  >( m, n );
            test_deepTranspose_work< double >( m, n );
            test_deepTranspose_work< std::complex<float>  >( m, n );
            test_deepTranspose_work< std::complex<double> >( m, n );
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_deepConjTranspose_work(int m, int n)
{
    if (verbose)
        printf( "%s< %s >( m=%3d, n=%3d )\n", __func__, type_name<scalar_t>().c_str(), m, n );

    using blas::conj;

    int lda  = slate::roundup( m, 16 );
    int ldah = slate::roundup( n, 16 );
    std::vector<scalar_t> data( lda*n );
    std::vector<scalar_t> dataH( m*ldah );

    int64_t idist = 3;
    int64_t iseed[4] = { 1, 2, 3, 5 };
    lapack::larnv( idist, iseed, data.size(), data.data() );
    lapack::larnv( idist, iseed, dataH.size(), dataH.data() );

    slate::Tile< scalar_t > A( m, n, data.data(), lda, HostNum,
                               slate::TileKind::UserOwned );
    slate::Tile< scalar_t > AH( n, m, dataH.data(), ldah, HostNum,
                                slate::TileKind::UserOwned );

    slate::tile::deepConjTranspose( std::move( A ), std::move( AH ) );
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            test_assert( A(i, j) == conj(AH(j, i)) );

    // deepConjTranspose( std::move(A), std::move(A) );  // error

    if (m == n) {
        slate::tile::deepConjTranspose( std::move( A ) );
        // Check that A == A^H.
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                test_assert( A(i, j) == AH(i, j) );
    }
    else {
        // deepConjTranspose( std::move(A) );  // error
    }
}

void test_deepConjTranspose()
{
    for (int m = 8; m <= 16; m += 2) {
        for (int n = 8; n <= 16; n += 2) {
            test_deepConjTranspose_work< float  >( m, n );
            test_deepConjTranspose_work< double >( m, n );
            test_deepConjTranspose_work< std::complex<float>  >( m, n );
            test_deepConjTranspose_work< std::complex<double> >( m, n );
        }
    }
}

//------------------------------------------------------------------------------
enum class Section {
    newline = 0,  // zero flag forces newline
    blas_section,
    norm,
    factor,
    convert,
    copy,
};

//------------------------------------------------------------------------------
// Similar routine list to testsweeper. No params yet.
typedef void (*test_func_ptr)();

typedef struct {
    const char* name;
    test_func_ptr func;
    Section section;
} routines_t;

//------------------------------------------------------------------------------
std::vector< routines_t > routines = {
    { "gemm",   test_gemm,   Section::blas_section },
    { "syrk",   test_syrk,   Section::blas_section },
    { "herk",   test_herk,   Section::blas_section },
    { "trsm",   test_trsm,   Section::blas_section },
    { "",       nullptr,     Section::newline      },

    { "genorm", test_genorm, Section::norm         },
    { "",       nullptr,     Section::newline      },

    { "potrf",  test_potrf,  Section::factor       },
    { "",       nullptr,     Section::newline      },

    { "convert_layout",        test_convert_layout,        Section::convert },
    { "device_convert_layout", test_device_convert_layout, Section::convert },
    { "",                      nullptr,                    Section::newline },

    { "deepTranspose",         test_deepTranspose,         Section::copy },
    { "deepConjTranspose",     test_deepConjTranspose,     Section::copy },
    { "",                      nullptr,                    Section::newline },
};

//------------------------------------------------------------------------------
// todo: usage as in testsweeper.
void usage()
{
    printf("Usage: %s [routines]\n", g_argv[0]);
    int col = 0;
    Section last_section = routines[0].section;
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
    if (g_argc == 1) {
        // run all tests
        for (size_t j = 0; j < routines.size(); ++j)
            if (routines[j].func != nullptr)
                run_test(routines[j].func, routines[j].name, MPI_COMM_WORLD);
    }
    else {
        // run tests mentioned on command line
        for (int i = 1; i < g_argc; ++i) {
            std::string arg = g_argv[i];
            if (arg == "-h" || arg == "--help") {
                usage();
                break;
            }
            if (arg == "-v" || arg == "--verbose") {
                ++verbose;
                continue;
            }
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
