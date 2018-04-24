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
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#include "test.hh"
#include "slate_Tile.hh"
#include "slate_Tile_blas.hh"

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// conjugates the matrix A, in-place.
template <typename scalar_t>
void conjugate( int m, int n, scalar_t* A, int lda )
{
    using blas::conj;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[ i + j*lda ] = conj( A[ i + j*lda ] );
}

// -----------------------------------------------------------------------------
// conjugates the tile A, in-place.
template <typename scalar_t>
void conjugate( slate::Tile< scalar_t >& A )
{
    using blas::conj;
    for (int j = 0; j < A.nb(); ++j)
        for (int i = 0; i < A.mb(); ++i)
            A(i,j) = conj( A(i,j) );
}

// -----------------------------------------------------------------------------
// copy op(A) to opAref
template <typename scalar_t>
void copy( slate::Tile< scalar_t > const& A, scalar_t* opAref, int lda )
{
    using blas::conj;
    for (int j = 0; j < A.nb(); ++j) {
        for (int i = 0; i < A.mb(); ++i) {
            // currently, A(i,j) does transpose but not conj.
            if (A.op() == blas::Op::ConjTrans)
                opAref[ i + j*lda ] = conj( A(i,j) );
            else
                opAref[ i + j*lda ] = A(i,j);
        }
    }
}

// -----------------------------------------------------------------------------
// check op(A) == B, within absolute or relative tolerance.
// Assert aborts on failure.
template <typename scalar_t>
void test_assert_equal( slate::Tile< scalar_t > const& A, scalar_t const* B, int ldb,
                        blas::real_type<scalar_t> abs_tol=0,
                        blas::real_type<scalar_t> rel_tol=0 )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::conj;

    // whether uplo(A) is general, lower, or upper
    bool general = (A.uplo() == blas::Uplo::General);
    bool lower =
        (A.uplo() == blas::Uplo::Lower && A.op() == blas::Op::NoTrans) ||
        (A.uplo() == blas::Uplo::Upper && A.op() != blas::Op::NoTrans);
    bool upper =
        (A.uplo() == blas::Uplo::Upper && A.op() == blas::Op::NoTrans) ||
        (A.uplo() == blas::Uplo::Lower && A.op() != blas::Op::NoTrans);
    assert( general || lower || upper );

    for (int j = 0; j < A.nb(); ++j) {
        for (int i = 0; i < A.mb(); ++i) {
            if (general || (lower && i >= j) || (upper && i <= j)) {
                real_t abs_error;
                // currently, A(i,j) does transpose but not conj.
                if (A.op() == blas::Op::ConjTrans)
                    abs_error = std::abs( conj( A(i,j) ) - B[ i + j*ldb ] );
                else
                    abs_error = std::abs( A(i,j) - B[ i + j*ldb ] );
                real_t rel_error = abs_error / std::abs( A(i,j) );

                // print elements if assert will fail
                if (! (abs_error <= abs_tol || rel_error <= rel_tol)) {
                    if (A.op() == blas::Op::ConjTrans) {
                        printf( "A(%3d, %3d) %8.4f + %8.4fi\n"
                                "B           %8.4f + %8.4fi, abs_error %.2e, rel_error %.2e\n",
                                i, j, real( A(i,j) ), -imag( A(i,j) ),
                                      real( B[ i + j*ldb ] ), imag( B[ i + j*ldb ] ),
                                abs_error, rel_error );
                    }
                    else {
                        printf( "A(%3d, %3d) %8.4f + %8.4fi\n"
                                "B           %8.4f + %8.4fi, abs_error %.2e, rel_error %.2e\n",
                                i, j, real( A(i,j) ), imag( A(i,j) ),
                                      real( B[ i + j*ldb ] ), imag( B[ i + j*ldb ] ),
                                abs_error, rel_error );
                    }
                }

                test_assert( abs_error <= abs_tol || rel_error <= rel_tol );
            }
        }
    }
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_gemm()
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + " >");
    Test name( msg.c_str() );

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
    if (g_verbose) {
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
        slate::Tile< scalar_t > C( Cm, Cn, Cdata.data(), ldc, g_host_num );
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
        slate::Tile< scalar_t > B( Bm, Bn, Bdata.data(), ldb, g_host_num );
        B.op( ops[ib] );
        assert( B.mb() == k );
        assert( B.nb() == n );

        // setup A such that op(A) is m-by-k
        int Am = (ia == 0 ? m : k);
        int An = (ia == 0 ? k : m);
        int lda = Am + 1;
        std::vector< scalar_t > Adata( lda*An );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( Am, An, Adata.data(), lda, g_host_num );
        A.op( ops[ia] );
        assert( A.mb() == m );
        assert( A.nb() == k );

        test_message( "gemm( opA=%c, opB=%c, opC=%c )",
                      char(A.op()), char(B.op()), char(C.op()) );

        if (g_verbose) {
            print( "A", A );
            print( "B", B );
            print( "C", C );
        }

        // run test
        try {
            gemm( alpha, A, B, beta, C );

            // It should throw error if and only if
            // C is complex and
            // ((C is transposed and either A or B is conj-transposed) or
            //  (C is conj-transposed and either A or B is tranpsosed)).
            assert( ! (slate::is_complex< scalar_t >::value &&
                       ((ic == 1 && (ia == 2 || ib == 2)) ||
                        (ic == 2 && (ia == 1 || ib == 1)))) );
        }
        catch (std::exception& e) {
            printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    ((ic == 1 && (ia == 2 || ib == 2)) ||
                     (ic == 2 && (ia == 1 || ib == 1))) );
            continue;
        }

        if (g_verbose) {
            print( "Chat", C );
            print( "Aref", Am, An, Adata.data(), lda );
            print( "Bref", Bm, Bn, Bdata.data(), ldb );
            print( "Cref", m, n, opCref.data(), ldopc );
        }

        // reference solution
        blas::gemm( blas::Layout::ColMajor, A.op(), B.op(), m, n, k,
                    alpha, Adata.data(), lda,
                           Bdata.data(), ldb,
                    beta, opCref.data(), ldopc );

        if (g_verbose) {
            print( "Chat_ref", m, n, opCref.data(), ldopc );
        }

        test_assert_equal( C, opCref.data(), ldopc, 3*sqrt(k)*eps, 3*sqrt(k)*eps );
    }}}
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_syrk()
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + " >");
    Test name( msg.c_str() );

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
    if (g_verbose) {
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
        slate::Tile< scalar_t > C( n, n, Cdata.data(), ldc, g_host_num );
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
        // currently, C(i,j) does transpose but not conj.
        std::vector< scalar_t > opCref( ldc*n );
        copy( C, opCref.data(), ldc );

        // setup A such that op(A) is n-by-k
        int Am = (ia == 0 ? n : k);
        int An = (ia == 0 ? k : n);
        int lda = Am + 1;
        std::vector< scalar_t > Adata( lda*An );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( Am, An, Adata.data(), lda, g_host_num );
        A.op( ops[ia] );
        assert( A.mb() == n );
        assert( A.nb() == k );

        test_message( "syrk( uplo=%c, opA=%c, opC=%c )",
                      char(C.uplo()), char(A.op()), char(C.op()) );

        if (g_verbose) {
            print( "A", A );
            print( "C", C );
        }

        // run test
        try {
            if (C.op() == blas::Op::ConjTrans)  // TODO
                conjugate( C );
            syrk( alpha, A, beta, C );
            if (C.op() == blas::Op::ConjTrans)  // TODO
                conjugate( C );

            // It should throw error if and only if
            // C is complex and
            // C is conj-transposed or A is conj-transposed.
            assert( ! (slate::is_complex< scalar_t >::value &&
                       (ic == 2 || ia == 2)) );
        }
        catch (std::exception& e) {
            printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    (ic == 2 || ia == 2) );
            continue;
        }

        if (g_verbose) {
            print( "Chat", C );
            print( "Aref", Am, An, Adata.data(), lda );
            print( "Cref", n, n, opCref.data(), ldc );
        }

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

        if (g_verbose) {
            print( "Chat_ref", n, n, opCref.data(), ldc );
        }

        test_assert_equal( C, opCref.data(), ldc, 3*sqrt(k)*eps, 3*sqrt(k)*eps );
    }}}
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_herk()
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + " >");
    Test name( msg.c_str() );

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
    if (g_verbose) {
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
        slate::Tile< scalar_t > C( n, n, Cdata.data(), ldc, g_host_num );
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
        // currently, C(i,j) does transpose but not conj.
        std::vector< scalar_t > opCref( ldc*n );
        copy( C, opCref.data(), ldc );

        // setup A such that op(A) is n-by-k
        int Am = (ia == 0 ? n : k);
        int An = (ia == 0 ? k : n);
        int lda = Am + 1;
        std::vector< scalar_t > Adata( lda*An );
        lapack::larnv( 1, iseed, Adata.size(), Adata.data() );
        slate::Tile< scalar_t > A( Am, An, Adata.data(), lda, g_host_num );
        A.op( ops[ia] );
        assert( A.mb() == n );
        assert( A.nb() == k );

        test_message( "herk( uplo=%c, opA=%c, opC=%c )",
                      char(C.uplo()), char(A.op()), char(C.op()) );

        if (g_verbose) {
            print( "A", A );
            print( "C", C );
        }

        // run test
        try {
            if (C.op() == blas::Op::Trans)  // TODO
                conjugate( C );
            herk( alpha, A, beta, C );
            if (C.op() == blas::Op::Trans)  // TODO
                conjugate( C );

            // It should throw error if and only if
            // C is complex and
            // (C or A is transposed).
            assert( ! (slate::is_complex< scalar_t >::value &&
                       (ic == 1 || ia == 1)) );
        }
        catch (std::exception& e) {
            printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    (ic == 1 || ia == 1) );
            continue;
        }

        if (g_verbose) {
            print( "Chat", C );
            print( "Aref", Am, An, Adata.data(), lda );
            print( "Cref", n, n, opCref.data(), ldc );
        }

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

        if (g_verbose) {
            print( "Chat_ref", n, n, opCref.data(), ldc );
        }

        test_assert_equal( C, opCref.data(), ldc, 3*sqrt(k)*eps, 3*sqrt(k)*eps );
    }}}
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_potrf()
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + " >");
    Test name( msg.c_str() );

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
        slate::Tile< scalar_t > A( n, n, Adata.data(), lda, g_host_num );
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
        // currently, A(i,j) does transpose but not conj.
        std::vector< scalar_t > opAref( lda*n );
        copy( A, opAref.data(), lda );

        test_message( "potrf( op=%c, uplo=%c )",
                      char(A.op()), char(A.uplo()) );

        if (g_verbose) {
            print( "A", A );
        }

        // run test
        int info = potrf( A );
        test_assert( info == 0 );

        if (g_verbose) {
            print( "Ahat", A );
            print( "opA", n, n, opAref.data(), lda );
        }

        // reference solution
        // transpose flips uplo
        blas::Uplo op_uplo = uplo;
        if (A.op() != blas::Op::NoTrans) {
            op_uplo = (op_uplo == blas::Uplo::Lower ? blas::Uplo::Upper
                                                     : blas::Uplo::Lower);
        }
        info = lapack::potrf( op_uplo, n, opAref.data(), lda );
        test_assert( info == 0 );

        if (g_verbose) {
            print( "opAhat", n, n, opAref.data(), lda );
        }

        test_assert_equal( A, opAref.data(), lda, 3*eps, 3*eps );
    }}
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_trsm()
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + " >");
    Test name( msg.c_str() );

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
    if (g_verbose) {
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
        slate::Tile< scalar_t > A( An, An, Adata.data(), lda, g_host_num );
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
        int info = lapack::potrf( A.uplo(), An, Adata.data(), lda );
        assert( info == 0 );

        // setup B such that op(B) is m-by-n
        int Bm = (ib == 0 ? m : n);
        int Bn = (ib == 0 ? n : m);
        int ldb = Bm + 1;
        std::vector< scalar_t > Bdata( ldb*Bn );
        lapack::larnv( 1, iseed, Bdata.size(), Bdata.data() );
        slate::Tile< scalar_t > B( Bm, Bn, Bdata.data(), ldb, g_host_num );
        B.op( ops[ib] );
        assert( B.mb() == m );
        assert( B.nb() == n );

        // opBref = op(B) is m-by-n
        // currently, B(i,j) does transpose but not conj.
        int ldopb = m + 1;
        std::vector< scalar_t > opBref( ldopb*n );
        copy( B, opBref.data(), ldopb );

        test_message( "trsm( side=%c, uplo=%c, opA=%c, diag=%c, opB=%c )",
                      char(side), char(A.uplo()), char(A.op()), char(diag), char(B.op()) );

        if (g_verbose) {
            print( "A", A );
            print( "B", B );
        }

        // run test
        try {
            trsm( side, diag, alpha, A, B );

            // It should throw error if and only if
            // B is complex and
            // ((B is transposed and A is conj-transposed) or
            //  (B is conj-transposed and A is tranpsosed)).
            assert( ! (slate::is_complex< scalar_t >::value &&
                       ((ib == 1 && ia == 2) ||
                        (ib == 2 && ia == 1))) );
        }
        catch (std::exception& e) {
            printf( "%%      not allowed\n" );
            assert( slate::is_complex< scalar_t >::value &&
                    ((ib == 1 && ia == 2) ||
                     (ib == 2 && ia == 1)) );
            continue;
        }

        if (g_verbose) {
            print( "Bhat", B );
            print( "Aref", An, An, Adata.data(), lda );
            print( "Bref", m, n, opBref.data(), ldopb );
        }

        // reference solution
        blas::trsm( blas::Layout::ColMajor, side, A.uplo(), A.op(), diag, m, n,
                    alpha, Adata.data(), lda,
                           opBref.data(), ldopb );

        if (g_verbose) {
            print( "Bhat_ref", m, n, opBref.data(), ldopb );
        }

        test_assert_equal( B, opBref.data(), ldopb, 3*eps, 3*eps );
    }}}}}
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    g_mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank( g_mpi_comm, &g_mpi_rank );

    bool do_all   = true;
    bool do_gemm  = false;
    bool do_syrk  = false;
    bool do_herk  = false;
    bool do_potrf = false;
    bool do_trsm  = false;

    // parse options
    for (int i = 1; i < argc; ++i) {
        std::string arg( argv[i] );
        if (arg == "-v") {
            g_verbose = true;
        }
        else if (arg == "gemm") {
            do_gemm = true;
            do_all  = false;
        }
        else if (arg == "syrk") {
            do_syrk = true;
            do_all  = false;
        }
        else if (arg == "herk") {
            do_herk = true;
            do_all  = false;
        }
        else if (arg == "potrf") {
            do_potrf = true;
            do_all   = false;
        }
        else if (arg == "trsm") {
            do_trsm = true;
            do_all  = false;
        }
    }

    // run tests
    if (do_all || do_gemm) {
        test_gemm< double >();
        test_gemm< std::complex<double> >();
    }
    if (do_all || do_syrk) {
        test_syrk< double >();
        test_syrk< std::complex<double> >();
    }
    if (do_all || do_herk) {
        test_herk< double >();
        test_herk< std::complex<double> >();
    }
    if (do_all || do_potrf) {
        test_potrf< double >();
        test_potrf< std::complex<double> >();
    }
    if (do_all || do_trsm) {
        test_trsm< double >();
        test_trsm< std::complex<double> >();
    }

    MPI_Finalize();
    return 0;
}
