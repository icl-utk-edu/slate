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
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "test.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"

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

slate::Target targets[] = {
    slate::Target::HostTask,
    slate::Target::HostNest,
    slate::Target::HostBatch,
    slate::Target::Devices,
};

// -----------------------------------------------------------------------------
const char* target_name(slate::Target target)
{
    switch (target) {
        case slate::Target::Host:      return "Host";
        case slate::Target::HostTask:  return "HostTask";
        case slate::Target::HostNest:  return "HostNest";
        case slate::Target::HostBatch: return "HostBatch";
        case slate::Target::Devices:   return "Devices";
        default: assert(false);
    }
}

// -----------------------------------------------------------------------------
// conjugates the matrix A, in-place.
template <typename scalar_t>
void conjugate(int m, int n, scalar_t* A, int lda)
{
    using blas::conj;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[ i + j*lda ] = conj(A[ i + j*lda ]);
}

// -----------------------------------------------------------------------------
// conjugates the tile A, in-place.
template <typename scalar_t>
void conjugate(slate::Tile<scalar_t>& A)
{
    using blas::conj;
    for (int j = 0; j < A.nb(); ++j)
        for (int i = 0; i < A.mb(); ++i)
            A(i,j) = conj(A(i,j));
}

// -----------------------------------------------------------------------------
// copy op(A) to opAref
template <typename scalar_t>
void copy(slate::Matrix<scalar_t> /*const*/ & A, scalar_t* opAref, int lda)
{
    using blas::Op;
    using blas::conj;

    int jj = 0;
    for (int j = 0; j < A.nt(); ++j) {
        int jb = A.tileNb( j );
        int jj_save = jj;

        int ii = 0;
        for (int i = 0; i < A.mt(); ++i) {
            int ib = A.tileMb(i);
            int ii_save = ii;

            auto Aij = A(i,j);
            assert(Aij.mb() == ib);
            assert(Aij.nb() == jb);
            jj = jj_save;
            for (int jt = 0; jt < jb; ++jt, ++jj) {
                ii = ii_save;
                for (int it = 0; it < ib; ++it, ++ii) {
                    opAref[ ii + jj*lda ] = Aij(it, jt);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// copy op(A) to opAref
template <typename scalar_t>
void copy(slate::BaseTrapezoidMatrix<scalar_t> /*const*/ & A,
           scalar_t* opAref, int lda)
{
    using blas::Uplo;
    using blas::Op;
    using blas::conj;

    // whether uplo(A) is lower or upper
    bool lower = (A.uploLogical() == Uplo::Lower);
    bool upper = ! lower;

    int jj = 0;
    for (int j = 0; j < A.nt(); ++j) {
        int jb = A.tileNb(j);
        int jj_save = jj;

        int ii = 0;
        for (int i = 0; i < A.mt(); ++i) {
            int ib = A.tileMb(i);
            int ii_save = ii;

            if ((lower && i >= j) || (upper && i <= j)) {
                auto Aij = A(i,j);
                assert(Aij.mb() == ib);
                assert(Aij.nb() == jb);
                jj = jj_save;
                for (int jt = 0; jt < jb; ++jt, ++jj) {
                    ii = ii_save;
                    for (int it = 0; it < ib; ++it, ++ii) {
                        opAref[ ii + jj*lda ] = Aij(it, jt);
                    }
                }
            }
            else {
                jj = jj_save;
                for (int jt = 0; jt < jb; ++jt, ++jj) {
                    ii = ii_save;
                    for (int it = 0; it < ib; ++it, ++ii) {
                        opAref[ ii + jj*lda ] = nan("");
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// check op(A) == B, within absolute or relative tolerance.
// Assert aborts on failure.
template <typename scalar_t>
void test_assert_equal(slate::Matrix<scalar_t> /*const*/& A,
                        scalar_t const* B, int ldb,
                        blas::real_type<scalar_t> abs_tol=0,
                        blas::real_type<scalar_t> rel_tol=0 )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::conj;
    using blas::Op;

    int jj = 0;
    for (int j = 0; j < A.nt(); ++j) {
        int jb = A.tileNb(j);
        int jj_save = jj;

        int ii = 0;
        for (int i = 0; i < A.mt(); ++i) {
            int ib = A.tileMb(i);
            int ii_save = ii;

            auto Aij = A(i,j);
            assert(Aij.mb() == ib);
            assert(Aij.nb() == jb);
            jj = jj_save;
            for (int jt = 0; jt < jb; ++jt, ++jj) {
                ii = ii_save;
                for (int it = 0; it < ib; ++it, ++ii) {
                    real_t abs_error;
                    abs_error = std::abs(Aij(it, jt) - B[ ii + jj*ldb ]);
                    real_t rel_error = abs_error / std::abs(Aij(it, jt));

                    // print elements if assert will fail
                    if (! (abs_error <= abs_tol || rel_error <= rel_tol)) {
                        printf("A(%3d, %3d) %8.4f + %8.4fi\n"
                               "B           %8.4f + %8.4fi, abs_error %.2e, rel_error %.2e\n",
                               ii, jj,
                               real(Aij(it, jt)), imag(Aij(it, jt)),
                               real(B[ ii + jj*ldb ]), imag(B[ ii + jj*ldb ]),
                               abs_error, rel_error);
                    }

                    test_assert(abs_error <= abs_tol || rel_error <= rel_tol);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// check op(A) == B, within absolute or relative tolerance.
// Assert aborts on failure.
template <typename scalar_t>
void test_assert_equal(slate::BaseTrapezoidMatrix<scalar_t> /*const*/& A,
                       scalar_t const* B, int ldb,
                       blas::real_type<scalar_t> abs_tol=0,
                       blas::real_type<scalar_t> rel_tol=0 )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::conj;
    using blas::Uplo;
    using blas::Op;

    // whether uplo(A) is lower or upper
    bool lower = (A.uploLogical() == Uplo::Lower);
    bool upper = ! lower;

    int jj = 0;
    for (int j = 0; j < A.nt(); ++j) {
        int jb = A.tileNb(j);
        int jj_save = jj;

        int ii = 0;
        for (int i = 0; i < A.mt(); ++i) {
            int ib = A.tileMb(i);
            int ii_save = ii;

            if ((lower && i >= j) || (upper && i <= j)) {
                auto Aij = A(i,j);
                assert(Aij.mb() == ib);
                assert(Aij.nb() == jb);
                jj = jj_save;
                for (int jt = 0; jt < jb; ++jt, ++jj) {
                    ii = ii_save;
                    for (int it = 0; it < ib; ++it, ++ii) {
                        if ((lower && ii >= jj) || (upper && ii <= jj)) {
                            real_t abs_error;
                            abs_error = std::abs(Aij(it, jt) - B[ ii + jj*ldb ]);
                            real_t rel_error = abs_error / std::abs(Aij(it, jt));

                            // print elements if assert will fail
                            if (! (abs_error <= abs_tol || rel_error <= rel_tol)) {
                                printf("A(%3d, %3d) %8.4f + %8.4fi\n"
                                       "B           %8.4f + %8.4fi, abs_error %.2e, rel_error %.2e\n",
                                       ii, jj,
                                       real(Aij(it, jt)), imag(Aij(it, jt)),
                                       real(B[ ii + jj*ldb ]), imag(B[ ii + jj*ldb ]),
                                       abs_error, rel_error);
                            }

                            test_assert(abs_error <= abs_tol || rel_error <= rel_tol);
                        }
                    }
                }
            }
            else {
                ii += ib;
                jj += jb;
            }
        }
    }
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_gemm(slate::Target target)
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + ", " + target_name(target) + " >");
    Test name(msg.c_str());

    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    const blas::Layout layout = blas::Layout::ColMajor;
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int nb = 16;
    int m = 80;
    int n = 64;
    // todo: k < nb causes error in creating tiles
    int k = 16;  // block col * block row update
    assert(k <= nb);
    int p = 1;
    int q = 1;

    scalar_t alpha, beta;
    lapack::larnv(1, iseed, 1, &alpha);
    lapack::larnv(1, iseed, 1, &beta);
    if (g_verbose) {
        printf("alpha = %.4f + %.4fi;\n"
               "beta  = %.4f + %.4fi;\n",
               real(alpha), imag(alpha),
               real(beta),  imag(beta));
    }

    // test all combinations of op(C), op(B), op(A)
    for (int ic = 0; ic < 3; ++ic) {
    for (int ib = 0; ib < 3; ++ib) {
    for (int ia = 0; ia < 3; ++ia) {
        // setup C such that op(C) is m-by-n
        int Cm = (ic == 0 ? m : n);
        int Cn = (ic == 0 ? n : m);
        int ldc = Cm + 1;
        std::vector<scalar_t> Cdata(ldc*Cn);
        lapack::larnv(1, iseed, Cdata.size(), Cdata.data());
        auto C = slate::Matrix<scalar_t>::fromLAPACK(Cm, Cn, Cdata.data(), ldc, nb, p, q, g_mpi_comm);
        if (ic == 1)
            C = transpose(C);
        else if (ic == 2)
            C = conjTranspose(C);
        assert(C.mt() == slate::ceildiv(m, nb));
        assert(C.nt() == slate::ceildiv(n, nb));
        if (target == slate::Target::Devices)
            C.allocateBatchArrays();

        // opCref = op(C) is m-by-n
        int ldopc = m + 1;
        std::vector<scalar_t> opCref(ldopc*n);
        copy(C, opCref.data(), ldopc);

        // setup B such that op(B) is k-by-n
        int Bm = (ib == 0 ? k : n);
        int Bn = (ib == 0 ? n : k);
        int ldb = Bm + 1;
        std::vector<scalar_t> Bdata(ldb*Bn);
        lapack::larnv(1, iseed, Bdata.size(), Bdata.data());
        auto B = slate::Matrix<scalar_t>::fromLAPACK(Bm, Bn, Bdata.data(), ldb, nb, p, q, g_mpi_comm);
        if (ib == 1)
            B = transpose(B);
        else if (ib == 2)
            B = conjTranspose(B);
        assert(B.mt() == slate::ceildiv(k, nb));
        assert(B.nt() == slate::ceildiv(n, nb));

        // setup A such that op(A) is m-by-k
        int Am = (ia == 0 ? m : k);
        int An = (ia == 0 ? k : m);
        int lda = Am + 1;
        std::vector<scalar_t> Adata(lda*An);
        lapack::larnv(1, iseed, Adata.size(), Adata.data());
        auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, Adata.data(), lda, nb, p, q, g_mpi_comm);
        if (ia == 1)
            A = transpose(A);
        else if (ia == 2)
            A = conjTranspose(A);
        assert(A.mt() == slate::ceildiv(m, nb));
        assert(A.nt() == slate::ceildiv(k, nb));

        test_message("gemm( opA=%c, opB=%c, opC=%c )",
                     char(A.op()), char(B.op()), char(C.op()));

        if (g_verbose) {
            print("A",    A);
            print("Aref", Am, An, Adata.data(), lda);
            print( "B",    B );
            print( "Bref", Bm, Bn, Bdata.data(), ldb );
            print( "C",    C );
            print( "Cref", m, n, opCref.data(), ldopc );
        }

        // run test
        try {
            // expects tmp rvalues, so use std::move
            switch (target) {
                case slate::Target::Host:
                    assert(false);
                    break;
                case slate::Target::HostTask:
                    slate::internal::gemm<slate::Target::HostTask>(
                            alpha, std::move(A), std::move(B),
                            beta,  std::move(C), layout);
                    break;
                case slate::Target::HostNest:
                    slate::internal::gemm<slate::Target::HostNest>(
                            alpha, std::move(A), std::move(B),
                            beta,  std::move(C), layout);
                    break;
                case slate::Target::HostBatch:
                    slate::internal::gemm<slate::Target::HostBatch>(
                            alpha, std::move(A), std::move(B),
                            beta,  std::move(C), layout);
                    break;
                case slate::Target::Devices:
                    slate::internal::gemm<slate::Target::Devices>(
                            alpha, std::move(A), std::move(B),
                            beta,  std::move(C), layout);
                    // move data back to host
                    for (int j = 0; j < C.nt(); ++j)
                        for (int i = 0; i < C.mt(); ++i)
                            C.tileGetForReading(i, j, LayoutConvert(layout));
                    break;
            }

            // It should throw error if and only if
            // C is complex and
            // ((C is transposed and either A or B is conj-transposed) or
            //  (C is conj-transposed and either A or B is tranpsosed)).
            assert(! (slate::is_complex<scalar_t>::value &&
                      ((ic == 1 && (ia == 2 || ib == 2)) ||
                       (ic == 2 && (ia == 1 || ib == 1)))));
        }
        catch (std::exception& e) {
            printf("%%      not allowed\n");
            assert(slate::is_complex<scalar_t>::value &&
                   ((ic == 1 && (ia == 2 || ib == 2)) ||
                    (ic == 2 && (ia == 1 || ib == 1))));
            continue;
        }

        // reference solution
        blas::gemm(blas::Layout::ColMajor, A.op(), B.op(), m, n, k,
                   alpha, Adata.data(), lda,
                          Bdata.data(), ldb,
                   beta, opCref.data(), ldopc);

        if (g_verbose) {
            print("Chat",     C);
            print("Chat_ref", m, n, opCref.data(), ldopc);
        }

        real_t eps = std::numeric_limits<real_t>::epsilon();
        test_assert_equal(C, opCref.data(), ldopc, 3*sqrt(k)*eps, 3*sqrt(k)*eps);
    }}}
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_syrk(slate::Target target)
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + ", " + target_name(target) + " >");
    Test name(msg.c_str());

    using blas::Uplo;
    using blas::Op;
    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int nb = 4;
    int n = 12;
    int k = 4;
    assert( k <= nb );
    int p = 1;
    int q = 1;

    scalar_t alpha, beta;
    lapack::larnv(1, iseed, 1, &alpha);
    lapack::larnv(1, iseed, 1, &beta);
    if (g_verbose) {
        printf("alpha = %.4f + %.4fi;\n"
               "beta  = %.4f + %.4fi;\n",
               real(alpha), imag(alpha),
               real(beta),  imag(beta));
    }

    // test all combinations of op(C), op(A), uplo
    for (int ic = 0; ic < 3; ++ic) {
    for (int ia = 0; ia < 3; ++ia) {
    for (int iu = 0; iu < 2; ++iu) {
        Uplo uplo = uplos[iu];

        // setup C such that op(C) is n-by-n
        int ldc = n + 1;
        std::vector<scalar_t> Cdata(ldc*n);
        lapack::larnv(1, iseed, Cdata.size(), Cdata.data());
        auto C = slate::SymmetricMatrix<scalar_t>::fromLAPACK(uplo, n, Cdata.data(), ldc, nb, p, q, g_mpi_comm);
        if (ic == 1)
            C = transpose(C);
        else if (ic == 2)
            C = conjTranspose(C);
        assert(C.mt() == slate::ceildiv(n, nb));
        assert(C.nt() == slate::ceildiv(n, nb));
        if (target == slate::Target::Devices)
            C.allocateBatchArrays();

        // set unused data to nan
        scalar_t nan_ = nan("");
        if (uplo == blas::Uplo::Lower) {
            lapack::laset(lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                          &Cdata[ 0 + 1*ldc ], ldc);
        }
        else {
            lapack::laset(lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                          &Cdata[ 1 + 0*ldc ], ldc);
        }

        // opCref = op(C) is n-by-n
        // currently, C(i,j) does transpose but not conj.
        std::vector<scalar_t> opCref(ldc*n);
        copy(C, opCref.data(), ldc);

        // setup A such that op(A) is n-by-k
        int Am = (ia == 0 ? n : k);
        int An = (ia == 0 ? k : n);
        int lda = Am + 1;
        std::vector<scalar_t> Adata(lda*An);
        lapack::larnv(1, iseed, Adata.size(), Adata.data());
        auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, Adata.data(), lda, nb, p, q, g_mpi_comm);
        if (ia == 1)
            A = transpose(A);
        else if (ia == 2)
            A = conjTranspose(A);
        assert(A.mt() == slate::ceildiv(n, nb));
        assert(A.nt() == slate::ceildiv(k, nb));

        test_message("syrk( uplo=%c, opA=%c, opC=%c )",
                     char(C.uplo()), char(A.op()), char(C.op()));

        if (g_verbose) {
            print("A",    A);
            print("Aref", Am, An, Adata.data(), lda);
            print("C",    C);
            print("Cref", n, n, opCref.data(), ldc);
        }

        // run test
        try {
            //if (C.op() == Op::ConjTrans)  // TODO
            //    conjugate( C );
            // expects tmp rvalues, so use std::move
            switch (target) {
                case slate::Target::Host:
                    assert(false);
                    break;
                case slate::Target::HostTask:
                    slate::internal::syrk<slate::Target::HostTask>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    break;
                case slate::Target::HostNest:
                    slate::internal::syrk<slate::Target::HostNest>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    break;
                case slate::Target::HostBatch:
                    slate::internal::syrk<slate::Target::HostBatch>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    break;
                case slate::Target::Devices:
                    slate::internal::syrk<slate::Target::Devices>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    // move data back to host
                    // assume Lower, NoTrans or Upper, Trans
                    for (int j = 0; j < C.nt(); ++j)
                        for (int i = j; i < C.mt(); ++i)  // lower
                            C.tileGetForReading(i, j, slate::LayoutConvert::ColMajor);
                    break;
            }
            //if (C.op() == Op::ConjTrans)  // TODO
            //    conjugate( C );

            // It should throw error if and only if
            // C is complex and
            // C is conj-transposed or A is conj-transposed.
            // Also, only Lower, NoTrans or Upper, Trans/ConjTrans.
            assert( (C.uploLogical() == Uplo::Lower) &&
                    ! (C.is_complex && (ic == 2 || ia == 2)) );
        }
        catch (std::exception& e) {
            printf("%%      not allowed\n");
            assert(! ( (C.uploLogical() == Uplo::Lower) &&
                       ! (C.is_complex && (ic == 2 || ia == 2))));
            continue;
        }

        // reference solution
        // transpose flips uplo
        Uplo op_uplo = uplo;
        if (C.op() != Op::NoTrans) {
            op_uplo = (op_uplo == Uplo::Lower ? Uplo::Upper
                                              : Uplo::Lower);
        }
        blas::syrk(blas::Layout::ColMajor, op_uplo, A.op(), n, k,
                   alpha, Adata.data(), lda,
                   beta, opCref.data(), ldc);

        if (g_verbose) {
            print("Chat",     C);
            print("Chat_ref", n, n, opCref.data(), ldc);
        }

        real_t eps = std::numeric_limits<real_t>::epsilon();
        test_assert_equal(C, opCref.data(), ldc, 3*sqrt(k)*eps, 3*sqrt(k)*eps);
    }}}
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_herk(slate::Target target)
{
    auto msg = __func__ + ("< " + type_name<scalar_t>() + ", " + target_name(target) + " >");
    Test name(msg.c_str());

    using blas::Uplo;
    using blas::Op;
    using blas::real;
    using blas::imag;
    using blas::conj;
    using real_t = blas::real_type<scalar_t>;
    int64_t iseed[4] = { 0, 1, 2, 3 };

    int nb = 4;
    int n = 12;
    int k = 4;
    assert( k <= nb );
    int p = 1;
    int q = 1;

    real_t alpha, beta;
    lapack::larnv(1, iseed, 1, &alpha);
    lapack::larnv(1, iseed, 1, &beta);
    if (g_verbose) {
        printf("alpha = %.4f;\n"
               "beta  = %.4f;\n",
               alpha, beta);
    }

    // test all combinations of op(C), op(A), uplo
    for (int ic = 0; ic < 3; ++ic) {
    for (int ia = 0; ia < 3; ++ia) {
    for (int iu = 0; iu < 2; ++iu) {
        Uplo uplo = uplos[iu];

        // setup C such that op(C) is n-by-n
        int ldc = n + 1;
        std::vector<scalar_t> Cdata(ldc*n);
        lapack::larnv(1, iseed, Cdata.size(), Cdata.data());
        auto C = slate::HermitianMatrix<scalar_t>::fromLAPACK(uplo, n, Cdata.data(), ldc, nb, p, q, g_mpi_comm);
        if (ic == 1)
            C = transpose(C);
        else if (ic == 2)
            C = conjTranspose(C);
        assert(C.mt() == slate::ceildiv(n, nb));
        assert(C.nt() == slate::ceildiv(n, nb));
        if (target == slate::Target::Devices)
            C.allocateBatchArrays();

        // set unused data to nan
        scalar_t nan_ = nan("");
        if (uplo == blas::Uplo::Lower) {
            lapack::laset(lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                          &Cdata[ 0 + 1*ldc ], ldc);
        }
        else {
            lapack::laset(lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                          &Cdata[ 1 + 0*ldc ], ldc);
        }

        // opCref = op(C) is n-by-n
        // currently, C(i,j) does transpose but not conj.
        std::vector< scalar_t > opCref(ldc*n);
        copy(C, opCref.data(), ldc);

        // setup A such that op(A) is n-by-k
        int Am = (ia == 0 ? n : k);
        int An = (ia == 0 ? k : n);
        int lda = Am + 1;
        std::vector<scalar_t> Adata(lda*An);
        lapack::larnv(1, iseed, Adata.size(), Adata.data());
        auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, Adata.data(), lda, nb, p, q, g_mpi_comm);
        if (ia == 1)
            A = transpose(A);
        else if (ia == 2)
            A = conjTranspose(A);
        assert(A.mt() == slate::ceildiv(n, nb));
        assert(A.nt() == slate::ceildiv(k, nb));

        test_message("herk( uplo=%c, opA=%c, opC=%c )",
                     char(C.uplo()), char(A.op()), char(C.op()));

        if (g_verbose) {
            print("A",    A);
            print("Aref", Am, An, Adata.data(), lda);
            print("C",    C);
            print("Cref", n, n, opCref.data(), ldc);
        }

        // run test
        try {
            //if (C.op() == Op::ConjTrans)  // TODO
            //    conjugate( C );
            // expects tmp rvalues, so use std::move
            switch (target) {
                case slate::Target::Host:
                    assert(false);
                    break;
                case slate::Target::HostTask:
                    slate::internal::herk<slate::Target::HostTask>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    break;
                case slate::Target::HostNest:
                    slate::internal::herk<slate::Target::HostNest>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    break;
                case slate::Target::HostBatch:
                    slate::internal::herk<slate::Target::HostBatch>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    break;
                case slate::Target::Devices:
                    slate::internal::herk<slate::Target::Devices>(
                            alpha, std::move(A),
                            beta,  std::move(C));
                    // move data back to host
                    // assume Lower, NoTrans or Upper, Trans
                    for (int j = 0; j < C.nt(); ++j)
                        for (int i = j; i < C.mt(); ++i)  // lower
                            C.tileGetForReading(i, j, slate::LayoutConvert::ColMajor);
                    break;
            }
            //if (C.op() == Op::ConjTrans)  // TODO
            //    conjugate( C );

            // It should throw error if and only if
            // C is complex and
            // C is conj-transposed or A is conj-transposed.
            // Also, only Lower, NoTrans or Upper, Trans/ConjTrans.
            assert((C.uploLogical() == Uplo::Lower) &&
                   ! (C.is_complex && (ic == 1 || ia == 1)));
        }
        catch (std::exception& e) {
            printf("%%      not allowed\n");
            assert(! ((C.uploLogical() == Uplo::Lower) &&
                      ! (C.is_complex && (ic == 1 || ia == 1))));
            continue;
        }

        // reference solution
        // transpose flips uplo
        Uplo op_uplo = uplo;
        if (C.op() != Op::NoTrans) {
            op_uplo = (op_uplo == Uplo::Lower ? Uplo::Upper
                                              : Uplo::Lower);
        }
        blas::herk(blas::Layout::ColMajor, op_uplo, A.op(), n, k,
                   alpha, Adata.data(), lda,
                   beta, opCref.data(), ldc);

        if (g_verbose) {
            print("Chat",     C);
            print("Chat_ref", n, n, opCref.data(), ldc);
        }

        real_t eps = std::numeric_limits<real_t>::epsilon();
        test_assert_equal(C, opCref.data(), ldc, 3*sqrt(k)*eps, 3*sqrt(k)*eps);
    }}}
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    g_mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(g_mpi_comm, &g_mpi_rank);
    printf("num_devices %d\n", g_num_devices);

    bool do_all   = true;
    bool do_gemm  = false;
    bool do_syrk  = false;
    bool do_herk  = false;

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
    }

    // run tests
    if (do_all || do_gemm) {
        for (int it = 0; it < 4; ++it) {
            test_gemm<double>(targets[it]);
            test_gemm< std::complex<double> >(targets[it]);
        }
    }
    if (do_all || do_syrk) {
        for (int it = 0; it < 4; ++it) {
            test_syrk<double>(targets[it]);
            test_syrk< std::complex<double> >(targets[it]);
        }
    }
    if (do_all || do_herk) {
        for (int it = 0; it < 4; ++it) {
            test_herk<double>(targets[it]);
            test_herk< std::complex<double> >(targets[it]);
        }
    }

    MPI_Finalize();
    return 0;
}
