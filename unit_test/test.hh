// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// todo: both unit_test.hh and test.hh exist. Rename one of them?
#ifndef SLATE_UNIT_TEST_TEST_HH
#define SLATE_UNIT_TEST_TEST_HH

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

#include "slate/Matrix.hh"
#include "slate/BaseTrapezoidMatrix.hh"

#include <iostream>
#include <iomanip>

//------------------------------------------------------------------------------
using llong = long long;

// -----------------------------------------------------------------------------
// global variables
MPI_Comm g_mpi_comm;
int  g_mpi_rank    = -1;
int  g_mpi_size    = -1;
bool g_verbose     = false;
int  g_num_devices = omp_get_num_devices();

// -----------------------------------------------------------------------------
// type_name<T>() returns string describing type of T.
// see https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c

// for demangling on non-Microsoft platforms
#ifndef _MSC_VER
    #include <cxxabi.h>
#endif

template<typename T>
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;

    std::unique_ptr< char, void(*)(void*) > own(
        #ifndef _MSC_VER
            abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
        #else
            nullptr,
        #endif
        std::free
    );

    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

// -----------------------------------------------------------------------------
// Prints test name at start and end of test.
// Since destructors are called in reverse order of constructors,
// putting this first ensures that its destructor is called last when exiting
// the test function.
class Test {
public:
    // ----------------------------------------
    Test(const char* msg)
        : msg_(msg)
    {
        if (g_mpi_rank == 0) {
            std::cout << "%---------------------------------------- "
                      << msg_ << "\n" << std::flush;
        }
        MPI_Barrier(g_mpi_comm);
    }

    // ----------------------------------------
    ~Test()
    {
        std::cout << std::flush;
        MPI_Barrier(g_mpi_comm);

        if (g_mpi_rank == 0) {
            std::cout << "%---------------------------------------- "
                      << msg_ << " done\n\n" << std::flush;
        }
        MPI_Barrier(g_mpi_comm);
    }

    const char* msg_;
};

// -----------------------------------------------------------------------------
// Does barrier, then prints label on rank 0 for next sub-test.
void test_message(const char* format, ...)
{
    MPI_Barrier(g_mpi_comm);

    va_list ap;
    va_start(ap, format);
    char buf[ 1024 ];
    vsnprintf(buf, sizeof(buf), format, ap);
    va_end(ap);

    if (g_mpi_rank == 0) {
        std::cout << "%----- " << buf << "\n";
    }
}

// -----------------------------------------------------------------------------
// suppresses compiler warning if var is unused
#define unused( var ) \
    ((void)var)

// -----------------------------------------------------------------------------
// similar to assert(), but also prints out MPI rank.
#define test_assert( cond ) \
    do { \
        if (! (cond)) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": " \
                      << #cond << "\n"; \
            exit(1); \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// executes expr; asserts that the given exception was thrown.
#define test_assert_throw( expr, exception ) \
    do { \
        try { \
            expr; \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": did not throw expected exception\n"; \
        } \
        catch( exception& e ) {} \
        catch( ... ) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": wrong exception thrown\n"; \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// executes expr; asserts that no exception was thrown.
#define test_assert_no_throw( expr ) \
    do { \
        try { \
            expr; \
        } \
        catch( ... ) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": unexpected exception thrown\n"; \
        } \
    } while(0)

//------------------------------------------------------------------------------
// print Tile
template <typename scalar_t>
void print(const char* name, slate::Tile<scalar_t>& A)
{
    printf("%s = [\n", name);
    for (int i = 0; i < A.mb(); ++i) {
        for (int64_t j = 0; j < A.nb(); ++j) {
            printf(" %9.4f", A(i, j));
        }
        printf("\n");
    }
    printf("];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()));
}

//------------------------------------------------------------------------------
// print Tile, complex specialization
template <typename scalar_t>
void print(const char* name, slate::Tile< std::complex<scalar_t> >& A)
{
    printf("%s = [\n", name);
    for (int i = 0; i < A.mb(); ++i) {
        for (int64_t j = 0; j < A.nb(); ++j) {
            printf(" %9.4f + %9.4fi", real(A(i, j)), imag(A(i, j)));
        }
        printf("\n");
    }
    printf("];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()));
}

//------------------------------------------------------------------------------
// print Matrix
template <typename scalar_t>
void print(const char* name, slate::Matrix<scalar_t>& A)
{
    using blas::real;

    printf("%s = [  %% op=%c\n", name, char(A.op()));
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {
        int64_t ib = A.tileMb(i);

        // loop over block cols, print out address
        bool row_is_local = false;
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t jb = A.tileNb(j);

            if (j > 0)
                printf("    ");
            else
                printf("%%   ");
            if (A.tileIsLocal(i, j)) {
                row_is_local = true;
                auto Aij = A(i, j);
                printf("   %-18p %2lld by %2lld", (void*) Aij.data(), llong(ib), llong(jb));
                for (int64_t jt = 3; jt < jb; ++jt) { // above pointer is 3 columns
                    printf(" %9s", "");
                }
            }
            else {
                printf(" [...] ");
            }
        }
        printf("\n");

        if (row_is_local) {
            for (int64_t it = 0; it < ib; ++it) {
                // loop over block cols, then cols within block col
                for (int64_t j = 0; j < A.nt(); ++j) {
                    int64_t jb = A.tileNb(j);

                    printf("    ");
                    if (A.tileIsLocal(i, j)) {
                        auto Aij = A(i, j);
                        for (int64_t jt = 0; jt < jb; ++jt) {
                            printf(" %9.4f", real(Aij(it, jt)));
                        }
                    }
                    else {
                        printf(" [...] ");
                    }
                }
                printf("\n");
            }
        }
        if (i < A.mt()-1) {
            printf("\n");
        }
    }
    printf("];\n");
}

//------------------------------------------------------------------------------
// print Matrix, complex specialization
template <typename scalar_t>
void print(const char* name, slate::Matrix< std::complex<scalar_t> >& A)
{
    using blas::real;

    printf("%s = [  %% op=%c\n", name, char(A.op()));
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {
        int64_t ib = A.tileMb(i);

        // loop over block cols, print out address
        bool row_is_local = false;
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t jb = A.tileNb(j);

            if (j > 0)
                printf("    ");
            else
                printf("%%   ");
            if (A.tileIsLocal(i, j)) {
                row_is_local = true;
                auto Aij = A(i, j);
                printf("  %-21p", (void*) Aij.data() );
                for (int64_t jt = 1; jt < jb; ++jt) { // above pointer is 1 column
                    printf(" %22s", "");
                }
            }
            else {
                printf(" [...] ");
            }
        }
        printf("\n");

        if (row_is_local) {
            for (int64_t it = 0; it < ib; ++it) {
                // loop over block cols, then cols within block col
                for (int64_t j = 0; j < A.nt(); ++j) {
                    int64_t jb = A.tileNb(j);

                    printf("    ");
                    if (A.tileIsLocal(i, j)) {
                        auto Aij = A(i, j);
                        for (int64_t jt = 0; jt < jb; ++jt) {
                            printf(" %9.4f + %9.4fi", real(Aij(it, jt)), imag(Aij(it, jt)));
                        }
                    }
                    else {
                        printf(" [...] ");
                    }
                }
                printf("\n");
            }
        }
        if (i < A.mt()-1) {
            printf("\n");
        }
    }
    printf("];\n");
}

//------------------------------------------------------------------------------
// print Trapezoid matrix
template <typename scalar_t>
void print(const char* name, slate::BaseTrapezoidMatrix<scalar_t>& A)
{
    bool lower =
        ((A.uplo() == blas::Uplo::Lower && A.op() == blas::Op::NoTrans) ||
         (A.uplo() == blas::Uplo::Upper && A.op() != blas::Op::NoTrans));
    bool upper = ! lower;

    printf("%s = [  %% op=%c, uplo=%c\n", name, char(A.op()), char(A.uplo()));
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        bool row_is_local = false;
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t jb = A.tileNb(j);

            if (j > 0)
                printf("    ");
            else
                printf("%%   ");
            if (A.tileIsLocal(i, j)) {
                row_is_local = true;
                if ((lower && i >= j) || (upper && i <= j)) {
                    auto Aij = A(i, j);
                    printf("  %-18p", (void*) Aij.data());
                    for (int64_t jt = 2; jt < jb; ++jt) { // above pointer is 2 columns
                        printf(" %9s", "");
                    }
                }
                else {
                    for (int64_t jt = 0; jt < jb; ++jt) {
                        printf(" %9s", "");
                    }
                }
            }
            else {
                printf(" [...] ");
            }
        }
        printf("\n");

        if (row_is_local) {
            int64_t ib = A.tileMb(i);
            for (int64_t it = 0; it < ib; ++it) {

                // loop over block cols, then cols within block col
                for (int64_t j = 0; j < A.nt(); ++j) {
                    int64_t jb = A.tileNb(j);

                    printf("    ");
                    if (A.tileIsLocal(i, j)) {
                        if ((lower && i >= j) || (upper && i <= j)) {
                            auto Aij = A(i, j);
                            for (int64_t jt = 0; jt < jb; ++jt) {
                                printf(" %9.4f", Aij(it, jt));
                            }
                        }
                        else {
                            for (int64_t jt = 0; jt < jb; ++jt) {
                                printf(" %9s", "---");
                            }
                        }
                    }
                    else {
                        printf(" [...] ");
                    }
                }
                printf("\n");
            }
        }
        if (i < A.mt()-1) {
            printf("\n");
        }
    }
    printf("];\n");

    // symmetrize in Matlab
    if (A.uplo() == blas::Uplo::Lower)
        printf("%s = tril(%s) + tril(%s, -1)';\n", name, name, name);
    else
        printf("%s = triu(%s) + triu(%s,  1)';\n", name, name, name);
}

//------------------------------------------------------------------------------
// print Trapezoid matrix, complex specialization
template <typename scalar_t>
void print(const char* name, slate::BaseTrapezoidMatrix< std::complex<scalar_t> >& A)
{
    bool lower =
        ((A.uplo() == blas::Uplo::Lower && A.op() == blas::Op::NoTrans) ||
         (A.uplo() == blas::Uplo::Upper && A.op() != blas::Op::NoTrans));
    bool upper = ! lower;

    printf("%s = [  %% op=%c, uplo=%c\n", name, char(A.op()), char(A.uplo()));
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        bool row_is_local = false;
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t jb = A.tileNb(j);

            if (j > 0)
                printf("    ");
            else
                printf("%%   ");
            if (A.tileIsLocal(i, j)) {
                row_is_local = true;
                if ((lower && i >= j) || (upper && i <= j)) {
                    auto Aij = A(i, j);
                    printf("  %-21p", (void*) Aij.data());
                    for (int64_t jt = 1; jt < jb; ++jt) { // above pointer is 1 column
                        printf(" %22s", "");
                    }
                }
                else {
                    for (int64_t jt = 0; jt < jb; ++jt) {
                        printf(" %22s", "");
                    }
                }
            }
            else {
                printf(" [...] ");
            }
        }
        printf("\n");

        if (row_is_local) {
            int64_t ib = A.tileMb(i);
            for (int64_t it = 0; it < ib; ++it) {

                // loop over block cols, then cols within block col
                for (int64_t j = 0; j < A.nt(); ++j) {
                    int64_t jb = A.tileNb(j);

                    printf("    ");
                    if (A.tileIsLocal( i, j )) {
                        if ((lower && i >= j) || (upper && i <= j)) {
                            auto Aij = A(i, j);
                            for (int64_t jt = 0; jt < jb; ++jt) {
                                printf(" %9.4f + %9.4fi", real(Aij(it, jt)), imag(Aij(it, jt)));
                            }
                        }
                        else {
                            for (int64_t jt = 0; jt < jb; ++jt) {
                                printf(" %22s", "---");
                            }
                        }
                    }
                    else {
                        printf(" [...] ");
                    }
                }
                printf("\n");
            }
        }
        if (i < A.mt()-1) {
            printf("\n");
        }
    }
    printf("];\n");

    // symmetrize in Matlab
    if (A.uplo() == blas::Uplo::Lower)
        printf("%s = tril(%s) + tril(%s, -1)';\n", name, name, name);
    else
        printf("%s = triu(%s) + triu(%s,  1)';\n", name, name, name);
}

//------------------------------------------------------------------------------
// print LAPACK-style matrix
template <typename scalar_t>
void print(const char* name, int64_t m, int64_t n, scalar_t* A, int64_t lda)
{
    printf( "%s = [\n", name );
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf(" %9.4f", A[ i + j*lda ]);
        }
        printf("\n");
    }
    printf("];\n");
}

//------------------------------------------------------------------------------
// print LAPACK-style matrix, complex specialization
template <typename scalar_t>
void print(const char* name, int64_t m, int64_t n, std::complex<scalar_t>* A, int64_t lda)
{
    printf( "%s = [\n", name );
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf(" %9.4f + %9.4fi", real(A[ i + j*lda ]), imag(A[ i + j*lda ]));
        }
        printf("\n");
    }
    printf("];\n");
}

#endif // SLATE_UNIT_TEST_TEST_HH
