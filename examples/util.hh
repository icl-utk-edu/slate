#ifndef UTIL_H
#define UTIL_H

#include <blas.hh>

#include <stdio.h>
#include <math.h>

//------------------------------------------------------------------------------
void print_func_( int rank, const char* func )
{
    if (rank == 0)
        printf( "rank %d: %s\n", rank, func );
}

#ifdef __GNUC__
    #define print_func( rank ) print_func_( rank, __PRETTY_FUNCTION__ )
#else
    #define print_func( rank ) print_func_( rank, __func__ )
#endif

//------------------------------------------------------------------------------
// utility to create real or complex number
template <typename scalar_type>
scalar_type make( blas::real_type<scalar_type> re,
                  blas::real_type<scalar_type> im )
{
    return re;
}

template <typename T>
std::complex<T> make( T re, T im )
{
    return std::complex<T>( re, im );
}

//------------------------------------------------------------------------------
// generate random matrix A
template <typename scalar_type>
void random_matrix( int64_t m, int64_t n, scalar_type* A, int64_t lda )
{
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            A[ i + j*lda ] = make<scalar_type>( rand() / double(RAND_MAX),
                                                rand() / double(RAND_MAX) );
        }
    }
}

//------------------------------------------------------------------------------
// generate random, diagonally dominant matrix A
template <typename scalar_type>
void random_matrix_diag_dominant( int64_t m, int64_t n, scalar_type* A, int64_t lda )
{
    using blas::real;
    int64_t max_mn = std::max( m, n );
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            A[ i + j*lda ] = make<scalar_type>( rand() / double(RAND_MAX),
                                                rand() / double(RAND_MAX) );
        }
        if (j < m) {
            // make diagonal real & dominant
            A[ j + j*lda ] = real( A[ j + j*lda ] ) + max_mn;
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
template <typename matrix_type>
void random_matrix( matrix_type& A )
{
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                try {
                    auto T = A( i, j );
                    random_matrix( T.mb(), T.nb(), T.data(), T.stride() );
                }
                catch (...) {
                    // ignore missing tiles
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random, diagonally dominant matrix A
template <typename matrix_type>
void random_matrix_diag_dominant( matrix_type& A )
{
    using blas::real;
    int64_t max_mn = std::max( A.m(), A.n() );
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                try {
                    auto T = A( i, j );
                    random_matrix( T.mb(), T.nb(), T.data(), T.stride() );
                    if (i == j) {
                        // assuming tileMb == tileNb, then i == j are diagonal tiles
                        // make diagonal real & dominant
                        int64_t min_mb_nb = std::min( T.mb(), T.nb() );
                        for (int64_t ii = 0; ii < min_mb_nb; ++ii) {
                            T.at(ii, ii) = real( T.at(ii, ii) ) + max_mn;
                        }
                    }
                }
                catch (...) {
                    // ignore missing tiles
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void print_matrix( const char* label, int m, int n, scalar_type* A, int lda )
{
    using blas::real;
    using blas::imag;
    printf( "%s = [\n", label );
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (blas::is_complex<scalar_type>::value) {
                printf( "  %7.4f + %7.4fi", real(A[ i + j*lda ]), imag(A[ i + j*lda ]) );
            }
            else {
                printf( "  %7.4f", real(A[ i + j*lda ]) );
            }
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

//------------------------------------------------------------------------------
/// Determine a square or short-wide p-by-q grid that is as square as
/// possible to fit the MPI size. Worst case is p=1, q=mpi_size.
///
void grid_size( int mpi_size, int* p_out, int* q_out )
{
    int p, q;
    for (p = int( sqrt( mpi_size ) ); p > 0; --p) {
        q = int( mpi_size / p );
        if (p*q == mpi_size)
            break;
    }
    assert( p*q == mpi_size );
    *p_out = p;
    *q_out = q;
}

//------------------------------------------------------------------------------
/// Determine a square p-by-p grid to fit the MPI size.
///
void grid_size_square( int mpi_size, int* p_out, int* q_out )
{
    *p_out = int( sqrt( mpi_size ) );
    *q_out = *p_out;
}

//------------------------------------------------------------------------------
// suppress compiler "unused" warning for variable x
#define unused( x ) ((void) x)

//------------------------------------------------------------------------------
// Parse command line options:
// s = single,         sets types[ 0 ]
// d = double,         sets types[ 1 ]
// c = complex,        sets types[ 2 ]
// z = double-complex, sets types[ 3 ]
// If no options, sets all types to true.
// Throws error for unknown options.
void parse_args( int argc, char** argv, bool types[ 4 ] )
{
    if (argc == 1) {
        types[ 0 ] = types[ 1 ] = types[ 2 ] = types[ 3 ] = true;
    }
    else {
        types[ 0 ] = types[ 1 ] = types[ 2 ] = types[ 3 ] = false;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[ i ];
        if (arg == "s")
            types[ 0 ] = true;
        else if (arg == "d")
            types[ 1 ] = true;
        else if (arg == "c")
            types[ 2 ] = true;
        else if (arg == "z")
            types[ 3 ] = true;
        else {
            throw std::runtime_error(
                "unknown option: \"" + arg + "\"\n"
                + "Usage: " + argv[ 0 ] + " [s] [d] [c] [z]\n"
                + "for single, double, complex, double-complex.\n" );
        }
    }
}

#endif // UTIL_H
