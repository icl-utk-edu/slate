#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>

// macOS provides CMPLX only for clang, oddly.
#ifndef CMPLX
    #define CMPLX( re, im ) (re + im*_Complex_I)
#endif
#ifndef CMPLXF
    #define CMPLXF( re, im ) (re + im*_Complex_I)
#endif

#define MAX( x, y ) ( ( ( x ) > ( y ) ) ? ( x ) : ( y ) )
#define MIN( x, y ) ( ( ( x ) < ( y ) ) ? ( x ) : ( y ) )

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
// generate random matrix A
void random_Tile_r32( slate_Tile_r32 T )
{
    int64_t m   = slate_Tile_mb_r32( T );
    int64_t n   = slate_Tile_nb_r32( T );
    int64_t lda = slate_Tile_stride_r32( T );
    float*  A   = slate_Tile_data_r32( T );
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            A[ i + j*lda ] = rand() / (float) RAND_MAX;
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_Tile_r64( slate_Tile_r64 T )
{
    int64_t m   = slate_Tile_mb_r64( T );
    int64_t n   = slate_Tile_nb_r64( T );
    int64_t lda = slate_Tile_stride_r64( T );
    double* A   = slate_Tile_data_r64( T );
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            A[ i + j*lda ] = rand() / (double) RAND_MAX;
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_Tile_c32( slate_Tile_c32 T )
{
    int64_t m   = slate_Tile_mb_c32( T );
    int64_t n   = slate_Tile_nb_c32( T );
    int64_t lda = slate_Tile_stride_c32( T );
    float _Complex*  A = slate_Tile_data_c32( T );
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            float complex z = CMPLXF( rand() / (float) RAND_MAX,
                                      rand() / (float) RAND_MAX );
            A[ i + j*lda ] = z;
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_Tile_c64( slate_Tile_c64 T )
{
    int64_t m   = slate_Tile_mb_c64( T );
    int64_t n   = slate_Tile_nb_c64( T );
    int64_t lda = slate_Tile_stride_c64( T );
    double _Complex* A = slate_Tile_data_c64( T );
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            double complex z = CMPLX( rand() / (double) RAND_MAX,
                                      rand() / (double) RAND_MAX );
            A[ i + j*lda ] = z;
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_Matrix_r32( slate_Matrix_r32 A )
{
    for (int64_t j = 0; j < slate_Matrix_nt_r32( A ); ++j) {
        for (int64_t i = 0; i < slate_Matrix_mt_r32( A ); ++i) {
            if (slate_Matrix_tileIsLocal_r32( A, i, j )) {
                slate_Tile_r32 T = slate_Matrix_at_r32( A, i, j );
                random_Tile_r32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_Matrix_r64( slate_Matrix_r64 A )
{
    for (int64_t j = 0; j < slate_Matrix_nt_r64( A ); ++j) {
        for (int64_t i = 0; i < slate_Matrix_mt_r64( A ); ++i) {
            if (slate_Matrix_tileIsLocal_r64( A, i, j )) {
                slate_Tile_r64 T = slate_Matrix_at_r64( A, i, j );
                random_Tile_r64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_Matrix_c32( slate_Matrix_c32 A )
{
    for (int64_t j = 0; j < slate_Matrix_nt_c32( A ); ++j) {
        for (int64_t i = 0; i < slate_Matrix_mt_c32( A ); ++i) {
            if (slate_Matrix_tileIsLocal_c32( A, i, j )) {
                slate_Tile_c32 T = slate_Matrix_at_c32( A, i, j );
                random_Tile_c32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_Matrix_c64( slate_Matrix_c64 A )
{
    for (int64_t j = 0; j < slate_Matrix_nt_c64( A ); ++j) {
        for (int64_t i = 0; i < slate_Matrix_mt_c64( A ); ++i) {
            if (slate_Matrix_tileIsLocal_c64( A, i, j )) {
                slate_Tile_c64 T = slate_Matrix_at_c64( A, i, j );
                random_Tile_c64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_SymmetricMatrix_r32(
    slate_SymmetricMatrix_r32 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_SymmetricMatrix_nt_r32( A ); ++j) {
        for (int64_t i = 0; i < slate_SymmetricMatrix_mt_r32( A ); ++i) {
            if (slate_SymmetricMatrix_tileIsLocal_r32( A, i, j )) {
                slate_Tile_r32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_SymmetricMatrix_at_r32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_SymmetricMatrix_at_r32(A, i, j);
                else
                    continue;
                random_Tile_r32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_SymmetricMatrix_r64(
    slate_SymmetricMatrix_r64 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_SymmetricMatrix_nt_r64( A ); ++j) {
        for (int64_t i = 0; i < slate_SymmetricMatrix_mt_r64( A ); ++i) {
            if (slate_SymmetricMatrix_tileIsLocal_r64( A, i, j )) {
                slate_Tile_r64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_SymmetricMatrix_at_r64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_SymmetricMatrix_at_r64(A, i, j);
                else
                    continue;
                random_Tile_r64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_SymmetricMatrix_c32(
    slate_SymmetricMatrix_c32 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_SymmetricMatrix_nt_c32( A ); ++j) {
        for (int64_t i = 0; i < slate_SymmetricMatrix_mt_c32( A ); ++i) {
            if (slate_SymmetricMatrix_tileIsLocal_c32( A, i, j )) {
                slate_Tile_c32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_SymmetricMatrix_at_c32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_SymmetricMatrix_at_c32(A, i, j);
                else
                    continue;
                random_Tile_c32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_SymmetricMatrix_c64(
    slate_SymmetricMatrix_c64 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_SymmetricMatrix_nt_c64( A ); ++j) {
        for (int64_t i = 0; i < slate_SymmetricMatrix_mt_c64( A ); ++i) {
            if (slate_SymmetricMatrix_tileIsLocal_c64( A, i, j )) {
                slate_Tile_c64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_SymmetricMatrix_at_c64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_SymmetricMatrix_at_c64(A, i, j);
                else
                    continue;
                random_Tile_c64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_HermitianMatrix_r32(
    slate_HermitianMatrix_r32 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_r32( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_r32( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_r32( A, i, j )) {
                slate_Tile_r32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_r32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_r32(A, i, j);
                else
                    continue;
                random_Tile_r32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_HermitianMatrix_r64(
    slate_HermitianMatrix_r64 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_r64( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_r64( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_r64( A, i, j )) {
                slate_Tile_r64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_r64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_r64(A, i, j);
                else
                    continue;
                random_Tile_r64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_HermitianMatrix_c32(
    slate_HermitianMatrix_c32 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_c32( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_c32( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_c32( A, i, j )) {
                slate_Tile_c32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_c32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_c32(A, i, j);
                else
                    continue;
                random_Tile_c32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_HermitianMatrix_c64(
    slate_HermitianMatrix_c64 A, slate_Uplo uplo )
{
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_c64( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_c64( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_c64( A, i, j )) {
                slate_Tile_c64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_c64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_c64(A, i, j);
                else
                    continue;
                random_Tile_c64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_TriangularMatrix_r32(
    slate_TriangularMatrix_r32 A, slate_Uplo uplo, slate_Diag diag )
{
    for (int64_t j = 0; j < slate_TriangularMatrix_nt_r32( A ); ++j) {
        for (int64_t i = 0; i < slate_TriangularMatrix_mt_r32( A ); ++i) {
            if (slate_TriangularMatrix_tileIsLocal_r32( A, i, j )) {
                slate_Tile_r32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_TriangularMatrix_at_r32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_TriangularMatrix_at_r32(A, i, j);
                else
                    continue;

                random_Tile_r32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_TriangularMatrix_r64(
    slate_TriangularMatrix_r64 A, slate_Uplo uplo, slate_Diag diag )
{
    for (int64_t j = 0; j < slate_TriangularMatrix_nt_r64( A ); ++j) {
        for (int64_t i = 0; i < slate_TriangularMatrix_mt_r64( A ); ++i) {
            if (slate_TriangularMatrix_tileIsLocal_r64( A, i, j )) {
                slate_Tile_r64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_TriangularMatrix_at_r64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_TriangularMatrix_at_r64(A, i, j);
                else
                    continue;

                random_Tile_r64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_TriangularMatrix_c32(
    slate_TriangularMatrix_c32 A, slate_Uplo uplo, slate_Diag diag )
{
    for (int64_t j = 0; j < slate_TriangularMatrix_nt_c32( A ); ++j) {
        for (int64_t i = 0; i < slate_TriangularMatrix_mt_c32( A ); ++i) {
            if (slate_TriangularMatrix_tileIsLocal_c32( A, i, j )) {
                slate_Tile_c32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_TriangularMatrix_at_c32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_TriangularMatrix_at_c32(A, i, j);
                else
                    continue;

                random_Tile_c32( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random matrix A
void random_TriangularMatrix_c64(
    slate_TriangularMatrix_c64 A, slate_Uplo uplo, slate_Diag diag )
{
    for (int64_t j = 0; j < slate_TriangularMatrix_nt_c64( A ); ++j) {
        for (int64_t i = 0; i < slate_TriangularMatrix_mt_c64( A ); ++i) {
            if (slate_TriangularMatrix_tileIsLocal_c64( A, i, j )) {
                slate_Tile_c64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_TriangularMatrix_at_c64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_TriangularMatrix_at_c64(A, i, j);
                else
                    continue;

                random_Tile_c64( T );
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random, diagonally dominant matrix A
void random_HermitianMatrix_diag_dominant_r32(
    slate_HermitianMatrix_r32 A, slate_Uplo uplo )
{
    int64_t max_mn = MAX( slate_HermitianMatrix_m_r32( A ),
                          slate_HermitianMatrix_n_r32( A ) );
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_r32( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_r32( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_r32( A, i, j )) {
                slate_Tile_r32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_r32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_r32(A, i, j);
                else
                    continue;

                random_Tile_r32( T );

                if (i == j) {
                    // assuming tileMb == tileNb, then i == j are diagonal tiles
                    // make diagonal real & dominant
                    int64_t min_mb_nb = MIN(
                              slate_Tile_mb_r32( T ), slate_Tile_nb_r32( T ) );
                    float* data = slate_Tile_data_r32( T );
                    for (int64_t ii = 0; ii < min_mb_nb; ++ii) {
                        data[ ii + ii*slate_Tile_stride_r32( T ) ] =
                          creal( data[ ii + ii*slate_Tile_stride_r32( T ) ] ) +
                          max_mn;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random, diagonally dominant matrix A
void random_HermitianMatrix_diag_dominant_r64(
    slate_HermitianMatrix_r64 A, slate_Uplo uplo )
{
    int64_t max_mn = MAX( slate_HermitianMatrix_m_r64( A ),
                          slate_HermitianMatrix_n_r64( A ) );
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_r64( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_r64( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_r64( A, i, j )) {
                slate_Tile_r64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_r64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_r64(A, i, j);
                else
                    continue;

                random_Tile_r64( T );

                if (i == j) {
                    // assuming tileMb == tileNb, then i == j are diagonal tiles
                    // make diagonal real & dominant
                    int64_t min_mb_nb = MIN(
                              slate_Tile_mb_r64( T ), slate_Tile_nb_r64( T ) );
                    double* data = slate_Tile_data_r64( T );
                    for (int64_t ii = 0; ii < min_mb_nb; ++ii) {
                        data[ ii + ii*slate_Tile_stride_r64( T ) ] =
                          creal( data[ ii + ii*slate_Tile_stride_r64( T ) ] ) +
                          max_mn;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random, diagonally dominant matrix A
void random_HermitianMatrix_diag_dominant_c32(
    slate_HermitianMatrix_c32 A, slate_Uplo uplo )
{
    int64_t max_mn = MAX( slate_HermitianMatrix_m_c32( A ),
                          slate_HermitianMatrix_n_c32( A ) );
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_c32( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_c32( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_c32( A, i, j )) {
                slate_Tile_c32 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_c32(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_c32(A, i, j);
                else
                    continue;

                random_Tile_c32( T );

                if (i == j) {
                    // assuming tileMb == tileNb, then i == j are diagonal tiles
                    // make diagonal real & dominant
                    int64_t min_mb_nb = MIN(
                              slate_Tile_mb_c32( T ), slate_Tile_nb_c32( T ) );
                    float _Complex* data = slate_Tile_data_c32( T );
                    for (int64_t ii = 0; ii < min_mb_nb; ++ii) {
                        data[ ii + ii*slate_Tile_stride_c32( T ) ] =
                          creal( data[ ii + ii*slate_Tile_stride_c32( T ) ] ) +
                          max_mn;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// generate random, diagonally dominant matrix A
void random_HermitianMatrix_diag_dominant_c64(
    slate_HermitianMatrix_c64 A, slate_Uplo uplo )
{
    int64_t max_mn = MAX( slate_HermitianMatrix_m_c64( A ),
                          slate_HermitianMatrix_n_c64( A ) );
    for (int64_t j = 0; j < slate_HermitianMatrix_nt_c64( A ); ++j) {
        for (int64_t i = 0; i < slate_HermitianMatrix_mt_c64( A ); ++i) {
            if (slate_HermitianMatrix_tileIsLocal_c64( A, i, j )) {
                slate_Tile_c64 T;
                if (uplo == slate_Uplo_Upper && i <= j)
                    T = slate_HermitianMatrix_at_c64(A, i, j);
                else if (uplo == slate_Uplo_Lower && i >= j)
                    T = slate_HermitianMatrix_at_c64(A, i, j);
                else
                    continue;

                random_Tile_c64( T );

                if (i == j) {
                    // assuming tileMb == tileNb, then i == j are diagonal tiles
                    // make diagonal real & dominant
                    int64_t min_mb_nb = MIN(
                              slate_Tile_mb_c64( T ), slate_Tile_nb_c64( T ) );
                    double _Complex* data = slate_Tile_data_c64( T );
                    for (int64_t ii = 0; ii < min_mb_nb; ++ii) {
                        data[ ii + ii*slate_Tile_stride_c64( T ) ] =
                          creal( data[ ii + ii*slate_Tile_stride_c64( T ) ] ) +
                          max_mn;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
void print_matrix_r32( const char* label, int m, int n, float* A, int lda )
{
    printf( "%s = [\n", label );
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf( "  %7.4f", A[ i + j*lda ] );
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

//------------------------------------------------------------------------------
void print_matrix_r64( const char* label, int m, int n, double* A, int lda )
{
    printf( "%s = [\n", label );
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf( "  %7.4f", A[ i + j*lda ] );
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

//------------------------------------------------------------------------------
void print_matrix_c32(
    const char* label, int m, int n, float _Complex* A, int lda )
{
    printf( "%s = [\n", label );
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf( "  %7.4f + %7.4fi",
                    creal(A[ i + j*lda ]), cimag(A[ i + j*lda ]) );
        }
        printf( "\n" );
    }
    printf( "];\n" );
}


//------------------------------------------------------------------------------
void print_matrix_c64(
    const char* label, int m, int n, double _Complex* A, int lda )
{
    printf( "%s = [\n", label );
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf( "  %7.4f + %7.4fi",
                    creal(A[ i + j*lda ]), cimag(A[ i + j*lda ]) );
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
    for (p = (int) sqrt( mpi_size ); p > 0; --p) {
        q = (int) mpi_size / p;
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
    *p_out = (int) sqrt( mpi_size );
    *q_out = *p_out;
}

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
        if (strcmp( argv[ i ], "s" ) == 0)
            types[ 0 ] = true;
        else if (strcmp( argv[ i ], "d" ) == 0)
            types[ 1 ] = true;
        else if (strcmp( argv[ i ], "c" ) == 0)
            types[ 2 ] = true;
        else if (strcmp( argv[ i ], "z" ) == 0)
            types[ 3 ] = true;
        else {
            printf( "unknown option: \"%s\"\nUsage: %s [s] [d] [c] [z]\nfor single, double, complex, double-complex.\n",
                    argv[i], argv[0] );
        }
    }
}
#endif // UTIL_H
