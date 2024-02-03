// slate06_linear_system_lu.c
// Solve AX = B using LU factorization

#include <slate/c_api/slate.h>
#include <mpi.h>

#include "util.h"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
void test_lu_r32()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_r32 A = slate_Matrix_create_r32(
        n, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r32 B = slate_Matrix_create_r32(
        n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r32( A );
    slate_Matrix_insertLocalTiles_r32( B );
    random_Matrix_r32( A );
    random_Matrix_r32( B );

    slate_lu_solve_r32( A, B, NULL );

    slate_Matrix_destroy_r32( A );
    slate_Matrix_destroy_r32( B );
}

//------------------------------------------------------------------------------
void test_lu_r64()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_r64 A = slate_Matrix_create_r64(
        n, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r64 B = slate_Matrix_create_r64(
        n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r64( A );
    slate_Matrix_insertLocalTiles_r64( B );
    random_Matrix_r64( A );
    random_Matrix_r64( B );

    slate_lu_solve_r64( A, B, NULL );

    slate_Matrix_destroy_r64( A );
    slate_Matrix_destroy_r64( B );
}

//------------------------------------------------------------------------------
void test_lu_c32()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_c32 A = slate_Matrix_create_c32(
        n, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c32 B = slate_Matrix_create_c32(
        n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c32( A );
    slate_Matrix_insertLocalTiles_c32( B );
    random_Matrix_c32( A );
    random_Matrix_c32( B );

    slate_lu_solve_c32( A, B, NULL );

    slate_Matrix_destroy_c32( A );
    slate_Matrix_destroy_c32( B );
}

//------------------------------------------------------------------------------
void test_lu_c64()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_c64 A = slate_Matrix_create_c64(
        n, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c64 B = slate_Matrix_create_c64(
        n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c64( A );
    slate_Matrix_insertLocalTiles_c64( B );
    random_Matrix_c64( A );
    random_Matrix_c64( B );

    slate_lu_solve_c64( A, B, NULL );

    slate_Matrix_destroy_c64( A );
    slate_Matrix_destroy_c64( B );
}

//------------------------------------------------------------------------------
void test_lu_inverse_r32()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_r32 A = slate_Matrix_create_r32(
        n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r32( A );
    random_Matrix_r32( A );
    slate_Pivots pivots = slate_Pivots_create();

    slate_lu_factor_r32( A, pivots, NULL );
    slate_lu_inverse_using_factor_r32( A, pivots, NULL );

    slate_Matrix_destroy_r32( A );
    slate_Pivots_destroy( pivots );
}

//------------------------------------------------------------------------------
void test_lu_inverse_r64()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_r64 A = slate_Matrix_create_r64(
        n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r64( A );
    random_Matrix_r64( A );
    slate_Pivots pivots = slate_Pivots_create();

    slate_lu_factor_r64( A, pivots, NULL );
    slate_lu_inverse_using_factor_r64( A, pivots, NULL );

    slate_Matrix_destroy_r64( A );
    slate_Pivots_destroy( pivots );
}

//------------------------------------------------------------------------------
void test_lu_inverse_c32()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_c32 A = slate_Matrix_create_c32(
        n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c32( A );
    random_Matrix_c32( A );
    slate_Pivots pivots = slate_Pivots_create();

    slate_lu_factor_c32( A, pivots, NULL );
    slate_lu_inverse_using_factor_c32( A, pivots, NULL );

    slate_Matrix_destroy_c32( A );
    slate_Pivots_destroy( pivots );
}

//------------------------------------------------------------------------------
void test_lu_inverse_c64()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;
    assert( mpi_size == grid_p*grid_q );
    slate_Matrix_c64 A = slate_Matrix_create_c64(
        n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c64( A );
    random_Matrix_c64( A );
    slate_Pivots pivots = slate_Pivots_create();

    slate_lu_factor_c64( A, pivots, NULL );
    slate_lu_inverse_using_factor_c64( A, pivots, NULL );

    slate_Matrix_destroy_c64( A );
    slate_Pivots_destroy( pivots );
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    // Parse command line to set types for s, d, c, z precisions.
    bool types[ 4 ];
    parse_args( argc, argv, types );

    int provided = 0;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    assert( provided == MPI_THREAD_MULTIPLE );

    MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

    // Determine p-by-q grid for this MPI size.
    grid_size( mpi_size, &grid_p, &grid_q );
    if (mpi_rank == 0) {
        printf( "mpi_size %d, grid_p %d, grid_q %d\n",
                mpi_size, grid_p, grid_q );
    }

    // so random_matrix is different on different ranks.
    srand( 100 * mpi_rank );

    if (types[ 0 ]) {
        test_lu_r32();
        test_lu_inverse_r32();
    }
    if (mpi_rank == 0)
        printf( "\n" );

    if (types[ 1 ]) {
        test_lu_r64();
        test_lu_inverse_r64();
    }
    if (mpi_rank == 0)
        printf( "\n" );

    if (types[ 2 ]) {
        test_lu_c32();
        test_lu_inverse_c32();
    }
    if (mpi_rank == 0)
        printf( "\n" );

    if (types[ 3 ]) {
        test_lu_c64();
        test_lu_inverse_c64();
    }

    MPI_Finalize();

    return 0;
}
