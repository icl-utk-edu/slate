// ex05_blas.c
// BLAS routines

#include <slate/c_api/slate.h>
#include <mpi.h>

#include "util.h"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
void test_gemm_r32()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_r32 A = slate_Matrix_create_r32(
        m, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r32 B = slate_Matrix_create_r32(
        k, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r32 C = slate_Matrix_create_r32(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r32( A );
    slate_Matrix_insertLocalTiles_r32( B );
    slate_Matrix_insertLocalTiles_r32( C );
    random_matrix_type_r32( A );
    random_matrix_type_r32( B );
    random_matrix_type_r32( C );

    // C = alpha A B + beta C, where A, B, C are all general matrices.
    slate_multiply_r32( alpha, A, B, beta, C, NULL );

    if (slate_Matrix_num_devices_r32( C ) > 0) {
        // Execute on GPU devices with lookahead of 2.
        slate_Options opts = slate_Options_create();
        slate_Options_set_Target( opts, slate_Target_Devices );
        slate_Options_set_Lookahead( opts, 2 );

        slate_multiply_r32( alpha, A, B, beta, C, opts );

        slate_Options_destroy( opts );
    }

    slate_Matrix_destroy_r32( A );
    slate_Matrix_destroy_r32( B );
    slate_Matrix_destroy_r32( C );
}

//------------------------------------------------------------------------------
void test_gemm_r64()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_r64 A = slate_Matrix_create_r64(
        m, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r64 B = slate_Matrix_create_r64(
        k, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r64 C = slate_Matrix_create_r64(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r64( A );
    slate_Matrix_insertLocalTiles_r64( B );
    slate_Matrix_insertLocalTiles_r64( C );
    random_matrix_type_r64( A );
    random_matrix_type_r64( B );
    random_matrix_type_r64( C );

    // C = alpha A B + beta C, where A, B, C are all general matrices.
    slate_multiply_r64( alpha, A, B, beta, C, NULL );

    if (slate_Matrix_num_devices_r64( C ) > 0) {
        // Execute on GPU devices with lookahead of 2.
        slate_Options opts = slate_Options_create();
        slate_Options_set_Target( opts, slate_Target_Devices );
        slate_Options_set_Lookahead( opts, 2 );

        slate_multiply_r64( alpha, A, B, beta, C, opts );

        slate_Options_destroy( opts );
    }

    slate_Matrix_destroy_r64( A );
    slate_Matrix_destroy_r64( B );
    slate_Matrix_destroy_r64( C );
}

//------------------------------------------------------------------------------
void test_gemm_c32()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_c32 A = slate_Matrix_create_c32(
        m, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c32 B = slate_Matrix_create_c32(
        k, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c32 C = slate_Matrix_create_c32(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c32( A );
    slate_Matrix_insertLocalTiles_c32( B );
    slate_Matrix_insertLocalTiles_c32( C );
    random_matrix_type_c32( A );
    random_matrix_type_c32( B );
    random_matrix_type_c32( C );

    // C = alpha A B + beta C, where A, B, C are all general matrices.
    slate_multiply_c32( alpha, A, B, beta, C, NULL );

    if (slate_Matrix_num_devices_c32( C ) > 0) {
        // Execute on GPU devices with lookahead of 2.
        slate_Options opts = slate_Options_create();
        slate_Options_set_Target( opts, slate_Target_Devices );
        slate_Options_set_Lookahead( opts, 2 );

        slate_multiply_c32( alpha, A, B, beta, C, opts );

        slate_Options_destroy( opts );
    }

    slate_Matrix_destroy_c32( A );
    slate_Matrix_destroy_c32( B );
    slate_Matrix_destroy_c32( C );
}

//------------------------------------------------------------------------------
void test_gemm_c64()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_c64 A = slate_Matrix_create_c64(
        m, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c64 B = slate_Matrix_create_c64(
        k, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c64 C = slate_Matrix_create_c64(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c64( A );
    slate_Matrix_insertLocalTiles_c64( B );
    slate_Matrix_insertLocalTiles_c64( C );
    random_matrix_type_c64( A );
    random_matrix_type_c64( B );
    random_matrix_type_c64( C );

    // C = alpha A B + beta C, where A, B, C are all general matrices.
    slate_multiply_c64( alpha, A, B, beta, C, NULL );

    if (slate_Matrix_num_devices_c64( C ) > 0) {
        // Execute on GPU devices with lookahead of 2.
        slate_Options opts = slate_Options_create();
        slate_Options_set_Target( opts, slate_Target_Devices );
        slate_Options_set_Lookahead( opts, 2 );

        slate_multiply_c64( alpha, A, B, beta, C, opts );

        slate_Options_destroy( opts );
    }

    slate_Matrix_destroy_c64( A );
    slate_Matrix_destroy_c64( B );
    slate_Matrix_destroy_c64( C );
}

//------------------------------------------------------------------------------
void test_gemm_trans_r32()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_r32 A = slate_Matrix_create_r32(
        k, m,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r32 B = slate_Matrix_create_r32(
        n, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r32 C = slate_Matrix_create_r32(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r32( A );
    slate_Matrix_insertLocalTiles_r32( B );
    slate_Matrix_insertLocalTiles_r32( C );
    random_matrix_type_r32( A );
    random_matrix_type_r32( B );
    random_matrix_type_r32( C );

    // Matrices can be transposed or conjugate-transposed beforehand.
    // C = alpha A^T B^H + beta C
    slate_Matrix_transpose_in_place_r32( A );
    slate_Matrix_conj_transpose_in_place_r32( B );
    slate_multiply_r32( alpha, A, B, beta, C, NULL );  // simplified API

    slate_Matrix_destroy_r32( A );
    slate_Matrix_destroy_r32( B );
    slate_Matrix_destroy_r32( C );
}

//------------------------------------------------------------------------------
void test_gemm_trans_r64()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_r64 A = slate_Matrix_create_r64(
        k, m,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r64 B = slate_Matrix_create_r64(
        n, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_r64 C = slate_Matrix_create_r64(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_r64( A );
    slate_Matrix_insertLocalTiles_r64( B );
    slate_Matrix_insertLocalTiles_r64( C );
    random_matrix_type_r64( A );
    random_matrix_type_r64( B );
    random_matrix_type_r64( C );

    // Matrices can be transposed or conjugate-transposed beforehand.
    // C = alpha A^T B^H + beta C
    slate_Matrix_transpose_in_place_r64( A );
    slate_Matrix_conj_transpose_in_place_r64( B );
    slate_multiply_r64( alpha, A, B, beta, C, NULL );  // simplified API

    slate_Matrix_destroy_r64( A );
    slate_Matrix_destroy_r64( B );
    slate_Matrix_destroy_r64( C );
}

//------------------------------------------------------------------------------
void test_gemm_trans_c32()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_c32 A = slate_Matrix_create_c32(
        k, m,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c32 B = slate_Matrix_create_c32(
        n, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c32 C = slate_Matrix_create_c32(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c32( A );
    slate_Matrix_insertLocalTiles_c32( B );
    slate_Matrix_insertLocalTiles_c32( C );
    random_matrix_type_c32( A );
    random_matrix_type_c32( B );
    random_matrix_type_c32( C );

    // Matrices can be transposed or conjugate-transposed beforehand.
    // C = alpha A^T B^H + beta C
    slate_Matrix_transpose_in_place_c32( A );
    slate_Matrix_conj_transpose_in_place_c32( B );
    slate_multiply_c32( alpha, A, B, beta, C, NULL );  // simplified API

    slate_Matrix_destroy_c32( A );
    slate_Matrix_destroy_c32( B );
    slate_Matrix_destroy_c32( C );
}

//------------------------------------------------------------------------------
void test_gemm_trans_c64()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate_Matrix_c64 A = slate_Matrix_create_c64(
        k, m,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c64 B = slate_Matrix_create_c64(
        n, k,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_c64 C = slate_Matrix_create_c64(
        m, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate_Matrix_insertLocalTiles_c64( A );
    slate_Matrix_insertLocalTiles_c64( B );
    slate_Matrix_insertLocalTiles_c64( C );
    random_matrix_type_c64( A );
    random_matrix_type_c64( B );
    random_matrix_type_c64( C );

    // Matrices can be transposed or conjugate-transposed beforehand.
    // C = alpha A^T B^H + beta C
    slate_Matrix_transpose_in_place_c64( A );
    slate_Matrix_conj_transpose_in_place_c64( B );
    slate_multiply_c64( alpha, A, B, beta, C, NULL );  // simplified API

    slate_Matrix_destroy_c64( A );
    slate_Matrix_destroy_c64( B );
    slate_Matrix_destroy_c64( C );
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
        test_gemm_r32();
        test_gemm_trans_r32();
        if (mpi_rank == 0)
            printf( "\n" );
    }

    if (types[ 1 ]) {
        test_gemm_r64();
        test_gemm_trans_r64();
        if (mpi_rank == 0)
            printf( "\n" );
    }

    if (types[ 2 ]) {
        test_gemm_c32();
        test_gemm_trans_c32();
        if (mpi_rank == 0)
            printf( "\n" );
    }

    if (types[ 3 ]) {
        test_gemm_c64();
        test_gemm_trans_c64();
    }

    MPI_Finalize();

    return 0;
}
