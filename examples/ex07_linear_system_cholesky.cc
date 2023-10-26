// ex07_linear_system_cholesky.cc
// Solve AX = B using Cholesky factorization

/// !!!   Lines between `//---------- begin label`          !!!
/// !!!             and `//---------- end label`            !!!
/// !!!   are included in the SLATE Users' Guide.           !!!

#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_cholesky()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;

    //---------- begin solve1
    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    // ...
    //---------- end solve1

    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix_diag_dominant( A );
    random_matrix( B );

    //---------- begin solve2
    slate::chol_solve( A, B );  // simplified API

    slate::posv( A, B );        // traditional API
    //---------- end solve2
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_cholesky_mixed()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;
    scalar_type zero = 0;

    //---------- begin mixed1
    // mixed precision: factor in single, iterative refinement to double
    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> X( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B1( n, 1,   nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> X1( n, 1,   nb, grid_p, grid_q, MPI_COMM_WORLD );
    int iters = 0;
    //---------- end mixed1

    A.insertLocalTiles();
    B.insertLocalTiles();
    X.insertLocalTiles();
    B1.insertLocalTiles();
    X1.insertLocalTiles();
    random_matrix_diag_dominant( A );
    random_matrix( B );
    random_matrix( B1 );
    slate::set( zero, X );
    slate::set( zero, X1 );

    //---------- begin mixed2

    // todo: simplified API

    // traditional API
    slate::posv_mixed( A, B, X, iters );
    slate::posv_mixed_gmres( A, B1, X1, iters );  // only one RHS
    //---------- end mixed2

    printf( "rank %d: iters %d\n", mpi_rank, iters );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_cholesky_factor()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;

    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix_diag_dominant( A );
    random_matrix( B );

    // simplified API
    slate::chol_factor( A );
    slate::chol_solve_using_factor( A, B );

    // traditional API
    slate::potrf( A );     // factor
    slate::potrs( A, B );  // solve
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_cholesky_inverse()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    //---------- begin inverse1
    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    // ...
    //---------- end inverse1

    A.insertLocalTiles();
    random_matrix_diag_dominant( A );

    //---------- begin inverse2

    // simplified API
    slate::chol_factor( A );
    slate::chol_inverse_using_factor( A );

    // traditional API
    slate::potrf( A );  // factor
    slate::potri( A );  // inverse
    //---------- end inverse2
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    try {
        // Parse command line to set types for s, d, c, z precisions.
        bool types[ 4 ];
        parse_args( argc, argv, types );

        int provided = 0;
        slate_mpi_call(
            MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided ) );
        assert( provided == MPI_THREAD_MULTIPLE );

        slate_mpi_call(
            MPI_Comm_size( MPI_COMM_WORLD, &mpi_size ) );

        slate_mpi_call(
            MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank ) );

        // Determine p-by-q grid for this MPI size.
        grid_size( mpi_size, &grid_p, &grid_q );
        if (mpi_rank == 0) {
            printf( "mpi_size %d, grid_p %d, grid_q %d\n",
                    mpi_size, grid_p, grid_q );
        }

        // so random_matrix is different on different ranks.
        srand( 100 * mpi_rank );

        if (types[ 0 ]) {
            test_cholesky< float >();
            test_cholesky_factor< float >();
            test_cholesky_inverse< float >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 1 ]) {
            test_cholesky< double >();
            test_cholesky_factor< double >();
            test_cholesky_inverse< double >();
            test_cholesky_mixed< double >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 2 ]) {
            test_cholesky< std::complex<float> >();
            test_cholesky_factor< std::complex<float> >();
            test_cholesky_inverse< std::complex<float> >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 3 ]) {
            test_cholesky< std::complex<double> >();
            test_cholesky_factor< std::complex<double> >();
            test_cholesky_inverse< std::complex<double> >();
            test_cholesky_mixed< std::complex<double> >();
        }

        slate_mpi_call(
            MPI_Finalize() );
    }
    catch (std::exception const& ex) {
        fprintf( stderr, "%s", ex.what() );
        return 1;
    }
    return 0;
}
