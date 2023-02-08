// ex07_linear_system_cholesky.cc
// Solve AX = B using Cholesky factorization
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

    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix_diag_dominant( A );
    random_matrix( B );

    slate::chol_solve( A, B );  // simplified API

    slate::posv( A, B );        // traditional API
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_cholesky_mixed()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;

    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> X( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    X.insertLocalTiles();
    random_matrix_diag_dominant( A );
    random_matrix( B );
    scalar_type zero = 0;
    slate::set( zero, zero, X );

    // todo: simplified API

    // traditional API
    // TODO: pass using &iters?
    int iters = 0;
    slate::posvMixed( A, B, X, iters );
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

    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix_diag_dominant( A );

    // simplified API
    slate::chol_factor( A );
    slate::chol_inverse_using_factor( A );

    // traditional API
    slate::potrf( A );  // factor
    slate::potri( A );  // inverse
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int provided = 0;
    int err = MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    assert( err == 0 );
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

    test_cholesky< float >();
    test_cholesky< double >();
    test_cholesky< std::complex<float> >();
    test_cholesky< std::complex<double> >();

    test_cholesky_mixed< double >();
    test_cholesky_mixed< std::complex<double> >();

    test_cholesky_factor< float >();
    test_cholesky_factor< double >();
    test_cholesky_factor< std::complex<float> >();
    test_cholesky_factor< std::complex<double> >();

    test_cholesky_inverse< float >();
    test_cholesky_inverse< double >();
    test_cholesky_inverse< std::complex<float> >();
    test_cholesky_inverse< std::complex<double> >();

    slate_mpi_call(
        MPI_Finalize() );

    return 0;
}
