// ex12_generalized_hermitian_eig.cc
// Solve generalized Hermitian eigenvalues, types:
// 1. A = B Z Lambda Z^H
// 2. A B = Z Lambda Z^H
// 3. B A = Z Lambda Z^H
// where B is Hermitian positive definite.
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename T>
void test_hermitian_eig()
{
    using real_t = blas::real_type<T>;

    print_func( mpi_rank );

    // TODO: failing if n not divisible by nb?
    int64_t n=1000, nb=100;

    slate::HermitianMatrix<T> A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::HermitianMatrix<T> B( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    std::vector<real_t> Lambda( n );

    // Type 1: A = B Z Lambda Z^H, eigenvalues only
    random_matrix( A );
    random_matrix_diag_dominant( B );
    slate::eig_vals( 1, A, B, Lambda );  // simplified API

    // Or
    random_matrix( A );
    random_matrix_diag_dominant( B );
    slate::eig( 1, A, B, Lambda );       // simplified API

    random_matrix( A );
    random_matrix_diag_dominant( B );
    slate::hegv( 1, A, B, Lambda );      // traditional API

    // Type 2: A B = Z Lambda Z^H, eigenvalues only
    random_matrix( A );
    random_matrix_diag_dominant( B );
    slate::eig_vals( 2, A, B, Lambda );  // simplified API

    // Type 3: A = B Z Lambda Z^H, eigenvalues only
    random_matrix( A );
    random_matrix_diag_dominant( B );
    slate::eig_vals( 3, A, B, Lambda );  // simplified API

    //--------------------
    // Eigenvectors
    slate::Matrix<T> Z( n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    Z.insertLocalTiles();

    random_matrix( A );
    random_matrix_diag_dominant( B );
    slate::eig( 1, A, B, Lambda, Z );    // simplified API

    random_matrix( A );
    random_matrix_diag_dominant( B );
    slate::hegv( 1, A, B, Lambda, Z );   // traditional API
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
    // Hermitian eig requires square MPI grid.
    grid_size_square( mpi_size, &grid_p, &grid_q );
    if (mpi_rank == 0) {
        printf( "mpi_size %d, grid_p %d, grid_q %d\n",
                mpi_size, grid_p, grid_q );
    }

    // so random_matrix is different on different ranks.
    srand( 100 * mpi_rank );

    test_hermitian_eig< float >();
    test_hermitian_eig< double>();
    test_hermitian_eig< std::complex<float> >();
    test_hermitian_eig< std::complex<double> >();

    slate_mpi_call(
        MPI_Finalize() );

    return 0;
}
