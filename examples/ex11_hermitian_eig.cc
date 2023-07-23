// ex11_hermitian_eig.cc
// Solve Hermitian eigenvalues A = Z Lambda Z^H
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

    int64_t n=1000, nb=256;

    slate::HermitianMatrix<T> A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    std::vector<real_t> Lambda( n );

    // A = Z Lambda Z^H, eigenvalues only
    random_matrix( A );
    slate::eig_vals( A, Lambda );  // simplified API

    // Or
    random_matrix( A );
    slate::eig( A, Lambda );       // simplified API

    random_matrix( A );
    slate::heev( A, Lambda );      // traditional API

    //--------------------
    // Eigenvectors
    slate::Matrix<T> Z( n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    Z.insertLocalTiles();

    random_matrix( A );
    slate::eig( A, Lambda, Z );    // simplified API

    random_matrix( A );
    slate::heev( A, Lambda, Z );   // traditional API
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
        // Hermitian eig requires square MPI grid.
        grid_size_square( mpi_size, &grid_p, &grid_q );
        if (mpi_rank == 0) {
            printf( "mpi_size %d, grid_p %d, grid_q %d\n",
                    mpi_size, grid_p, grid_q );
        }

        // so random_matrix is different on different ranks.
        srand( 100 * mpi_rank );

        if (types[ 0 ]) {
            test_hermitian_eig< float >();
        }

        if (types[ 1 ]) {
            test_hermitian_eig< double >();
        }

        if (types[ 2 ]) {
            test_hermitian_eig< std::complex<float> >();
        }

        if (types[ 3 ]) {
            test_hermitian_eig< std::complex<double> >();
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
