// ex02_conversion.cc
// conversion between matrix types
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_conversion()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    slate::Matrix<scalar_type>
        A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );

    // triangular and symmetric matrices must be square -- take square slice.
    auto A_square = A.slice( 0, n-1, 0, n-1 );

    slate::TriangularMatrix<scalar_type>
    L( slate::Uplo::Lower,
       slate::Diag::Unit, A_square );

    slate::TriangularMatrix<scalar_type>
        U( slate::Uplo::Upper,
           slate::Diag::NonUnit, A_square );

    slate::SymmetricMatrix<scalar_type>
        S( slate::Uplo::Upper, A_square );
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

    test_conversion< float >();
    test_conversion< double >();
    test_conversion< std::complex<float> >();
    test_conversion< std::complex<double> >();

    slate_mpi_call(
        MPI_Finalize() );

    return 0;
}
