// ex02_conversion.cc
// conversion between matrix types
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_conversion()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256, p=2, q=2;
    assert( mpi_size == p*q );
    slate::Matrix<scalar_type>
        A( m, n, nb, p, q, MPI_COMM_WORLD );

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

    err = MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
    assert( err == 0 );
    if (mpi_size != 4) {
        printf( "Usage: mpirun -np 4 %s  # 4 ranks hard coded\n", argv[0] );
        return -1;
    }

    err = MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    assert( err == 0 );

    // so random_matrix is different on different ranks.
    srand( 100 * mpi_rank );

    test_conversion< float >();
    test_conversion< double >();
    test_conversion< std::complex<float> >();
    test_conversion< std::complex<double> >();

    err = MPI_Finalize();
    assert( err == 0 );
}
