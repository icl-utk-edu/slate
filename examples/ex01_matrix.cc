// ex01_matrix.cc
// create 2000 x 1000 matrix on 2 x 2 MPI process grid
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_matrix()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256, p=2, q=2;
    assert( mpi_size == p*q );
    slate::Matrix<scalar_type> A( m, n, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                slate::Tile<scalar_type> T = A( i, j );
                // or: auto T = A( i, j );
                random_matrix( T.mb(), T.nb(), T.data(), T.stride() );
            }
        }
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    printf( "argc %d\n", argc );
    for (int i = 0; i < argc; ++i) {
        printf( "argv: '%s'\n", argv[i] );
    }
    printf( "\n" );

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

    test_matrix< float >();
    test_matrix< double >();
    test_matrix< std::complex<float> >();
    test_matrix< std::complex<double> >();

    err = MPI_Finalize();
    assert( err == 0 );
}
