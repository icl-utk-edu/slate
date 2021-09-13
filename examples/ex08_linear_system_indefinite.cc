// ex08_linear_system_indefinite.cc
// Solve AX = B using Aasen's symmetric indefinite factorization
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;

//------------------------------------------------------------------------------
// TODO: failing
void test_hesv()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=100, p=2, q=2;
    assert( mpi_size == p*q );
    slate::HermitianMatrix<double>
        A( slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, nrhs, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    // simplified API
    slate::indefinite_solve( A, B );

    // traditional API
    // workspaces
    // todo: drop H (internal workspace)
    slate::Matrix<double>     H( n, n, nb, p, q, MPI_COMM_WORLD );
    slate::BandMatrix<double> T( n, n, nb, nb, nb, p, q, MPI_COMM_WORLD );
    slate::Pivots pivots, pivots2;

    slate::hesv( A, pivots, T, pivots2, H, B );
}

//------------------------------------------------------------------------------
// TODO: failing
void test_hetrf()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=100, p=2, q=2;
    assert( mpi_size == p*q );
    slate::HermitianMatrix<double>
        A( slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, nrhs, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    // workspaces
    // todo: drop H (internal workspace)
    slate::Matrix<double>     H( n, n, nb, p, q, MPI_COMM_WORLD );
    slate::BandMatrix<double> T( n, n, nb, nb, nb, p, q, MPI_COMM_WORLD );
    slate::Pivots pivots, pivots2;

    // simplified API
    slate::indefinite_factor( A, pivots, T, pivots2, H );
    slate::indefinite_solve_using_factor( A, pivots, T, pivots2, B );

    // traditional API
    slate::hetrf( A, pivots, T, pivots2, H );  // factor
    slate::hetrs( A, pivots, T, pivots2, B );  // solve
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

    test_hesv();
    test_hetrf();

    err = MPI_Finalize();
    assert( err == 0 );
}
