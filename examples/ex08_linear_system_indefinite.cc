// ex08_linear_system_indefinite.cc
// Solve AX = B using Aasen's symmetric indefinite factorization
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
// TODO: failing
void test_hesv()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=100;

    slate::HermitianMatrix<double>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    // simplified API
    slate::indefinite_solve( A, B );

    // traditional API
    // workspaces
    // todo: drop H (internal workspace)
    slate::Matrix<double>     H( n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::BandMatrix<double> T( n, n, nb, nb, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Pivots pivots, pivots2;

    slate::hesv( A, pivots, T, pivots2, H, B );
}

//------------------------------------------------------------------------------
// TODO: failing
void test_hetrf()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=100;

    slate::HermitianMatrix<double>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    // workspaces
    // todo: drop H (internal workspace)
    slate::Matrix<double>     H( n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::BandMatrix<double> T( n, n, nb, nb, nb, grid_p, grid_q, MPI_COMM_WORLD );
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

    test_hesv();
    test_hetrf();

    slate_mpi_call(
        MPI_Finalize() );

    return 0;
}
