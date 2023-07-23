// ex09_least_squares.cc
// Solve over- and under-determined AX = B
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_gels_overdetermined()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nrhs=100, nb=256;

    int64_t max_mn = std::max( m, n );
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> BX( max_mn, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    BX.insertLocalTiles();
    auto B = BX;  // == BX.slice( 0, m-1, 0, nrhs-1 );
    auto X = BX.slice( 0, n-1, 0, nrhs-1 );
    random_matrix( A );
    random_matrix( B );

    // solve AX = B, solution in X
    slate::least_squares_solve( A, BX );  // simplified API

    slate::gels( A, BX );                 // traditional API
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_gels_underdetermined()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nrhs=100, nb=256;

    int64_t max_mn = std::max( m, n );
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> BX( max_mn, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    BX.insertLocalTiles();
    auto B = BX.slice( 0, n-1, 0, nrhs-1 );
    auto X = BX;  // == BX.slice( 0, m-1, 0, nrhs-1 );
    random_matrix( A );
    random_matrix( B );

    // solve A^H X = B, solution in X
    auto AH = conj_transpose( A );
    slate::least_squares_solve( AH, BX );  // simplified API

    slate::gels( AH, BX );                 // traditional API
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
            test_gels_overdetermined < float >();
            test_gels_underdetermined< float >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 1 ]) {
            test_gels_overdetermined < double >();
            test_gels_underdetermined< double >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 2 ]) {
            test_gels_overdetermined < std::complex<float> >();
            test_gels_underdetermined< std::complex<float> >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 3 ]) {
            test_gels_overdetermined < std::complex<double> >();
            test_gels_underdetermined< std::complex<double> >();
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
