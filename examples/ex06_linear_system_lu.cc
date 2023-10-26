// ex06_linear_system_lu.cc
// Solve AX = B using LU factorization

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
void test_lu()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;

    //---------- begin solve1
    slate::Matrix<scalar_type> A( n, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    // ...
    //---------- end solve1

    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    //---------- begin solve2

    slate::lu_solve( A, B );        // simplified API

    slate::Pivots pivots;
    slate::gesv( A, pivots, B );    // traditional API
    //---------- end solve2
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_lu_mixed()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;
    scalar_type zero = 0;

    //---------- begin mixed1
    // mixed precision: factor in single, iterative refinement to double
    slate::Matrix<scalar_type> A( n, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> X( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B1( n, 1,   nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> X1( n, 1,   nb, grid_p, grid_q, MPI_COMM_WORLD );
    int iters = 0;
    // ...
    //---------- end mixed1

    A.insertLocalTiles();
    B.insertLocalTiles();
    X.insertLocalTiles();
    B1.insertLocalTiles();
    X1.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( B1 );
    set( zero, X );
    set( zero, X1 );
    slate::Pivots pivots;

    //---------- begin mixed2

    // todo: simplified API

    // traditional API
    slate::gesv_mixed( A, pivots, B, X, iters );
    slate::gesv_mixed_gmres( A, pivots, B1, X1, iters );  // only one RHS
    //---------- end mixed2

    printf( "rank %d: iters %d\n", mpi_rank, iters );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_lu_factor()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256;

    slate::Matrix<scalar_type> A( n, n,    nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    slate::Pivots pivots;

    // simplified API
    slate::lu_factor( A, pivots );
    slate::lu_solve_using_factor( A, pivots, B );

    // traditional API
    slate::getrf( A, pivots );     // factor
    slate::getrs( A, pivots, B );  // solve
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_lu_inverse()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    //---------- begin inverse1
    slate::Matrix<scalar_type> A( n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Pivots pivots;
    // ...
    //---------- end inverse1

    A.insertLocalTiles();
    random_matrix( A );

    //---------- begin inverse2

    // simplified API
    slate::lu_factor( A, pivots );
    slate::lu_inverse_using_factor( A, pivots );

    // traditional API
    slate::getrf( A, pivots );  // factor
    slate::getri( A, pivots );  // inverse
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
            test_lu< float >();
            test_lu_factor< float >();
            test_lu_inverse< float >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 1 ]) {
            test_lu< double >();
            test_lu_factor< double >();
            test_lu_inverse< double >();
            test_lu_mixed< double >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 2 ]) {
            test_lu< std::complex<float> >();
            test_lu_factor< std::complex<float> >();
            test_lu_inverse< std::complex<float> >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 3 ]) {
            test_lu< std::complex<double> >();
            test_lu_factor< std::complex<double> >();
            test_lu_inverse< std::complex<double> >();
            test_lu_mixed< std::complex<double> >();
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
