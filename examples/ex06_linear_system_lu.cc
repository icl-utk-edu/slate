// ex06_linear_system_lu.cc
// Solve AX = B using LU factorization
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_lu()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256, p=2, q=2;
    assert( mpi_size == p*q );
    slate::Matrix<scalar_type> A( n, n,    nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    // simplified API
    slate::lu_solve( A, B );

    // traditional API
    slate::Pivots pivots;
    slate::gesv( A, pivots, B );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_lu_mixed()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256, p=2, q=2;
    assert( mpi_size == p*q );
    slate::Matrix<scalar_type> A( n, n,    nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> X( n, nrhs, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    X.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    scalar_type zero = 0;
    slate::set( zero, zero, X );
    random_matrix( X );
    slate::Pivots pivots;

    // todo: simplified API

    // traditional API
    // TODO: pass using &iters?
    int iters = 0;
    slate::gesvMixed( A, pivots, B, X, iters );
    printf( "rank %d: iters %d\n", mpi_rank, iters );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_lu_factor()
{
    print_func( mpi_rank );

    int64_t n=1000, nrhs=100, nb=256, p=2, q=2;
    assert( mpi_size == p*q );
    slate::Matrix<scalar_type> A( n, n,    nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type> B( n, nrhs, nb, p, q, MPI_COMM_WORLD );
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

    int64_t n=1000, nb=256, p=2, q=2;
    assert( mpi_size == p*q );
    slate::Matrix<scalar_type> A( n, n, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );
    slate::Pivots pivots;

    // simplified API
    slate::lu_factor( A, pivots );
    slate::lu_inverse_using_factor( A, pivots );

    // traditional API
    slate::getrf( A, pivots );  // factor
    slate::getri( A, pivots );  // inverse
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

    test_lu< float >();
    test_lu< double >();
    test_lu< std::complex<float> >();
    test_lu< std::complex<double> >();

    test_lu_mixed< double >();
    test_lu_mixed< std::complex<double> >();

    test_lu_factor< float >();
    test_lu_factor< double >();
    test_lu_factor< std::complex<float> >();
    test_lu_factor< std::complex<double> >();

    test_lu_inverse< float >();
    test_lu_inverse< double >();
    test_lu_inverse< std::complex<float> >();
    test_lu_inverse< std::complex<double> >();

    err = MPI_Finalize();
    assert( err == 0 );
}
