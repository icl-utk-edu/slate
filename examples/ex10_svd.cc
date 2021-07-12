// ex10_svd.cc
// Solve Singular Value Decomposition, A = U Sigma V^H
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;

//------------------------------------------------------------------------------
template <typename T>
void test_svd()
{
    using real_t = blas::real_type<T>;

    print_func( mpi_rank );

    // TODO: failing if m, n not divisible by nb?
    int64_t m=2000, n=1000, nb=100, p=2, q=2;
    assert( mpi_size == p*q );
    int64_t min_mn = std::min( m, n );
    slate::Matrix<T> A( m, n, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );
    std::vector<real_t> Sigma( min_mn );

    // A = U Sigma V^H, singular values only
    slate::svd_vals( A, Sigma );  // simplified API

    random_matrix( A );

    slate::gesvd( A, Sigma );     // traditional API (deprecated)

    // TODO: singular vectors
    // U is m x min_mn (reduced SVD) or m x m (full SVD)
    // V is min_mn x n (reduced SVD) or n x n (full SVD)
    // slate::Matrix<T>  U( m, min_mn, nb, p, q, MPI_COMM_WORLD );
    // slate::Matrix<T> VH( min_mn, n, nb, p, q, MPI_COMM_WORLD );
    // U.insertLocalTiles();
    // VT.insertLocalTiles();
    // slate::svd( A, U, Sigma, VH );  // both U and V^H
    // slate::svd( A, U, Sigma     );  // only U
    // slate::svd( A,    Sigma, VH );  // only V^H
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

    test_svd<double>();

    err = MPI_Finalize();
    assert( err == 0 );
}
