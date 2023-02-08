// ex10_svd.cc
// Solve Singular Value Decomposition, A = U Sigma V^H
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename T>
void test_svd()
{
    using real_t = blas::real_type<T>;

    print_func( mpi_rank );

    // TODO: failing if m, n not divisible by nb?
    int64_t m=2000, n=1000, nb=100;

    int64_t min_mn = std::min( m, n );
    slate::Matrix<T> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
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
    // slate::Matrix<T>  U( m, min_mn, nb, grid_p, grid_q, MPI_COMM_WORLD );
    // slate::Matrix<T> VH( min_mn, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
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

    test_svd<double>();

    slate_mpi_call(
        MPI_Finalize() );

    return 0;
}
