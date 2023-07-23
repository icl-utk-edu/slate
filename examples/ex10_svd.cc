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

    int64_t m=2000, n=1000, nb=256;

    int64_t min_mn = std::min( m, n );
    slate::Matrix<T> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    std::vector<real_t> Sigma( min_mn );

    // A = U Sigma V^H, singular values only
    random_matrix( A );
    slate::svd_vals( A, Sigma );

    random_matrix( A );
    slate::svd( A, Sigma );

    // traditional LAPACK API deprecated
    #ifdef DEPRECATED
        random_matrix( A );
        slate::gesvd( A, Sigma );
    #endif

    // U is m x min_mn (reduced SVD) or m x m (full SVD)
    // V is min_mn x n (reduced SVD) or n x n (full SVD)
    // todo: full SVD not yet supported?
    slate::Matrix<T>  U( m, min_mn, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<T> VH( min_mn, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    U.insertLocalTiles();
    VH.insertLocalTiles();

    // empty, 0x0 matrices as placeholders for U and VH.
    slate::Matrix<T> Uempty, Vempty;

    random_matrix( A );
    slate::svd( A, Sigma, U, VH );  // both U and V^H

    random_matrix( A );
    slate::svd( A, Sigma, U, Vempty );  // only U

    random_matrix( A );
    slate::svd( A, Sigma, Uempty, VH );  // only V^H

    // traditional LAPACK API deprecated
    #ifdef DEPRECATED
        random_matrix( A );
        slate::gesvd( A, Sigma, U, VH );
    #endif
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
            test_svd< float >();
        }

        if (types[ 1 ]) {
            test_svd< double >();
        }

        if (types[ 2 ]) {
            test_svd< std::complex<float> >();
        }

        if (types[ 3 ]) {
            test_svd< std::complex<double> >();
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
