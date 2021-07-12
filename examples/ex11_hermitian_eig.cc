// ex11_hermitian_eig.cc
// Solve Hermitian eigenvalues A = X Lambda X^H
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;

//------------------------------------------------------------------------------
template <typename T>
void test_hermitian_eig()
{
    using real_t = blas::real_type<T>;

    print_func( mpi_rank );

    // TODO: failing if n not divisible by nb?
    int64_t n=1000, nb=100, p=2, q=2;
    assert( mpi_size == p*q );
    slate::HermitianMatrix<T> A( slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );
    std::vector<real_t> Lambda( n );

    // A = X Lambda X^H, eigenvalues only
    slate::eig_vals( A, Lambda );  // simplified API

    //slate::heev( A, Lambda );      // traditional API

    // TODO: revert to above interface.
    // Empty matrix of eigenvectors.
    slate::Matrix<T> Xempty;
    slate::heev( slate::Job::NoVec, A, Lambda, Xempty );

    // TODO: eigenvectors
    // slate::Matrix<T> X( n, n, nb, p, q, MPI_COMM_WORLD );
    // X.insertLocalTiles();
    // slate::eig( A, Lambda, X );   // simplified API
    // slate::heev( A, Lambda, X );  // traditional API
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

    test_hermitian_eig< float >();
    test_hermitian_eig< double>();
    test_hermitian_eig< std::complex<float> >();
    test_hermitian_eig< std::complex<double> >();

    err = MPI_Finalize();
    assert( err == 0 );
}
