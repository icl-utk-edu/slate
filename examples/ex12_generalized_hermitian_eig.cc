// ex12_generalized_hermitian_eig.cc
// Solve generalized Hermitian eigenvalues, types:
// 1. A = B X Lambda X^H
// 2. A B = X Lambda X^H
// 3. B A = X Lambda X^H
// where B is Hermitian positive definite.
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
    slate::HermitianMatrix<T> B( slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix_diag_dominant( B );
    std::vector<real_t> Lambda( n );

    // A = B X Lambda X^H, eigenvalues only
    slate::eig_vals( 1, A, B, Lambda );  // simplified API

    //slate::hegv( 1, A, B, Lambda );      // traditional API

    // TODO: revert to above interface.
    // Empty matrix of eigenvectors.
    slate::Matrix<T> Xempty;
    slate::hegv( 1, slate::Job::NoVec, A, B, Lambda, Xempty );

    // TODO: eigenvectors
    // slate::Matrix<T> X( n, n, nb, p, q, MPI_COMM_WORLD );
    // X.insertLocalTiles();
    // slate::eig( 1, A, B, Lambda, X );  // simplified API
    // slate::hegv( A, B, Lambda, X );    // traditional API
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
