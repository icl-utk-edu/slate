// ex04_norm.cc
// BLAS routines
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_general_norm()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );

    blas::real_type<scalar_type> A_norm;

    A_norm = slate::norm( slate::Norm::One, A );
    printf( "rank %d: one norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Inf, A );
    printf( "rank %d: inf norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Max, A );
    printf( "rank %d: max norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Fro, A );
    printf( "rank %d: fro norm %.6f\n", mpi_rank, A_norm );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_symmetric_norm()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    slate::SymmetricMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );

    blas::real_type<scalar_type> A_norm;

    A_norm = slate::norm( slate::Norm::One, A );
    printf( "rank %d: one norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Inf, A );
    printf( "rank %d: inf norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Max, A );
    printf( "rank %d: max norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Fro, A );
    printf( "rank %d: fro norm %.6f\n", mpi_rank, A_norm );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_hermitian_norm()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );

    blas::real_type<scalar_type> A_norm;

    A_norm = slate::norm( slate::Norm::One, A );
    printf( "rank %d: one norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Inf, A );
    printf( "rank %d: inf norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Max, A );
    printf( "rank %d: max norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Fro, A );
    printf( "rank %d: fro norm %.6f\n", mpi_rank, A_norm );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_triangular_norm()
{
    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    slate::TriangularMatrix<scalar_type>
        A( slate::Uplo::Lower, slate::Diag::NonUnit, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );

    blas::real_type<scalar_type> A_norm;

    A_norm = slate::norm( slate::Norm::One, A );
    printf( "rank %d: one norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Inf, A );
    printf( "rank %d: inf norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Max, A );
    printf( "rank %d: max norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Fro, A );
    printf( "rank %d: fro norm %.6f\n", mpi_rank, A_norm );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_trapezoid_norm()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    slate::TrapezoidMatrix<scalar_type>
        A( slate::Uplo::Lower, slate::Diag::NonUnit, m, n, nb,
           grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    random_matrix( A );

    blas::real_type<scalar_type> A_norm;

    A_norm = slate::norm( slate::Norm::One, A );
    printf( "rank %d: one norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Inf, A );
    printf( "rank %d: inf norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Max, A );
    printf( "rank %d: max norm %.6f\n", mpi_rank, A_norm );

    A_norm = slate::norm( slate::Norm::Fro, A );
    printf( "rank %d: fro norm %.6f\n", mpi_rank, A_norm );
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

    test_general_norm< float >();
    test_general_norm< double >();
    test_general_norm< std::complex<float> >();
    test_general_norm< std::complex<double> >();

    test_symmetric_norm< float >();
    test_symmetric_norm< double >();
    test_symmetric_norm< std::complex<float> >();
    test_symmetric_norm< std::complex<double> >();

    test_hermitian_norm< float >();
    test_hermitian_norm< double >();
    test_hermitian_norm< std::complex<float> >();
    test_hermitian_norm< std::complex<double> >();

    test_triangular_norm< float >();
    test_triangular_norm< double >();
    test_triangular_norm< std::complex<float> >();
    test_triangular_norm< std::complex<double> >();

    test_trapezoid_norm< float >();
    test_trapezoid_norm< double >();
    test_trapezoid_norm< std::complex<float> >();
    test_trapezoid_norm< std::complex<double> >();

    slate_mpi_call(
        MPI_Finalize() );

    return 0;
}
