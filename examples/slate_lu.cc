/// !!!   Getting Started example in SLATE Users' Guide.    !!!
/// !!!   NOTE: if lines are added or deleted here,         !!!
/// !!!   change line numbers in getting_started.tex        !!!

#include <slate/slate.hh>
#include <blas.hh>
#include <mpi.h>
#include <stdio.h>

//------------------------------------------------------------------------------
// put random data in matrix A
template <typename matrix_type>
void random_matrix( matrix_type& A )
{
    // for each tile in the matrix
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                // set data-values in the local tile
                auto tile = A( i, j );
                auto tiledata = tile.data();
                for (int64_t jj = 0; jj < tile.nb(); ++jj) {
                    for (int64_t ii = 0; ii < tile.mb(); ++ii) {
                        tiledata[ ii + jj*tile.stride() ] =
                            (1.0 - (rand() / double(RAND_MAX))) * 100;
                    }
                }
            }
        }
    }
}


//------------------------------------------------------------------------------
// create matrices, call LU solver, check result
template <typename scalar_t>
void lu_example( int64_t n, int64_t nrhs, int64_t nb, int64_t p, int64_t q )
{
    // Get associated real type, e.g., double for complex<double>.
    using real_t = blas::real_type<scalar_t>;
    using llong = long long;  // guaranteed >= 64 bits
    const scalar_t one = 1;
    int err=0, mpi_size=0, mpi_rank=0;

    // Get MPI size, must be p*q for this example
    err = MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
    assert( err == 0 );
    if (mpi_size != p*q) {
        printf( "Usage: mpirun -np %lld ... # %lld ranks hard coded\n",
                llong( p*q ), llong( p*q ) );
        assert( mpi_size == p*q );
    }

    // Get MPI rank
    err = MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    assert( err == 0 );

    // Create SLATE matrices A and B
    slate::Matrix<scalar_t> A( n, n,    nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<scalar_t> B( n, nrhs, nb, p, q, MPI_COMM_WORLD );

    // Allocate local space for A, B on distributed nodes
    A.insertLocalTiles();
    B.insertLocalTiles();

    // Set random seed so data is different on each rank
    srand( 100 * mpi_rank );
    // Initialize the data for A, B
    random_matrix( A );
    random_matrix( B );

    // For residual error check,
    // create A0 as an empty matrix like A and copy A to A0
    slate::Matrix<scalar_t> A0 = A.emptyLike();
    A0.insertLocalTiles();
    slate::copy( A, A0 );
    // Create B0 as an empty matrix like B and copy B to B0
    slate::Matrix<scalar_t> B0 = B.emptyLike();
    B0.insertLocalTiles();
    slate::copy( B, B0 );

    // Call the SLATE LU solver
    slate::Pivots pivots;
    slate::Options opts = {{slate::Option::Target, slate::Target::HostTask}};
    double time = omp_get_wtime();
    slate::gesv( A, pivots, B, opts);
    time = omp_get_wtime() - time;

    // Compute residual ||A0 * X  - B0|| / ( ||X|| * ||A|| * N )
    real_t A_norm = slate::norm( slate::Norm::One, A );
    real_t X_norm = slate::norm( slate::Norm::One, B );
    slate::gemm( -one, A0, B, one, B0 );
    real_t R_norm = slate::norm( slate::Norm::One, B0 );
    real_t residual = R_norm / (X_norm * A_norm * n);
    real_t tol = std::numeric_limits<real_t>::epsilon();
    int status_ok = (residual < tol);

    if (mpi_rank == 0) {
        printf("gesv n %lld nb %lld pxq %lldx%lld "
               "residual %.2e tol %.2e sec %.2e ",
               llong( n ), llong( nb ), llong( p ), llong( q ),
               residual, tol, time);
        if (status_ok) printf("PASS\n");
        else printf("FAILED\n");
    }
}


//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    // Initialize MPI, require MPI_THREAD_MULTIPLE support
    int err=0, mpi_provided=0;
    err = MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided );
    assert( err == 0 && mpi_provided == MPI_THREAD_MULTIPLE );

    // Call the LU example
    int64_t n=5000, nrhs=1, nb=256, p=2, q=2;
    lu_example< double >( n, nrhs, nb, p, q );

    err = MPI_Finalize();
    assert( err == 0 );
}
