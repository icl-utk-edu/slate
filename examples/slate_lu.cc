// slate_lu.cc
// Getting started

/// !!!   Lines between `//---------- begin label`          !!!
/// !!!             and `//---------- end label`            !!!
/// !!!   are included in the SLATE Users' Guide.           !!!

//---------- begin sec1
#include <slate/slate.hh>
#include <blas.hh>
#include <mpi.h>
#include <stdio.h>

// Forward function declarations
template <typename scalar_type>
void lu_example( int64_t n, int64_t nrhs, int64_t nb, int p, int q );

template <typename matrix_type>
void random_matrix( matrix_type& A );

int main( int argc, char** argv )
{
    // Initialize MPI, requiring MPI_THREAD_MULTIPLE support.
    int err=0, mpi_provided=0;
    err = MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided );
    if (err != 0 || mpi_provided != MPI_THREAD_MULTIPLE) {
        throw std::runtime_error( "MPI_Init failed" );
    }

    // Call the LU example.
    int64_t n=5000, nrhs=1, nb=256, p=2, q=2;
    lu_example<double>( n, nrhs, nb, p, q );

    err = MPI_Finalize();
    if (err != 0) {
        throw std::runtime_error( "MPI_Finalize failed" );
    }
    return 0;
}

//---------- end sec1
//---------- begin sec2
// Create matrices, call LU solver, and check result.
template <typename scalar_t>
void lu_example( int64_t n, int64_t nrhs, int64_t nb, int p, int q )
{
    // Get associated real type, e.g., double for complex<double>.
    using real_t = blas::real_type<scalar_t>;
    using llong = long long;  // guaranteed >= 64 bits
    const scalar_t one = 1;
    int err=0, mpi_size=0, mpi_rank=0;

    // Get MPI size. Must be >= p*q for this example.
    err = MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
    if (err != 0) {
        throw std::runtime_error( "MPI_Comm_size failed" );
    }
    if (mpi_size < p*q) {
        printf( "Usage: mpirun -np %d ... # %d ranks hard coded\n",
                p*q, p*q );
        return;
    }

    // Get MPI rank
    err = MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    if (err != 0) {
        throw std::runtime_error( "MPI_Comm_rank failed" );
    }

    // Create SLATE matrices A and B. /* \label{line:lu-AB} */
    slate::Matrix<scalar_t> A( n, n,    nb, p, q, MPI_COMM_WORLD );
    slate::Matrix<scalar_t> B( n, nrhs, nb, p, q, MPI_COMM_WORLD );

    // Allocate local space for A, B on distributed nodes. /* \label{line:lu-insert} */
    A.insertLocalTiles();
    B.insertLocalTiles();

    // Set random seed so data is different on each MPI rank.
    srand( 100 * mpi_rank );
    // Initialize the data for A, B. /* \label{line:lu-rand} */
    random_matrix( A );
    random_matrix( B );

    // For residual error check,
    // create A0 as an empty matrix like A and copy A to A0.
    slate::Matrix<scalar_t> A0 = A.emptyLike();
    A0.insertLocalTiles();
    slate::copy( A, A0 );  /* \label{line:lu-copy} */
    // Create B0 as an empty matrix like B and copy B to B0.
    slate::Matrix<scalar_t> B0 = B.emptyLike();
    B0.insertLocalTiles();
    slate::copy( B, B0 );

//---------- end sec2
//---------- begin sec3
    // Call the SLATE LU solver.
    slate::Options opts = {  /* \label{line:lu-opts} */
        {slate::Option::Target, slate::Target::HostTask}
    };
    double time = omp_get_wtime();
    slate::lu_solve( A, B, opts );  /* \label{line:lu-solve} */
    time = omp_get_wtime() - time;

    // Compute residual ||A0 * X  - B0|| / ( ||X|| * ||A0|| * n )  /* \label{line:lu-residual} */
    real_t A_norm = slate::norm( slate::Norm::One, A0 );
    real_t X_norm = slate::norm( slate::Norm::One, B );
    slate::gemm( -one, A0, B, one, B0 );
    real_t R_norm = slate::norm( slate::Norm::One, B0 );
    real_t residual = R_norm / (X_norm * A_norm * n);
    real_t tol = std::numeric_limits<real_t>::epsilon();
    bool status_ok = (residual < tol);

    if (mpi_rank == 0) {
        printf( "lu_solve n %lld, nb %lld, p-by-q %lld-by-%lld, "
                "residual %.2e, tol %.2e, time %.2e sec, %s\n",
                llong( n ), llong( nb ), llong( p ), llong( q ),
                residual, tol, time,
                status_ok ? "pass" : "FAILED" );
    }
}

// Put random data in matrix A.
// todo: replace with:
//      auto rand_entry = []( int64_t i, int64_t j ) {
//          return 1.0 - rand() / double( RAND_MAX );
//      }
//      set( rand_entry, A );
template <typename matrix_type>
void random_matrix( matrix_type& A )
{
    // For each tile in the matrix
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                // set data values in the local tile.
                auto tile = A( i, j );
                auto tiledata = tile.data();
                for (int64_t jj = 0; jj < tile.nb(); ++jj) {
                    for (int64_t ii = 0; ii < tile.mb(); ++ii) {
                        tiledata[ ii + jj*tile.stride() ]
                            = 1.0 - (rand() / double(RAND_MAX));
                    }
                }
            }
        }
    }
}
//---------- end sec3
