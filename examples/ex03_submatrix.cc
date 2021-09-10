// ex03_submatrix.cc
// A.sub and A.slice
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;

// -----------------------------------------------------------------------------
template <typename scalar_type>
void test_submatrix()
{
    using llong = long long;

    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256, p=2, q=2;
    assert( mpi_size == p*q );
    slate::Matrix<scalar_type>
        A( m, n, nb, p, q, MPI_COMM_WORLD );
    printf( "rank %d: A mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(A.mt()), llong(A.nt()), llong(A.m()), llong(A.n()) );

    // --------------------
    // shallow copy of all of A
    auto B = A;
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    // same
    B = A.sub( 0, A.mt()-1, 0, A.nt()-1 );
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    // first block-column, A[ 0:mt-1, 0:0 ] as tile indices
    B = A.sub( 0, A.mt()-1, 0, 0 );
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    // first block-row, A[ 0:0, 0:nt-1 ] as tile indices
    B = A.sub( 0, 0, 0, A.nt()-1 );
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    // --------------------
    // shallow copy of all of A
    B = A.slice( 0, A.m()-1, 0, A.n()-1 );
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    // first column, A[ 0:m-1, 0:0 ]
    B = A.slice( 0, A.m()-1, 0, 0 );
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    // first row, A[ 0:0, 0:n-1 ]
    B = A.slice( 0, 0, 0, A.n()-1 );
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );
}

// -----------------------------------------------------------------------------
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

    test_submatrix< float >();

    err = MPI_Finalize();
    assert( err == 0 );
}
