// ex03_submatrix.cc
// A.sub and A.slice
#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_submatrix()
{
    using llong = long long;

    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    slate::Matrix<scalar_type>
        A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
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
            test_submatrix< float >();
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
