// ex03_submatrix.cc
// A.sub and A.slice

/// !!!   Lines between `//---------- begin label`          !!!
/// !!!             and `//---------- end label`            !!!
/// !!!   are included in the SLATE Users' Guide.           !!!

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
    int64_t i1=1, i2=3, j1=2, j2=3;
    int64_t row1=100, row2=300, col1=200, col2=400;

    slate::Matrix<scalar_type>
        A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    printf( "rank %d: A mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(A.mt()), llong(A.nt()), llong(A.m()), llong(A.n()) );

    //---------------------------------------- sub-matrix

    //---------- begin sub1
    // view of A( i1 : i2, j1 : j2 ) as tile indices, inclusive
    auto B = A.sub( i1, i2, j1, j2 );
    //---------- end sub1
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------- begin sub2

    // view of all of A
    B = A;
    //---------- end sub2
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------- begin sub3

    // same, view of all of A
    B = A.sub( 0, A.mt()-1, 0, A.nt()-1 );
    //---------- end sub3
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------- begin sub4

    // view of first block-column, A[ 0:mt-1, 0:0 ] as tile indices
    B = A.sub( 0, A.mt()-1, 0, 0 );
    //---------- end sub4
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------- begin sub5

    // view of first block-row, A[ 0:0, 0:nt-1 ] as tile indices
    B = A.sub( 0, 0, 0, A.nt()-1 );
    //---------- end sub5
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------------------------------------- slicing

    //---------- begin slice1
    // view of A( row1 : row2, col1 : col2 ), inclusive
    B = A.slice( row1, row2, col1, col2 );
    //---------- end slice1
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------- begin slice2

    // view of all of A
    B = A.slice( 0, A.m()-1, 0, A.n()-1 );
    //---------- end slice2
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------- begin slice3

    // view of first column, A[ 0:m-1, 0:0 ]
    B = A.slice( 0, A.m()-1, 0, 0 );
    //---------- end slice3
    printf( "rank %d: B mt=%lld, nt=%lld, m=%lld, n=%lld\n",
            mpi_rank, llong(B.mt()), llong(B.nt()), llong(B.m()), llong(B.n()) );

    //---------- begin slice4

    // view of first row, A[ 0:0, 0:n-1 ]
    B = A.slice( 0, 0, 0, A.n()-1 );
    //---------- end slice4
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
