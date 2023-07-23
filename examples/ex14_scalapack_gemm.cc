// ex14_scalapack_gemm.cc
#include <mpi.h>

#include "util.hh"
#include "scalapack.h"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
// We don't include slate.hh here, so define a simple slate_mpi_call.
void slate_mpi_call_( int err, const char* file, int line )
{
    if (err != 0) {
        char msg[ 80 ];
        snprintf( msg, sizeof(msg), "MPI error %d at %s:%d", err, file, line );
        throw std::runtime_error( msg );
    }
}

#define slate_mpi_call( err ) \
        slate_mpi_call_( err, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_pgemm()
{
    print_func( mpi_rank );

    // constants
    int izero = 0, ione = 1;

    // problem size and distribution
    int m = 15, n = 18, k = 13, nb = 4;

    // initialize BLACS communication
    int p_, q_, nprocs, ictxt, iam, myrow, mycol, info;
    Cblacs_pinfo( &iam, &nprocs );
    assert( grid_p * grid_q <= nprocs );
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Col", grid_p, grid_q );
    Cblacs_gridinfo( ictxt, &p_, &q_, &myrow, &mycol );
    assert( p_ == grid_p );
    assert( q_ == grid_q );

    // matrix A: get local size, allocate, create descriptor, initialize
    int mlocA = numroc( &m, &nb, &myrow, &izero, &grid_p );
    int nlocA = numroc( &k, &nb, &mycol, &izero, &grid_q );
    int lldA  = mlocA;
    int descA[9];
    descinit( descA, &m, &k, &nb, &nb, &izero, &izero, &ictxt, &lldA, &info );
    assert( info == 0 );
    std::vector<scalar_type> dataA( lldA * nlocA );
    random_matrix( mlocA, nlocA, &dataA[0], lldA );

    // matrix B: get local size, allocate, create descriptor, initialize
    int mlocB = numroc( &k, &nb, &myrow, &izero, &grid_p );
    int nlocB = numroc( &n, &nb, &mycol, &izero, &grid_q );
    int lldB  = mlocB;
    int descB[9];
    descinit( descB, &k, &n, &nb, &nb, &izero, &izero, &ictxt, &lldB, &info );
    assert( info == 0 );
    std::vector<scalar_type> dataB( lldB * nlocB );
    random_matrix( mlocB, nlocB, &dataB[0], lldB );

    // matrix C: get local size, allocate, create descriptor, initialize
    int mlocC = numroc( &m, &nb, &myrow, &izero, &grid_p );
    int nlocC = numroc( &n, &nb, &mycol, &izero, &grid_q );
    int lldC  = mlocC;
    int descC[9];
    descinit( descC, &m, &n, &nb, &nb, &izero, &izero, &ictxt, &lldC, &info );
    assert( info == 0 );
    std::vector<scalar_type> dataC( lldC * nlocC );
    random_matrix( mlocC, nlocC, &dataC[0], lldC );

    scalar_type alpha = 2.7183;
    scalar_type beta  = 3.1415;

    // gemm: C = alpha A B + beta C
    if constexpr (std::is_same< scalar_type, float >::value) {
        psgemm( "notrans", "notrans", &m, &n, &k,
                &alpha, &dataA[0], &ione, &ione, descA,
                        &dataB[0], &ione, &ione, descB,
                &beta,  &dataC[0], &ione, &ione, descC );
    }
    else if constexpr (std::is_same< scalar_type, double >::value) {
        pdgemm( "notrans", "notrans", &m, &n, &k,
                &alpha, &dataA[0], &ione, &ione, descA,
                        &dataB[0], &ione, &ione, descB,
                &beta,  &dataC[0], &ione, &ione, descC );
    }
    else if constexpr (std::is_same< scalar_type, std::complex<float> >::value) {
        pcgemm( "notrans", "notrans", &m, &n, &k,
                &alpha, &dataA[0], &ione, &ione, descA,
                        &dataB[0], &ione, &ione, descB,
                &beta,  &dataC[0], &ione, &ione, descC );
    }
    else if constexpr (std::is_same< scalar_type, std::complex<double> >::value) {
        pzgemm( "notrans", "notrans", &m, &n, &k,
                &alpha, &dataA[0], &ione, &ione, descA,
                        &dataB[0], &ione, &ione, descB,
                &beta,  &dataC[0], &ione, &ione, descC );
    }
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
            test_pgemm< float >();
        }

        if (types[ 1 ]) {
            test_pgemm< double >();
        }

        if (types[ 2 ]) {
            test_pgemm< std::complex<float> >();
        }

        if (types[ 3 ]) {
            test_pgemm< std::complex<double> >();
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
