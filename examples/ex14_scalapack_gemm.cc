// ex14_scalapack_gemm.cc
#include <mpi.h>

#include "util.hh"
#include "scalapack.h"

int mpi_size = 0;
int mpi_rank = 0;

//------------------------------------------------------------------------------
void test_pdgemm()
{
    print_func( mpi_rank );

    // constants
    int izero = 0, ione = 1;

    // problem size and distribution
    int m = 15, n = 18, k = 13, nb = 4, p = 2, q = 2;

    // initialize BLACS communication
    int p_, q_, nprocs, ictxt, iam, myrow, mycol, info;
    Cblacs_pinfo( &iam, &nprocs );
    assert( p*q <= nprocs );
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Col", p, q );
    Cblacs_gridinfo( ictxt, &p_, &q_, &myrow, &mycol );
    assert( p_ == p );
    assert( q_ == q );

    // matrix A: get local size, allocate, create descriptor, initialize
    int mlocA = numroc( &m, &nb, &myrow, &izero, &p );
    int nlocA = numroc( &k, &nb, &mycol, &izero, &q );
    int lldA  = mlocA;
    int descA[9];
    descinit( descA, &m, &k, &nb, &nb, &izero, &izero, &ictxt, &lldA, &info );
    assert( info == 0 );
    std::vector<double> dataA( lldA * nlocA );
    random_matrix( mlocA, nlocA, &dataA[0], lldA );

    // matrix B: get local size, allocate, create descriptor, initialize
    int mlocB = numroc( &k, &nb, &myrow, &izero, &p );
    int nlocB = numroc( &n, &nb, &mycol, &izero, &q );
    int lldB  = mlocB;
    int descB[9];
    descinit( descB, &k, &n, &nb, &nb, &izero, &izero, &ictxt, &lldB, &info );
    assert( info == 0 );
    std::vector<double> dataB( lldB * nlocB );
    random_matrix( mlocB, nlocB, &dataB[0], lldB );

    // matrix C: get local size, allocate, create descriptor, initialize
    int mlocC = numroc( &m, &nb, &myrow, &izero, &p );
    int nlocC = numroc( &n, &nb, &mycol, &izero, &q );
    int lldC  = mlocC;
    int descC[9];
    descinit( descC, &m, &n, &nb, &nb, &izero, &izero, &ictxt, &lldC, &info );
    assert( info == 0 );
    std::vector<double> dataC( lldC * nlocC );
    random_matrix( mlocC, nlocC, &dataC[0], lldC );

    double alpha = 2.7183;
    double beta  = 3.1415;

    // gemm: C = alpha A B + beta C
    pdgemm( "notrans", "notrans", &m, &n, &k,
            &alpha, &dataA[0], &ione, &ione, descA,
                    &dataB[0], &ione, &ione, descB,
            &beta,  &dataC[0], &ione, &ione, descC );
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

    test_pdgemm();

    err = MPI_Finalize();
    assert( err == 0 );

    return 0;
}
