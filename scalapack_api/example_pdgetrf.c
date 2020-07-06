// Simple example to show usage of scalapack_api
// Runs on a single node (1x1 grid) using fixed sizes
// Compile with e.g. mkl libraries
// mpicc -o example_pdgetrf example_pdgetrf.c -L../lib -lslate_scalapack_api -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
// export LD_LIBRARY_PATH=`pwd`/../icl/slate/lib:$LD_LIBRARY_PATH
// env SLATE_SCALAPACK_VERBOSE=1 mpirun -n 1 ./example_pdgetrf

#include <stdlib.h>
#include <stdio.h>

void blacs_get_( int *ctxt, int *what, int *val );
void blacs_gridinit_( int *ctxt, char *layout, int *nprow, int *npcol );
void descinit_( int* desc, int* m, int* n, int* mb, int* nb,
               int* irsrc, int* icsrc, int* ictxt, int* lld, int* info );
void pdgetrf_( int* m, int* n, double* a, int* ia, int* ja, int* desca,
               int* ipiv, int* info );

int main(int argc, char *argv[])
{
    int nb=256, ia=1, ja=1, izero=0, imone=-1, p=1, q=1;
    int m, n, minmn, info, ictxt, descA[9];
    m = n = minmn = 2560;
    double *A = (double *)malloc( n * m * sizeof(double) );
    int *ipiv = (int *)malloc( minmn * sizeof(int) );
    for (int j=0; j<m*n; j++)
        A[j] = 0.5 - (double)rand() / RAND_MAX;
    blacs_get_( &imone, &izero, &ictxt );
    blacs_gridinit_( &ictxt, "Col-major", &p, &q );
    descinit_( descA, &m, &n, &nb, &nb, &izero, &izero, &ictxt, &m, &info );
    printf("Intercept ScaLAPACK calls to pdgetrf using scalapack_api\n");
    pdgetrf_( &m, &n, A, &ia, &ja, descA, ipiv, &info );
    printf("Done\n");
}
