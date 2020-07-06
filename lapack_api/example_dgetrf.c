// Simple example to show usage of lapack_api
// Compile with e.g. mkl libraries
// mpicc -o example_dgetrf example_dgetrf.c -L../lib -lslate_lapack_api -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
// export LD_LIBRARY_PATH=`pwd`/../icl/slate/lib:$LD_LIBRARY_PATH
// env SLATE_LAPACK_VERBOSE=1 ./example_dgetrf

#include <stdlib.h>
#include <stdio.h>

int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
int slate_dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);

int main(int argc, char *argv[])
{
    int m, n, lda, minmn, info;
    m = n = lda = minmn = 100;
    double *A = (double *)malloc( n * m * sizeof(double) );
    int *ipiv = (int *)malloc( minmn * sizeof(int) );
    for (int j=0; j<m*n; j++)
        A[j] = 0.5 - (double)rand() / RAND_MAX;

    printf("Run LAPACK dgetrf\n");
    dgetrf_( &m, &n, A, &lda, ipiv, &info );
    printf("Run SLATE dgetrf using lapack_api\n");
    slate_dgetrf_( &m, &n, A, &lda, ipiv, &info );
    printf("Done\n");
}
