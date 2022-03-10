// Simple example to show usage of scalapack_api
// Runs on 4 processes (2x2 grid) using fixed problem sizes
// Compile with e.g. mkl libraries and paths to the other libraries
// export LD_LIBRARY_PATH=`pwd`/../lib:`pwd`/../lapackpp/lib::`pwd`/../blaspp/lib:$CUDADIR/lib64:$MKLROOT/lib/intel64
// export RUNPATH=`pwd`/../lib:`pwd`/../lapackpp/lib::`pwd`/../blaspp/lib:$CUDADIR/lib64:$MKLROOT/lib/intel64
// mpicc -o example_pdgetrf example_pdgetrf.c -L../lib -L../lapackpp/lib -L../blaspp/lib -lslate_scalapack_api -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
// env SLATE_SCALAPACK_VERBOSE=1 mpirun -np 4 ./example_pdgetrf

// Note: Assuming Fortran add underscore name mangling for BLACS/ScaLAPACK calls

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
void blacs_get_( int *ctxt, int *what, int *val );
void blacs_gridinit_( int *ctxt, char *layout, int *nprow, int *npcol );
void blacs_gridinfo_( int *ctxt, int *nprow, int *npcol, int *myprow, int *mypcol );
int blacs_pnum_( int *ctxt, int *myprow, int *mypcol );
int numroc_( int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);
void descinit_( int* desc, int* m, int* n, int* mb, int* nb,
               int* irsrc, int* icsrc, int* ictxt, int* lld, int* info );
void pdgetrf_( int* m, int* n,
               double* a, int* ia, int* ja, int* desca, int* ipiv, int* info );
void pdgetrs_( int* transa, int* n, int* nrhs,
               double* a, int* ia, int* ja, int* desc_a, int* ipiv,
               double* b, int* ib, int* jb, int* desc_b, int* info);
#ifdef __cplusplus
}
#endif

int main(int argc, char *argv[])
{
    int m=240, n=m, nb=128, nrhs = 10;
    int nprow=2, npcol=2, myprow, mypcol;
    int info, ictxt, izero=0, imone=-1;

    blacs_get_( &imone, &izero, &ictxt );
    blacs_gridinit_( &ictxt, (char*)"Col-major", &nprow, &npcol );
    blacs_gridinfo_( &ictxt, &nprow, &npcol, &myprow, &mypcol);

    int mypnum = blacs_pnum_( &ictxt, &myprow, &mypcol );

    int ia=1, ja=1, descA[9];
    int localm = numroc_(&m, &nb, &myprow, &izero, &nprow);
    int localn = numroc_(&n, &nb, &mypcol, &izero, &npcol);
    double *A = (double *)malloc( localm * localn * sizeof(double) );
    descinit_( descA, &m, &n, &nb, &nb, &izero, &izero, &ictxt, &localm, &info );
    for (int j=0; j<localm*localn; j++)
        A[j] = 0.5 - (double)rand() / RAND_MAX;

    int minmn = (m < n) ? m : n;
    int *ipiv = (int *)malloc( minmn * sizeof(int) );

    if (mypnum==0) printf("Intercepting ScaLAPACK calls to pdgetrf using scalapack_api\n");
    pdgetrf_( &m, &n, A, &ia, &ja, descA, ipiv, &info );

    int ib=1, jb=1, descB[9], notrans = 'n';
    localm = numroc_(&m, &nb, &myprow, &izero, &nprow);
    localn = numroc_(&nrhs, &nb, &mypcol, &izero, &npcol);
    double *B = (double *)malloc( localm * localn * sizeof(double) );
    descinit_( descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &localm, &info );
    for (int j=0; j<localm*localn; j++)
        B[j] = 0.5 - (double)rand() / RAND_MAX;

    if (mypnum==1) printf("Intercepting ScaLAPACK calls to pdgetrs using scalapack_api\n");
    pdgetrs_( &notrans, &n, &nrhs, A, &ia, &ja, descA, ipiv, B, &ib, &jb, descB, &info );

    if (mypnum==1) printf("Done\n");
}
