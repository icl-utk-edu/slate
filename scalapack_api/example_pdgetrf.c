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
#include <assert.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
void blacs_get_(const int* ctxt, const int* what, int* val);
void blacs_gridinit_(const int* ctxt, const char* layout, const int* nprow, const int* npcol);
void blacs_gridinfo_(const int* ctxt, int* nprow, int* npcol, int* myprow, int* mypcol);
int blacs_pnum_(const int* ctxt, const int* myprow, const int* mypcol);
int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, int* nprocs);
void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
               const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);
void pdgetrf_(const int* m, const int* n,
              double* a, const int* ia, const int* ja, const int* desca, const int* ipiv, int* info);
void pdgetrs_(const char* transa, const int* n, const int* nrhs,
              double* a, const int* ia, const int* ja, const int* desc_a, const int* ipiv,
              double* b, const int* ib, const int* jb, const int* desc_b, int* info);
void pdgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
             const double* alpha,
             double* a, const int* ia, const int* ja, const int* desc_a,
             double* b, const int* ib, const int* jb, const int* desc_b,
             const double* beta,
             double* c, const int* ic, const int* jc, const int* desc_c);
double pdlange_(const char* norm, const int* m, const int* n,
                double* a, const int* ia, const int* ja, const int* desca, double* work);
#ifdef __cplusplus
}
#endif

int main(int argc, char* argv[])
{
    int nb=200; // tile size
    int n=nb*4+17; // the matrix is not a fixed multiple of tiles
    int nrhs=2;
    int nprow=1, npcol=4, myprow, mypcol;
    int info, ictxt, izero=0, imone=-1, ione=1;

    // Initialize MPI and request multi-threaded support.  NOTE: The
    // multi-threaded support is required, and simply setting up BLACS
    // is not sufficient.
    int mpi_thread_level = 0, mpi_rank, mpi_size;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_thread_level);
    assert(mpi_thread_level >= MPI_THREAD_MULTIPLE);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Setup BLACS
    if (mpi_rank == 0) { printf("Setup BLACS as a row-major %d x %d (nprow x npcol) process grid \n", nprow, npcol); fflush(0); }
    blacs_get_(&imone, &izero, &ictxt);
    blacs_gridinit_(&ictxt, "row-major", &nprow, &npcol);
    blacs_gridinfo_(&ictxt, &nprow, &npcol, &myprow, &mypcol);
    int mypnum = blacs_pnum_(&ictxt, &myprow, &mypcol);

    // Setup A
    if (mypnum == 0) { printf("Setup A to be a %d x %d matrix\n", n, n); fflush(0); }
    int descA[9];
    int localma = numroc_(&n, &nb, &myprow, &izero, &nprow);
    int localna = numroc_(&n, &nb, &mypcol, &izero, &npcol);
    double* A = (double*)malloc(localma * localna * sizeof(double));
    descinit_(descA, &n, &n, &nb, &nb, &izero, &izero, &ictxt, &localma, &info);
    // seed random generator (do not use 1 => reset)
    int srand_seed = mpi_rank*13+17;
    srand(srand_seed);
    for (int j=0; j<localma*localna; j++)
        A[j] = 1+ 0.5 - (double)rand() / RAND_MAX;

    // Allocate space for pivots
    int* ipiv = (int*)malloc(n * sizeof(int));

    // LU factor A
    if (mypnum == 0) { printf("Calling pdgetrf_ to factor matrix A\n"); fflush(0); }
    pdgetrf_(&n, &n, A, &ione, &ione, descA, ipiv, &info);

    // Setup B
    if (mypnum == 0) { printf("Setup B to be %d x %d\n", n, nrhs); fflush(0); }
    int descB[9];
    int localmb = numroc_(&n, &nb, &myprow, &izero, &nprow);
    int localnb = numroc_(&nrhs, &nb, &mypcol, &izero, &npcol);
    double* B = (double*)malloc(localmb * localnb * sizeof(double));
    descinit_(descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &localmb, &info);
    for (int j=0; j<localmb*localnb; j++)
        B[j] = 0.5 - (double)rand() / RAND_MAX;

    // Save B for checking
    int descB_sav[9];
    double* B_sav = (double*)malloc(localmb * localnb * sizeof(double));
    descinit_(descB_sav, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &localmb, &info);
    for (int j=0; j<localmb*localnb; j++)
        B_sav[j] = B[j];

    // Use this to print a matrix if desired
    // for (int i=0; i<localmb; i++)
    //     for (int j=0; j<localnb; j++)
    //         printf("B[%d,%d][%d,%d] = %f\n", myprow, mypcol, i, j, B[j*localmb+i]); fflush(0);

    // Solve for x, Ax=B, storing x in B
    if (mypnum == 0) { printf("Calling pdgetrs_ to solve for x, Ax=B\n"); fflush(0); }
    const char notrans='n';
    pdgetrs_(&notrans, &n, &nrhs, A, &ione, &ione, descA, ipiv, B, &ione, &ione, descB, &info);

    // Get norm of X
    const char maxnorm = 'm';
    double* work = (double*)malloc(1);
    double Xnorm = pdlange_(&maxnorm, &n, &nrhs, B, &ione, &ione, descB, work);
    // printf("Xnorm = %f\n", Xnorm); fflush(0);

    // Reset the contants of matrix A
    srand(srand_seed);
    for (int j=0; j<localma*localna; j++)
        A[j] = 1+ 0.5 - (double)rand() / RAND_MAX;

    // Get norm of original A
    double Anorm = pdlange_(&maxnorm, &n, &n, A, &ione, &ione, descA, work);
    // printf("Anorm = %f\n", Anorm); fflush(0);

    if (mypnum == 0) { printf("Calling pdgemm_ for a (simple) check if A_orig * x - B_orig is (nearly) 0\n"); fflush(0); }

    // Calculate B_sav - Ax using pdgemm
    const double alpha=-1.0, beta=1.0;
    pdgemm_(
        &notrans, &notrans,
        &n, &nrhs, &n,
        &alpha,
        A, &ione, &ione, descA,
        B, &ione, &ione, descB,
        &beta,
        B_sav, &ione, &ione, descB_sav);

    // Get norm of residual from the dgemm above ( B_sav - Ax )
    double Rnorm = pdlange_(&maxnorm, &n, &nrhs, B_sav, &ione, &ione, descB_sav, work);
    double residual = Rnorm / (n * Anorm * Xnorm);
    if (mypnum == 0) printf("residual = (%g / (%d * %g * %g)) = %g\n", Rnorm, n, Anorm, Xnorm, residual);

    // Check if B_sav - Ax is (near)zero
    if (mypnum == 0) printf("Printing if any A_orig * x - B_orig > 1e-9 ...\n");
    for (int i=0; i<localmb; i++) {
        for (int j=0; j<localnb; j++) {
            if (B_sav[j*localmb+i] > 1e-9) {
                printf("problem at B[%d,%d][%d,%d] = %g > 1e-9\n", myprow, mypcol, i, j, B_sav[j*localmb+i]); fflush(0);
            }
        }
    }

    if (mypnum == 0) printf("Done\n");
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Finalize();
}
