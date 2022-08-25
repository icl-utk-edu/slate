// Simple example to show usage of scalapack_api
// Runs on 4 processes using fixed problem sizes
// On summit, compile and run with
// mpicxx -L./lib -Wl,-rpath,$SLATE_DIR/lib -L./testsweeper -Wl,-rpath,$SLATE_DIR/testsweeper -Wl,-rpath,$SLATE_DIR/blaspp/lib -Wl,-rpath,$SLATE_DIR/lapackpp/lib -g -fPIC -fopenmp -L./blaspp/lib -L./lapackpp/lib   scalapack_api/example_pdsgesv.c    -lslate_scalapack_api -lslate -ltestsweeper -lscalapack -lblaspp -llapackpp -L/sw/summit/cuda/11.4.2/lib64 -lessl -llapack -lcublas -lcudart    -o scalapack_api/example_pdsgesv
// bsub -P ACCOUNT -alloc_flags smt1 -nnodes 1 -W 1:00 -Is /bin/bash
// env SLATE_SCALAPACK_VERBOSE=1 jsrun -n4 -a1 -c1 -g1 -brs  ./scalapack_api/example_pdsgesv

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
void pdsgesv_(const int* n, const int* nrhs, double* a, const int* ia, const int* ja, const int* desca,
              int* ipiv, double* b, const int* ib, const int* jb, const int* descb,
              double* x, const int* ix, const int* jx, const int* descx, int* iter, int* info);
#ifdef __cplusplus
}
#endif

int main(int argc, char* argv[])
{
    int nb=200, n=nb*4, nrhs=1;
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
        A[j] = 0.5 - (double)rand() / RAND_MAX;

    // Get norm of original A
    const char onenorm = '1';
    double* work = (double*)malloc(n*sizeof(double)); // todo: fix size
    double Anorm = pdlange_(&onenorm, &n, &n, A, &ione, &ione, descA, work);
    // printf("Anorm = %f\n", Anorm); fflush(0);

    // Use this to print a matrix if desired
    // for (int i=0; i<localma; i++)
    //     for (int j=0; j<localna; j++)
    //         printf("A[%d,%d][%d,%d] = %f\n", myprow, mypcol, i, j, A[j*localma+i]); fflush(0);

    // Allocate space for pivots
    int* ipiv = (int*)malloc(n * sizeof(int));

    // Setup B
    if (mypnum == 0) { printf("Setup B to be %d x %d\n", n, nrhs); fflush(0); }
    int descB[9];
    int localmb = numroc_(&n, &nb, &myprow, &izero, &nprow);
    int localnb = numroc_(&nrhs, &nb, &mypcol, &izero, &npcol);
    double* B = (double*)malloc(localmb * localnb * sizeof(double));
    descinit_(descB, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &localmb, &info);
    for (int j=0; j<localmb*localnb; j++)
        B[j] = 0.5 - (double)rand() / RAND_MAX;

    // Allocate X (same as B)
    if (mypnum == 0) { printf("Setup X to be %d x %d\n", n, nrhs); fflush(0); }
    int descX[9];
    double* X = (double*)malloc(localmb * localnb * sizeof(double));
    descinit_(descX, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &localmb, &info);
    for (int j=0; j<localmb*localnb; j++)
        X[j] = 0;

    // Save B for checking
    int descBref[9];
    double* Bref = (double*)malloc(localmb * localnb * sizeof(double));
    descinit_(descBref, &n, &nrhs, &nb, &nb, &izero, &izero, &ictxt, &localmb, &info);
    for (int j=0; j<localmb*localnb; j++)
        Bref[j] = B[j];

    // Use this to print a matrix if desired
    // for (int i=0; i<localmb; i++)
    //     for (int j=0; j<localnb; j++)
    //         printf("B[%d,%d][%d,%d] = %f\n", myprow, mypcol, i, j, B[j*localmb+i]); fflush(0);

    int iters = 0;

    // Solve for x, Ax=B, storing x in B
    if (mypnum == 0) { printf("Calling pdsgesv_ to solve for x, Ax=B\n"); fflush(0); }
    pdsgesv_(&n, &nrhs, A, &ione, &ione, descA, ipiv, B, &ione, &ione, descB,
             X, &ione, &ione, descX, &iters, &info);

    if (mypnum == 0) { printf("pdsgesv_ completed in %d iterations\n", iters); fflush(0); }

    // Use this to print a matrix if desired
    // for (int i=0; i<localmb; i++)
    //     for (int j=0; j<localnb; j++)
    //         printf("X[%d,%d][%d,%d] = %f\n", myprow, mypcol, i, j, X[j*localmb+i]); fflush(0);

    // Get norm of the solution stored in X
    double Xnorm = pdlange_(&onenorm, &n, &nrhs, X, &ione, &ione, descX, work);
    // printf("Xnorm = %f\n", Xnorm); fflush(0);

    // Reset the contants of matrix A
    srand(srand_seed);
    for (int j=0; j<localma*localna; j++)
        A[j] = 0.5 - (double)rand() / RAND_MAX;

    if (mypnum == 0) { printf("Calling pdgemm_ for B_orig - A_orig * X to check results\n"); fflush(0); }

    // Calculate Bref - Ax using pdgemm
    const double alpha=-1.0, beta=1.0;
    const char notrans='n';
    pdgemm_(
        &notrans, &notrans,
        &n, &nrhs, &n,
        &alpha,
        A, &ione, &ione, descA,
        X, &ione, &ione, descX,
        &beta,
        Bref, &ione, &ione, descBref);

    // Get norm of residual from the dgemm above ( Bref - Ax )
    double Rnorm = pdlange_(&onenorm, &n, &nrhs, Bref, &ione, &ione, descBref, work);
    double residual = Rnorm / (n * Anorm * Xnorm);
    if (mypnum == 0) printf("residual = (%g / (%d * %g * %g)) = %g\n", Rnorm, n, Anorm, Xnorm, residual);

    // Check if Bref - Ax is (near)zero
    // if (mypnum == 0) printf("Printing if any A_orig * x - B_orig > 1e-9 ...\n");
    // for (int i=0; i<localmb; i++)
    //     for (int j=0; j<localnb; j++)
    //         if (Bref[j*localmb+i] > 1e-9)
    //             printf("problem at B[%d,%d][%d,%d] = %g > 1e-9\n", myprow, mypcol, i, j, Bref[j*localmb+i]); fflush(0);

    if (mypnum == 0) printf("Done\n");
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Finalize();
}
