
#include "Slate_Matrix.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <omp.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>

extern "C" void trace_off();
extern "C" void trace_on();
void print_lapack_matrix(int m, int n, double *a, int lda, int mb, int nb);

//------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    assert(argc == 3);
    int nb = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int n = nb*nt;
    int lda = n;

    //------------------------------------------------------
    double *a1 = (double*)malloc(sizeof(double)*nb*nb*nt*nt);
    assert(a1 != nullptr);

    int seed[] = {0, 0, 0, 1};
    int retval;
    retval = LAPACKE_dlarnv(1, seed, (size_t)lda*n, a1);
    assert(retval == 0);

    for (int i = 0; i < n; ++i)
        a1[i*lda+i] += sqrt(n);

    //------------------------------------------------------

    double *a2 = (double*)malloc(sizeof(double)*nb*nb*nt*nt);
    assert(a2 != nullptr);

    memcpy(a2, a1, sizeof(double)*lda*n);

    //------------------------------------------------------

    trace_off();
    Slate::Matrix<double> temp(n, n, a1, lda, nb, nb);
    temp.potrf(blas::Uplo::Lower);
    trace_on();

    Slate::Matrix<double> a(n, n, a1, lda, nb, nb);
    double start = omp_get_wtime();
    a.potrf(blas::Uplo::Lower);
    double time = omp_get_wtime()-start;
    a.copyFrom(n, n, a1, lda, nb, nb);

    retval = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a2, lda);
    assert(retval == 0);

    // print_lapack_matrix(n, n, a1, lda, nb, nb);
    // print_lapack_matrix(n, n, a2, lda, nb, nb);

    //------------------------------------------------------

    cblas_daxpy((size_t)lda*n, -1.0, a1, 1, a2, 1);

    double norm = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, a1, lda);
    double error = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, a2, lda);
    if (norm != 0)
        error /= norm;
    printf("\t%le\n", error);

    double gflops = (double)nb*nb*nb*nt*nt*nt/3.0/time/1000000000.0;
    printf("\t%.0lf GFLOPS\n", gflops);

    free(a1);
    free(a2);
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
void print_lapack_matrix(int m, int n, double *a, int lda, int mb, int nb)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%8.2lf", a[(size_t)lda*j+i]);
            if ((j+1)%nb == 0)
                printf(" |");
        }
        printf("\n");
        if ((i+1)%mb == 0) {
            for (int j = 0; j < (n+1)*8; ++j) {
                printf("-");
            }
            printf("\n");        
        }
    }
    printf("\n");
}
