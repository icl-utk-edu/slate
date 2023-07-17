// Simple example to show usage of lapack_api
// Compile with e.g. mkl libraries
// mpicc -o example_dgetrf example_dgetrf.c -L`pwd`/../lib  -Wl,-rpath,`pwd`/../lib -lslate_lapack_api -lslate -llapackpp -lblaspp -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
// env SLATE_LAPACK_VERBOSE=1 ./example_dgetrf
// Using slurm: salloc -N 1 -n 1 -wa[01] env SLATE_LAPACK_VERBOSE=1 mpirun -n 1 ./example_dgetrf


#include <stdlib.h>
#include <stdio.h>

//------------------------------------------------------------------------------

/// Print an LAPACK matrix.
void print_matrix(
    const char* label,
    int64_t m, int64_t n, double* A, int64_t lda,
    int width, int precision)
{
    char buf[ 1024 ];
    printf("%s = [\n", label);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            snprintf(buf, sizeof(buf), " %*.*f", width, precision, A[i + j*lda]);
            printf("%s", buf);
        }
        printf( "\n");
    }
    printf("];\n");
}

//------------------------------------------------------------------------------

int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
int slate_dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);

int dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
int slate_dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);

//------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int m, n, lda, min_mn, info;
    m = n = lda = min_mn = 4;
    double *A = (double *)malloc( n * m * sizeof(double) );
    int *ipiv = (int *)malloc( min_mn * sizeof(int) );

    printf("Setup matrix A\n");
    for (int j=0; j<n; ++j)
        for (int i=0; i<m; ++i)
            A[i+j*n] = i+j+1;
    for (int j=0; j<n; ++j)
        A[j+j*n] += n; // diag dom
    print_matrix("A", m, n, A, lda, 10, 5);

    printf("Run LAPACK dgetrf\n");
    dgetrf_( &m, &n, A, &lda, ipiv, &info );
    print_matrix("dgetrf(A)", m, n, A, lda, 10, 5);

    int lwork = n*m;
    double *work = (double *)malloc( n * m * sizeof(double) );
    printf("Run LAPACK dgetri\n");
    dgetri_( &n, A, &lda, ipiv, work, &lwork, &info );
    print_matrix("dgetri(A)", m, n, A, lda, 10, 5);


    printf("Reset matrix A\n");
    for (int j=0; j<n; ++j)
        for (int i=0; i<m; ++i)
            A[i+j*n] = i+j+1;
    for (int j=0; j<n; ++j)
        A[j+j*n] += n; // diag dom
    print_matrix("A", m, n, A, lda, 10, 5);

    printf("Run SLATE dgetrf using lapack_api\n");
    slate_dgetrf_( &m, &n, A, &lda, ipiv, &info );
    print_matrix("slate_dgetrf(A)", m, n, A, lda, 10, 5);

    //int lwork = n*m;
    //double *work = (double *)malloc( n * m * sizeof(double) );
    printf("Run SLATE dgetri using lapack_api\n");
    slate_dgetri_( &n, A, &lda, ipiv, work, &lwork, &info );
    print_matrix("slate_dgetri(A)", m, n, A, lda, 10, 5);

    printf("Done\n");
}
