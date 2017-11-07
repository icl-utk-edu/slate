
#include "slate_Matrix.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef SLATE_WITH_OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

extern "C" void trace_on();
extern "C" void trace_off();
extern "C" void trace_finish();

void print_lapack_matrix(int64_t m, int64_t n, double *a, int64_t lda,
                         int64_t mb, int64_t nb);
void diff_lapack_matrices(int64_t m, int64_t n, double *a, int64_t lda,
                          double *b, int64_t ldb,
                          int64_t mb, int64_t nb);

//------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    if (argc < 6) {
        printf("Usage: app nb nt p q lookahead [test]");
        return EXIT_FAILURE;
    }

    int64_t nb = atoll(argv[1]);
    int64_t nt = atoll(argv[2]);
    int64_t p = atoll(argv[3]);
    int64_t q = atoll(argv[4]);
    int64_t lookahead = atoll(argv[5]);
    bool test = argc == 7;

    int64_t n = nb*nt;
    int64_t lda = n;

    //--------------------
    // MPI initializations
    int mpi_rank;
    int mpi_size;
    int provided;
    int retval;

    retval = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    assert(retval == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_MULTIPLE);

    retval = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    assert(retval == MPI_SUCCESS);

    retval = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    assert(retval == MPI_SUCCESS);
    assert(mpi_size == p*q);

    //---------------------
    // test initializations
    double *a1 = nullptr;
    double *a2 = nullptr;

    if (test) {
        int seed[] = {0, 0, 0, 1};
        a1 = new double[nb*nb*nt*nt];
        lapack::larnv(1, seed, lda*n, a1);

        for (int64_t i = 0; i < n; ++i)
            a1[i*lda+i] += sqrt(n);

        if (mpi_rank == 0) {
            a2 = new double[nb*nb*nt*nt];
            memcpy(a2, a1, sizeof(double)*lda*n);
        }
    }

    trace_off();
    //-----------
    // warmup run
    // slate::Matrix<double> temp(n, n, a1, lda, nb, MPI_COMM_WORLD, p, q);
    // temp.potrf(blas::Uplo::Lower);

    slate::Matrix<double> a(n, n, a1, lda, nb, MPI_COMM_WORLD, p, q);
    trace_on();

    trace_cpu_start();
    MPI_Barrier(MPI_COMM_WORLD);
    trace_cpu_stop("Black");

    double start = omp_get_wtime();
    a.potrf(blas::Uplo::Lower, lookahead);

    trace_cpu_start();
    MPI_Barrier(MPI_COMM_WORLD);
    trace_cpu_stop("Black");

    double time = omp_get_wtime()-start;
    trace_finish();

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = (double)nb*nb*nb*nt*nt*nt/3.0;
        double gflops = ops/time/1000000000.0;
        printf("\t%.0lf GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        a.gather();

        if (mpi_rank == 0) {

            retval = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a2, lda);
            assert(retval == 0);

            a.copyFromFull(a1, lda);
            diff_lapack_matrices(n, n, a1, lda, a2, lda, nb, nb);

            cblas_daxpy((size_t)lda*n, -1.0, a1, 1, a2, 1);
            double norm =
                LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, a1, lda);
            delete[] a1;

            double error =
                LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, a2, lda);
            delete[] a2;

            if (norm != 0)
                error /= norm;
            printf("\t%le\n", error);
        }
    }

    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
void print_lapack_matrix(int64_t m, int64_t n, double *a, int64_t lda,
                         int64_t mb, int64_t nb)
{
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            printf("%8.2lf", a[(size_t)lda*j+i]);
            if ((j+1)%nb == 0)
                printf(" |");
        }
        printf("\n");
        if ((i+1)%mb == 0) {
            for (int64_t j = 0; j < (n+1)*8; ++j) {
                printf("-");
            }
            printf("\n");        
        }
    }
    printf("\n");
}

//------------------------------------------------------------------------------
void diff_lapack_matrices(int64_t m, int64_t n, double *a, int64_t lda,
                          double *b, int64_t ldb, int64_t mb, int64_t nb)
{
    for (int64_t i = 0; i < m; ++i) {
        if (i%mb == 2)
            i += mb-4;
        for (int64_t j = 0; j < n; ++j) {
            if (j%nb == 2)
                j += nb-4;
            double error = a[(size_t)lda*j+i] - b[(size_t)lda*j+i];
            printf("%c", error < 0.000001 ? '.' : '#');
            if ((j+1)%nb == 0)
                printf("|");
        }
        printf("\n");
        if ((i+1)%mb == 0) {
            for (int64_t j = 0; j < (n/nb)*5; ++j) {
                printf("-");
            }
            printf("\n");        
        }
    }
    printf("\n");
}
