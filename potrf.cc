
#include "slate.hh"
#include "slate_Debug.hh"

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
        int64_t seed[] = {0, 0, 0, 1};
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
    slate::Matrix<double> a(n, n, a1, lda, nb, MPI_COMM_WORLD, p, q);
    trace_on();

    trace_cpu_start();
    MPI_Barrier(MPI_COMM_WORLD);
    trace_cpu_stop("Black");

    double start = omp_get_wtime();
    slate::potrf<slate::Target::Devices>(slate::Uplo::Lower, a, lookahead);

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
        a.gather(a1, lda);

        if (mpi_rank == 0) {

            retval = lapack::potrf(lapack::Uplo::Lower, n, a2, lda);
            assert(retval == 0);

            // a.copyFromFull(a1, lda);
            slate::Debug::diffLapackMatrices(n, n, a1, lda, a2, lda, nb, nb);

            blas::axpy((size_t)lda*n, -1.0, a1, 1, a2, 1);
            double norm =
                lapack::lansy(lapack::Norm::Fro, lapack::Uplo::Lower, n, a1, lda);
            delete[] a1;

            double error =
                lapack::lange(lapack::Norm::Fro, n, n, a2, lda);
            delete[] a2;

            if (norm != 0)
                error /= norm;
            printf("\t%le\n", error);
        }
    }

    return EXIT_SUCCESS;
}
