
#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_trace_Trace.hh"

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

#ifdef _OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

//------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    if (argc < 6) {
        printf("Usage: %s n nb p q lookahead [test]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int64_t n = atol(argv[1]);
    int64_t nb = atol(argv[2]);
    int64_t p = atol(argv[3]);
    int64_t q = atol(argv[4]);
    int64_t lookahead = atol(argv[5]);
    bool test = argc == 7;
    
    printf( "n=%lld, nb=%lld, p=%lld, q=%lld\n", n, nb, p, q );
    // for now, gemm requires full tiles
    assert(n % nb == 0);

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
    double alpha = 1.234;
    double beta = 4.321;

    double *A1 = nullptr;
    double *B1 = nullptr;
    double *C1 = nullptr;
    double *C2 = nullptr;

    int64_t seed_a[] = {0, 1, 0, 0};
    A1 = new double[ lda*n ];
    lapack::larnv(1, seed_a, lda*n, A1);

    int64_t seed_b[] = {0, 0, 1, 0};
    B1 = new double[ lda*n ];
    lapack::larnv(1, seed_b, lda*n, B1);

    int64_t seed_c[] = {0, 0, 0, 1};
    C1 = new double[ lda*n ];
    lapack::larnv(1, seed_c, lda*n, C1);

    if (test) {
        if (mpi_rank == 0) {
            C2 = new double[ lda*n ];
            memcpy(C2, C1, sizeof(double)*lda*n);
        }
    }

    slate::Matrix<double> A(n, n, A1, lda, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, n, B1, lda, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> C(n, n, C1, lda, nb, p, q, MPI_COMM_WORLD);
    slate::trace::Trace::on();
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double start = omp_get_wtime();
    slate::gemm<slate::Target::Devices>(
        alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});

    MPI_Barrier(MPI_COMM_WORLD);
    double time = omp_get_wtime() - start;
    slate::trace::Trace::finish();

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = 2.0*n*n*n;
        double gflops = ops/time/1e9;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        C.gather(C1, lda);

        if (mpi_rank == 0) {
            blas::gemm(blas::Layout::ColMajor,
                       blas::Op::NoTrans, blas::Op::NoTrans,
                       n, n, n,
                       alpha, A1,  lda,
                              B1,  lda,
                       beta,  C2, lda);

            // C.copyFromFull(C1, lda);
            slate::Debug::diffLapackMatrices(n, n, C1, lda, C2, lda, nb, nb);

            blas::axpy((size_t)lda*n, -1.0, C1, 1, C2, 1);
            double norm =
                lapack::lange(lapack::Norm::Fro, n, n, C1, lda);

            double error =
                lapack::lange(lapack::Norm::Fro, n, n, C2, lda);

            if (norm != 0)
                error /= norm;
            printf("\t%le\n", error);

            delete[] C2;
            C2 = nullptr;
        }
    }
    delete[] C1;
    C1 = nullptr;

    MPI_Finalize();
    return EXIT_SUCCESS;
}
