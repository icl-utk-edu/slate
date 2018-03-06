
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

    double *A1 = nullptr;
    double *B1 = nullptr;
    double *B2 = nullptr;

    int64_t seed_a[] = {0, 1, 0, 0};
    A1 = new double[ lda*n ];
    lapack::larnv(1, seed_a, lda*n, A1);

    int64_t seed_c[] = {0, 0, 0, 1};
    B1 = new double[ lda*n ];
    lapack::larnv(1, seed_c, lda*n, B1);

    if (test) {
        if (mpi_rank == 0) {
            B2 = new double[ lda*n ];
            memcpy(B2, B1, sizeof(double)*lda*n);
        }
    }

    slate::TriangularMatrix<double> A(slate::Uplo::Lower,
                                      n, A1, lda,
                                      nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, n, B1, lda, nb, p, q, MPI_COMM_WORLD);
    slate::trace::Trace::on();
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double start = omp_get_wtime();
    slate::trmm<slate::Target::HostTask>(
        blas::Side::Left, blas::Diag::NonUnit,
        alpha, A, B, {{slate::Option::Lookahead, lookahead}});

    MPI_Barrier(MPI_COMM_WORLD);
    double time = omp_get_wtime() - start;
    slate::trace::Trace::finish();

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = (double)n*n*n;
        double gflops = ops/time/1e9;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        B.gather(B1, lda);

        if (mpi_rank == 0) {
            blas::trmm(blas::Layout::ColMajor,
                       blas::Side::Left, blas::Uplo::Lower,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, n,
                       alpha, A1, lda,
                              B2, lda);

            slate::Debug::diffLapackMatrices(n, n, B1, lda, B2, lda, nb, nb);

            blas::axpy((size_t)lda*n, -1.0, B1, 1, B2, 1);
            double norm =
                lapack::lange(lapack::Norm::Fro, n, n, B1, lda);

            double error =
                lapack::lange(lapack::Norm::Fro, n, n, B2, lda);

            if (norm != 0)
                error /= norm;
            printf("\t%le\n", error);

            delete[] B2;
            B2 = nullptr;
        }
    }
    delete[] B1;
    B1 = nullptr;

    MPI_Finalize();
    return EXIT_SUCCESS;
}
