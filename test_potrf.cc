
#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_Trace.hh"

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

    int64_t n  = atoll(argv[1]);
    int64_t nb = atoll(argv[2]);
    int p = atoi(argv[3]);
    int q = atoi(argv[4]);
    int64_t lookahead = atoll(argv[5]);
    bool test = argc == 7;
    
    printf( "n=%lld, nb=%lld, p=%d, q=%d\n", n, nb, p, q );
    // for now, potrf requires full tiles
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
    double *A1 = nullptr;
    double *A2 = nullptr;

    int64_t seed[] = {0, 0, 0, 1};
    A1 = new double[ lda*n ];
    lapack::larnv(1, seed, lda*n, A1);

    // brute force positive definite
    for (int64_t i = 0; i < n; ++i)
        A1[i*lda+i] += sqrt(n);

    if (test) {
        if (mpi_rank == 0) {
            A2 = new double[ lda*n ];
            memcpy(A2, A1, sizeof(double)*lda*n);
        }
    }

    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, A1, lda,
                                     nb, p, q, MPI_COMM_WORLD);
    slate::trace::Trace::on();
    {
        slate::trace::Block trace_block(slate::trace::Color::Black);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double start = omp_get_wtime();
    slate::potrf<slate::Target::HostTask>(A, lookahead);

    MPI_Barrier(MPI_COMM_WORLD);
    double time = omp_get_wtime() - start;
    slate::trace::Trace::finish();

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = (double) n*n*n / 3.0;
        double gflops = ops/time/1e9;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        //A.gather(A1, lda);

        if (mpi_rank == 0) {
            retval = lapack::potrf(lapack::Uplo::Lower, n, A2, lda);
            assert(retval == 0);

            // A.copyFromFull(A1, lda);
            slate::Debug::diffLapackMatrices(n, n, A1, lda, A2, lda, nb, nb);

            blas::axpy((size_t)lda*n, -1.0, A1, 1, A2, 1);
            double norm =
                lapack::lansy(lapack::Norm::Fro, lapack::Uplo::Lower, n, A1, lda);

            double error =
                lapack::lansy(lapack::Norm::Fro, lapack::Uplo::Lower, n, A2, lda);

            if (norm != 0)
                error /= norm;
            printf("\t%le\n", error);

            delete[] A2;
            A2 = nullptr;
        }
    }
    delete[] A1;
    A1 = nullptr;

    MPI_Finalize();
    return EXIT_SUCCESS;
}
