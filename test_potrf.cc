
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

#include "test.hh"

//------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    if (argc < 6) {
        printf("Usage: %s n nb p q lookahead [host|device] [test] [verbose] [trace]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int64_t n  = atoll(argv[1]);
    int64_t nb = atoll(argv[2]);
    int p = atoi(argv[3]);
    int q = atoi(argv[4]);
    int64_t lookahead = atoll(argv[5]);
    bool use_device = argc > 6 && std::string(argv[6]) == "device";
    bool test       = argc > 7 && std::string(argv[7]) == "test";
    bool verbose    = argc > 8 && std::string(argv[8]) == "verbose";
    bool trace      = argc > 9 && std::string(argv[9]) == "trace";

    printf( "n=%lld, nb=%lld, p=%d, q=%d, lookahead=%lld, use_device=%d, test=%d, verbose=%d, trace=%d\n",
            n, nb, p, q, lookahead, use_device, test, verbose, trace );
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
    double *Adata = nullptr;
    double *Aref  = nullptr;

    int64_t seed[] = {0, 0, 0, 1};
    Adata = new double[ lda*n ];
    lapack::larnv(1, seed, lda*n, Adata);

    // set unused data to nan
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < j && i < n; ++i) // upper, excluding diagonal
            Adata[i + j*lda] = nan("");

    // brute force positive definite
    for (int64_t i = 0; i < n; ++i)
        Adata[i + i*lda] += sqrt(n);

    if (test) {
        if (mpi_rank == 0) {
            Aref = new double[ lda*n ];
            memcpy(Aref, Adata, sizeof(double)*lda*n);
        }
    }

    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, Adata, lda,
                                     nb, p, q, MPI_COMM_WORLD);

    if (verbose && mpi_rank == 0) {
        printf( "Adata = " );
        print( n, n, Adata, lda );

        printf( "A = " );
        print( A );
    }

    if (trace) {
        slate::trace::Trace::on();
    }
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double start = omp_get_wtime();
    if (use_device) {
        slate::potrf<slate::Target::Devices>(A, lookahead);
    }
    else {
        slate::potrf<slate::Target::HostTask>(A, lookahead);
    }

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = omp_get_wtime() - start;
    if (trace) {
        slate::trace::Trace::finish();
    }

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = (double) n*n*n / 3.0;
        double gflops = ops/time/1e9;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    if (verbose && mpi_rank == 0) {
        printf( "Adata2 = " );
        print( n, n, Adata, lda );
    }

    //------------------
    // Test correctness.
    if (test) {
        A.gather(Adata, lda);

        if (mpi_rank == 0) {
            double Anorm =
                lapack::lanhe(lapack::Norm::Fro, lapack::Uplo::Lower, n, Aref, lda);

            retval = lapack::potrf(lapack::Uplo::Lower, n, Aref, lda);
            assert(retval == 0);

            //slate::Debug::diffLapackMatrices(n, n, Adata, lda, Aref, lda, nb, nb);

            blas::axpy((size_t)lda*n, -1.0, Aref, 1, Adata, 1);
            double error =
                lapack::lanhe(lapack::Norm::Fro, lapack::Uplo::Lower, n, Adata, lda);
            if (Anorm != 0)
                error /= Anorm;

            if (verbose) {
                printf( "Aref2 = " );
                print( n, n, Aref, lda );

                printf( "diff = " );
                print( n, n, Adata, lda );
            }

            printf("\t%.2e error\n", error);

            delete[] Aref;
            Aref = nullptr;
        }
    }
    delete[] Adata;
    Adata = nullptr;

    MPI_Finalize();
    return EXIT_SUCCESS;
}
