
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
template <typename scalar_t>
void test_potrf(
    int64_t n, int64_t nb, int p, int q, int64_t lookahead,
    slate::Target target, bool test, bool verbose, bool trace )
{
    using real_t = typename blas::traits<scalar_t>::real_t;
    using blas::real;

    //--------------------
    // MPI initializations
    int mpi_rank;
    int mpi_size;
    int retval;

    retval = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    assert(retval == MPI_SUCCESS);

    retval = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    assert(retval == MPI_SUCCESS);
    assert(mpi_size == p*q);

    if (mpi_rank == 0) {
        printf( "n=%lld, nb=%lld, p=%d, q=%d, lookahead=%lld, target=%d, test=%d, verbose=%d, trace=%d\n",
                n, nb, p, q, lookahead, int(target), test, verbose, trace );
    }

    //---------------------
    // test initializations
    int64_t lda = n;

    scalar_t *Adata = nullptr;
    scalar_t *Aref  = nullptr;

    int64_t seed[] = {0, 0, 0, 1};
    Adata = new scalar_t[ lda*n ];
    lapack::larnv(1, seed, lda*n, Adata);

    // set unused data to nan
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < j && i < n; ++i) // upper, excluding diagonal
            Adata[i + j*lda] = nan("");

    // brute force positive definite
    for (int64_t i = 0; i < n; ++i)
        Adata[i + i*lda] = real( Adata[ i + i*lda ] ) + n;

    if (test) {
        if (mpi_rank == 0) {
            Aref = new scalar_t[ lda*n ];
            memcpy(Aref, Adata, sizeof(scalar_t)*lda*n);
        }
    }

    slate::HermitianMatrix< scalar_t > A(slate::Uplo::Lower, n, Adata, lda,
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

    //---------------------
    // run test
    if (target == slate::Target::HostTask)
        slate::potrf< slate::Target::HostTask >(A, lookahead);
    else if (target == slate::Target::HostNest)
        slate::potrf< slate::Target::HostNest >(A, lookahead);
    else if (target == slate::Target::HostBatch)
        slate::potrf< slate::Target::HostBatch >(A, lookahead);
    else if (target == slate::Target::Devices)
        slate::potrf< slate::Target::Devices >(A, lookahead);

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
            real_t Anorm =
                lapack::lanhe(lapack::Norm::Fro, lapack::Uplo::Lower, n, Aref, lda);

            retval = lapack::potrf(lapack::Uplo::Lower, n, Aref, lda);
            assert(retval == 0);

            //slate::Debug::diffLapackMatrices(n, n, Adata, lda, Aref, lda, nb, nb);

            blas::axpy((size_t)lda*n, -1.0, Aref, 1, Adata, 1);
            real_t error =
                lapack::lanhe(lapack::Norm::Fro, lapack::Uplo::Lower, n, Adata, lda);
            if (Anorm != 0)
                error /= Anorm;

            if (verbose) {
                printf( "Aref2 = " );
                print( n, n, Aref, lda );

                printf( "diff = " );
                print( n, n, Adata, lda );
            }

            real_t eps = std::numeric_limits< real_t >::epsilon();
            bool okay = (error < 50*eps);
            printf("\t%.2e error, %s\n", error, okay ? "ok" : "failed");

            delete[] Aref;
            Aref = nullptr;
        }
    }
    delete[] Adata;
    Adata = nullptr;
}

//------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    if (argc < 6) {
        printf("Usage: %s n nb p q lookahead [HostTask|HostNest|HostBatch|Devices] [s|d|c|z] [test] [verbose] [trace]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int64_t n  = atoll(argv[1]);
    int64_t nb = atoll(argv[2]);
    int p = atoi(argv[3]);
    int q = atoi(argv[4]);
    int64_t lookahead = atoll(argv[5]);

    slate::Target target = slate::Target::HostTask;
    if (argc > 6) {
        if (std::string(argv[6]) == "HostTask")
            target = slate::Target::HostTask;
        else if (std::string(argv[6]) == "HostNest")
            target = slate::Target::HostNest;
        else if (std::string(argv[6]) == "HostBatch")
            target = slate::Target::HostBatch;
        else if (std::string(argv[6]) == "Devices")
            target = slate::Target::Devices;
        else {
            printf( "Unknown target: %s\n", argv[6] );
            exit(1);
        }
    }

    char type = 'd';
    if (argc > 7)
        type = argv[7][0];

    bool test    = argc >  8 && std::string(argv[ 8]) == "test";
    bool verbose = argc >  9 && std::string(argv[ 9]) == "verbose";
    bool trace   = argc > 10 && std::string(argv[10]) == "trace";

    //--------------------
    // MPI initializations
    int provided;
    int retval;

    retval = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    assert(retval == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_MULTIPLE);

    //--------------------
    // run test
    switch (type) {
        case 's':
            test_potrf< float >( n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'd':
            test_potrf< double >( n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'c':
            test_potrf< std::complex<float> >( n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'z':
            test_potrf< std::complex<double> >( n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        default:
            printf( "unknown datatype: %c\n", type );
            break;
    }

    //--------------------
    // MPI finalize
    MPI_Finalize();
    return EXIT_SUCCESS;
}
