
#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_trace_Trace.hh"

#include "test.hh"

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
template <typename scalar_t>
void test_potrf(
    blas::Uplo uplo, int64_t n, int64_t nb, int p, int q, int64_t lookahead,
    slate::Target target, bool test, bool verbose, bool trace )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    typedef long long lld;

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

    //---------------------
    // test initializations
    if (mpi_rank == 0) {
        printf( "uplo=%c, n=%lld, nb=%lld, p=%d, q=%d, lookahead=%lld, target=%d\n",
                char(uplo), lld(n), lld(nb), p, q, lld(lookahead), int(target) );
    }

    int64_t lda = n;

    scalar_t *Adata = nullptr;
    scalar_t *Aref  = nullptr;

    int64_t seed[] = {0, 0, 0, 1};
    Adata = new scalar_t[ lda*n ];
    lapack::larnv(1, seed, lda*n, Adata);

    // set unused data to nan
    scalar_t nan_ = nan("");
    if (uplo == blas::Uplo::Lower) {
        lapack::laset( lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                       &Adata[ 0 + 1*lda ], lda );
    }
    else {
        lapack::laset( lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                       &Adata[ 1 + 0*lda ], lda );
    }

    // brute force positive definite
    for (int64_t i = 0; i < n; ++i)
        Adata[i + i*lda] = real( Adata[ i + i*lda ] ) + n;

    if (test) {
        if (mpi_rank == 0) {
            Aref = new scalar_t[ lda*n ];
            memcpy(Aref, Adata, sizeof(scalar_t)*lda*n);
        }
    }

    slate::HermitianMatrix< scalar_t > A(uplo, n, Adata, lda,
                                         nb, p, q, MPI_COMM_WORLD);

    if (verbose && mpi_rank == 0) {
        print( "Adata", n, n, Adata, lda );
        print( "A", A );
    }

    //---------------------
    // run test
    if (trace)
        slate::trace::Trace::on();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double start = omp_get_wtime();

    switch (target) {
        case slate::Target::Host:
        case slate::Target::HostTask:
            slate::potrf< slate::Target::HostTask >(A, lookahead);
            break;
        case slate::Target::HostNest:
            slate::potrf< slate::Target::HostNest >(A, lookahead);
            break;
        case slate::Target::HostBatch:
            slate::potrf< slate::Target::HostBatch >(A, lookahead);
            break;
        case slate::Target::Devices:
            slate::potrf< slate::Target::Devices >(A, lookahead);
            break;
    }

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = omp_get_wtime() - start;

    if (trace)
        slate::trace::Trace::finish();

    if (verbose) {
        print( "A1res", n, n, Adata, lda );
        print( "Ares", A );
    }

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = (double) n*n*n / 3.0;
        double gflops = ops/time;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        A.gather(Adata, lda);

        if (mpi_rank == 0) {
            real_t Anorm =
                lapack::lanhe(lapack::Norm::Fro, uplo, n, Aref, lda);

            retval = lapack::potrf(uplo, n, Aref, lda);
            assert(retval == 0);

            if (verbose) {
                print( "Aref", n, n, Aref, lda );
            }
            if (verbose)
                slate::Debug::diffLapackMatrices(n, n, Adata, lda, Aref, lda, nb, nb);

            blas::axpy((size_t)lda*n, -1.0, Aref, 1, Adata, 1);
            real_t error =
                lapack::lanhe(lapack::Norm::Fro, uplo, n, Adata, lda);

            if (Anorm != 0)
                error /= Anorm;

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
    //--------------------
    // MPI initializations
    int provided;
    int retval;
    int mpi_rank;

    retval = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    assert(retval == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_MULTIPLE);

    retval = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    assert(retval == MPI_SUCCESS);

    //--------------------
    // parse command line
    if (argc < 7 && mpi_rank == 0) {
        printf("Usage: %s {upper,lower} n nb p q lookahead [HostTask|HostNest|HostBatch|Devices] [s|d|c|z] [test] [verbose] [trace]\n"
               "For uplo, only the first letter is used.\n", argv[0]);
        return EXIT_FAILURE;
    }

    int arg = 1;
    blas::Uplo uplo = blas::char2uplo( argv[arg][0] );  ++arg;
    int64_t n  = atoll(argv[arg]);  ++arg;
    int64_t nb = atoll(argv[arg]);  ++arg;
    int p      = atoi(argv[arg]);   ++arg;
    int q      = atoi(argv[arg]);   ++arg;
    int64_t lookahead = atoll(argv[arg]);  ++arg;

    slate::Target target = slate::Target::HostTask;
    if (argc > arg) {
        std::string s( argv[arg] );
        if (s == "HostTask")
            target = slate::Target::HostTask;
        else if (s == "HostNest")
            target = slate::Target::HostNest;
        else if (s == "HostBatch")
            target = slate::Target::HostBatch;
        else if (s == "Devices")
            target = slate::Target::Devices;
        else {
            printf( "Unknown target: %s\n", argv[arg] );
            return EXIT_FAILURE;
        }
        ++arg;
    }

    char datatype = 'd';
    if (argc > arg) {
        datatype = argv[arg][0];
        ++arg;
    }

    bool test    = argc > arg && std::string(argv[arg]) == "test";    ++arg;
    bool verbose = argc > arg && std::string(argv[arg]) == "verbose"; ++arg;
    bool trace   = argc > arg && std::string(argv[arg]) == "trace";   ++arg;

    //--------------------
    // run test
    switch (datatype) {
        case 's':
            test_potrf< float >( uplo, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'd':
            test_potrf< double >( uplo, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'c':
            test_potrf< std::complex<float> >( uplo, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'z':
            test_potrf< std::complex<double> >( uplo, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        default:
            printf( "unknown datatype: %c\n", datatype );
            break;
    }

    //--------------------
    MPI_Finalize();
    return EXIT_SUCCESS;
}
