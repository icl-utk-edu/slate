
#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_trace_Trace.hh"

#include "test.hh"
#include "blas_flops.hh"

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
void test_herk(
    blas::Op opA, blas::Uplo uplo,
    int64_t n, int64_t k, int64_t nb, int p, int q, int64_t lookahead,
    slate::Target target, bool test, bool verbose, bool trace )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::Op;
    using blas::real;
    using blas::imag;
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
        printf( "opA=%c, uplo=%c, n=%lld, k=%lld, nb=%lld, p=%d, q=%d, lookahead=%lld, target=%d\n",
                char(opA), char(uplo), lld(n), lld(k), lld(nb), p, q, lld(lookahead), int(target) );
    }

    // for now, herk on Devices requires full tiles
    if (target == slate::Target::Devices) {
        assert(n % nb == 0);
        assert(k % nb == 0);
    }

    // setup so op(A) is n-by-k
    int64_t Am = (opA == blas::Op::NoTrans ? n : k);
    int64_t An = (opA == blas::Op::NoTrans ? k : n);
    int64_t lda = Am;
    int64_t ldc = n;

    real_t alpha;
    real_t beta;
    int64_t seed[] = {0, 1, 2, 3};
    lapack::larnv(1, seed, 1, &alpha);
    lapack::larnv(1, seed, 1, &beta);

    scalar_t *A1 = nullptr;
    scalar_t *C1 = nullptr;
    scalar_t *C2 = nullptr;

    A1 = new scalar_t[ lda*An ];
    lapack::larnv(1, seed, lda*An, A1);

    C1 = new scalar_t[ ldc*n ];
    lapack::larnv(1, seed, ldc*n, C1);

    // set unused data to nan
    scalar_t nan_ = nan("");
    if (uplo == blas::Uplo::Lower) {
        lapack::laset( lapack::MatrixType::Upper, n-1, n-1, nan_, nan_,
                       &C1[ 0 + 1*ldc ], ldc );
    }
    else {
        lapack::laset( lapack::MatrixType::Lower, n-1, n-1, nan_, nan_,
                       &C1[ 1 + 0*ldc ], ldc );
    }

    if (test) {
        if (mpi_rank == 0) {
            C2 = new scalar_t[ ldc*n ];
            memcpy(C2, C1, sizeof(scalar_t)*ldc*n);
        }
    }

    slate::Matrix<scalar_t> A(Am, An, A1, lda, nb, p, q, MPI_COMM_WORLD);
    slate::HermitianMatrix<scalar_t> C(uplo, n, C1, ldc, nb, p, q, MPI_COMM_WORLD);

    if (opA == blas::Op::Trans) {
        A = transpose( A );
    }
    else if (opA == blas::Op::ConjTrans) {
        A = conj_transpose( A );
    }
    assert( A.mt() == C.mt() );

    if (verbose && mpi_rank == 0) {
        printf( "alpha = %.4f + %.4fi;\n"
                "beta  = %.4f + %.4fi;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        print( "A1", Am, An, A1, lda );
        print( "A",  A );
        print( "C1", n, k, C1, ldc );
        print( "C",  C );
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
            slate::herk<slate::Target::HostTask>(
                alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostNest:
            slate::herk<slate::Target::HostNest>(
                alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostBatch:
            slate::herk<slate::Target::HostBatch>(
                alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::Devices:
            slate::herk<slate::Target::Devices>(
                alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
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
        print( "C1res", n, n, C1, ldc );
        print( "Cres", C );
    }

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = blas::Gflop<scalar_t>::herk( n, k );
        double gflops = ops/time;
        printf("\t%.2f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        C.gather(C1, ldc);

        if (mpi_rank == 0) {
            blas::herk(blas::Layout::ColMajor,
                       uplo, opA,
                       n, k,
                       alpha, A1, lda,
                       beta,  C2, ldc);

            if (verbose && mpi_rank == 0) {
                print( "Cref", n, n, C2, ldc );
            }
            if (verbose)
                slate::Debug::diffLapackMatrices(n, n, C1, ldc, C2, ldc, nb, nb);

            blas::axpy((size_t)ldc*n, -1.0, C1, 1, C2, 1);
            real_t norm =
                lapack::lanhe(lapack::Norm::Fro, uplo, n, C1, ldc);

            real_t error =
                lapack::lanhe(lapack::Norm::Fro, uplo, n, C2, ldc);

            if (norm != 0)
                error /= norm;

            real_t eps = std::numeric_limits< real_t >::epsilon();
            bool okay = (error < 50*eps);
            printf("\t%.2e error, %s\n", error, okay ? "ok" : "failed");
        }
    }

    delete[] A1;
    A1 = nullptr;

    delete[] C1;
    C1 = nullptr;

    delete[] C2;
    C2 = nullptr;
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
    if (argc < 9 && mpi_rank == 0) {
        printf("Usage: %s {notrans,trans,conjtrans} {upper,lower} n k nb p q lookahead [HostTask|HostNest|HostBatch|Devices] [s|d|c|z] [test] [verbose] [trace]\n"
               "For opA, uplo, only the first letter is used.\n", argv[0]);
        return EXIT_FAILURE;
    }

    int arg = 1;
    blas::Op   opA  = blas::char2op  ( argv[arg][0] );  ++arg;
    blas::Uplo uplo = blas::char2uplo( argv[arg][0] );  ++arg;
    int64_t n  = atol(argv[arg]);  ++arg;
    int64_t k  = atol(argv[arg]);  ++arg;
    int64_t nb = atol(argv[arg]);  ++arg;
    int p      = atoi(argv[arg]);  ++arg;
    int q      = atoi(argv[arg]);  ++arg;
    int64_t lookahead = atol(argv[arg]);  ++arg;

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
            test_herk< float >( opA, uplo, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'd':
            test_herk< double >( opA, uplo, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'c':
            test_herk< std::complex<float> >( opA, uplo, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'z':
            test_herk< std::complex<double> >( opA, uplo, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        default:
            printf( "unknown datatype: %c\n", datatype );
            break;
    }

    //--------------------
    MPI_Finalize();
    return EXIT_SUCCESS;
}
