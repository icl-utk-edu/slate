
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
void test_symm(
    blas::Side side, blas::Uplo uplo,
    int64_t m, int64_t n, int64_t nb, int p, int q, int64_t lookahead,
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
        printf( "side=%c, uplo=%c, m=%lld, n=%lld, nb=%lld, p=%d, q=%d, lookahead=%lld, target=%d\n",
                char(side), char(uplo), lld(m), lld(n), lld(nb), p, q, lld(lookahead), int(target) );
    }

    // for now, symm on Devices requires full tiles
    if (target == slate::Target::Devices) {
        assert(m % nb == 0);
        assert(n % nb == 0);
    }

    int64_t An = (side == blas::Side::Left ? m : n);
    int64_t lda = An;
    int64_t ldb = m;
    int64_t ldc = m;

    // todo: complex
    scalar_t alpha = 1.234;
    scalar_t beta  = 4.321;

    scalar_t *A1 = nullptr;
    scalar_t *B1 = nullptr;
    scalar_t *C1 = nullptr;
    scalar_t *C2 = nullptr;

    int64_t seed_a[] = {0, 1, 0, 0};
    A1 = new scalar_t[ lda*An ];
    lapack::larnv(1, seed_a, lda*An, A1);

    int64_t seed_b[] = {0, 0, 1, 0};
    B1 = new scalar_t[ ldb*n ];
    lapack::larnv(1, seed_b, ldb*n, B1);

    int64_t seed_c[] = {0, 0, 0, 1};
    C1 = new scalar_t[ ldc*n ];
    lapack::larnv(1, seed_c, ldc*n, C1);

    if (test) {
        if (mpi_rank == 0) {
            C2 = new scalar_t[ ldc*n ];
            memcpy(C2, C1, sizeof(scalar_t)*ldc*n);
        }
    }

    slate::SymmetricMatrix<scalar_t> A(uplo, An, A1, lda, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> B( m,  n, B1, ldb, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> C( m,  n, C1, ldc, nb, p, q, MPI_COMM_WORLD);

    if (side == blas::Side::Left)
        assert( A.mt() == C.mt() );
    else
        assert( A.mt() == C.nt() );
    assert( B.mt() == C.mt() );
    assert( B.nt() == C.nt() );

    if (verbose && mpi_rank == 0) {
        printf( "alpha = %.4f + %.4fi;\n"
                "beta  = %.4f + %.4fi;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        print( "A1", An, An, A1, lda );
        print( "A",  A );
        print( "B1", m, n, B1, ldb );
        print( "B",  B );
        print( "C1", m, n, C1, ldc );
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
            slate::symm<slate::Target::HostTask>(
                side, alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostNest:
            slate::symm<slate::Target::HostNest>(
                side, alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostBatch:
            slate::symm<slate::Target::HostBatch>(
                side, alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::Devices:
            slate::symm<slate::Target::Devices>(
                side, alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
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
        print( "C1res", m, n, C1, ldc );
        print( "Cres", C );
    }

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = ((side == blas::Side::Left ? m*m*n : m*n*n) + (side == blas::Side::Left ? m*m*n : m*n*n)); // todo check
        double gflops = ops/time/1e9;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        C.gather(C1, ldc);

        if (mpi_rank == 0) {
            blas::symm(blas::Layout::ColMajor,
                       side, uplo,
                       m, n,
                       alpha, A1, lda,
                              B1, ldb,
                       beta,  C2, ldc);

            if (verbose && mpi_rank == 0) {
                print( "Cref", m, n, C2, ldc );
            }
            if (verbose)
                slate::Debug::diffLapackMatrices(m, n, C1, ldc, C2, ldc, nb, nb);

            blas::axpy((size_t)ldc*n, -1.0, C1, 1, C2, 1);
            real_t norm =
                lapack::lange(lapack::Norm::Fro, m, n, C1, ldc);

            real_t error =
                lapack::lange(lapack::Norm::Fro, m, n, C2, ldc);

            if (norm != 0)
                error /= norm;

            real_t eps = std::numeric_limits< real_t >::epsilon();
            bool okay = (error < 50*eps);
            printf("\t%.2e error, %s\n", error, okay ? "ok" : "failed");

            delete[] C2;
            C2 = nullptr;
        }
    }
    delete[] C1;
    C1 = nullptr;
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
        printf("Usage: %s {left,right} {upper,lower} m n nb p q lookahead [HostTask|HostNest|HostBatch|Devices] [s|d|c|z] [test] [verbose] [trace]\n"
               "For side, uplo, only the first letter is used.\n", argv[0]);
        return EXIT_FAILURE;
    }

    int arg = 1;
    blas::Side side = blas::char2side( argv[arg][0] );  ++arg;
    blas::Uplo uplo = blas::char2uplo( argv[arg][0] );  ++arg;
    int64_t m  = atol(argv[arg]);  ++arg;
    int64_t n  = atol(argv[arg]);  ++arg;
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
            test_symm< float >( side, uplo, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'd':
            test_symm< double >( side, uplo, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'c':
            test_symm< std::complex<float> >( side, uplo, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'z':
            test_symm< std::complex<double> >( side, uplo, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        default:
            printf( "unknown datatype: %c\n", datatype );
            break;
    }

    //--------------------
    MPI_Finalize();
    return EXIT_SUCCESS;
}
