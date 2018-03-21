
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
void test_gemm(
    blas::Op opA, blas::Op opB,
    int64_t m, int64_t n, int64_t k, int64_t nb, int p, int q, int64_t lookahead,
    slate::Target target, bool test, bool verbose, bool trace )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::Op;
    using blas::real;
    using blas::imag;

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
        printf( "opA=%c, opB=%c, m=%lld, n=%lld, k=%lld, nb=%lld, p=%d, q=%d, lookahead=%lld, target=%d\n",
                char(opA), char(opB), m, n, k, nb, p, q, lookahead, int(target) );
    }

    // for now, gemm on Devices requires full tiles
    if (target == slate::Target::Devices) {
        assert(m % nb == 0);
        assert(n % nb == 0);
        assert(k % nb == 0);
    }

    int64_t Am = (opA == Op::NoTrans ? m : k);
    int64_t An = (opA == Op::NoTrans ? k : m);
    int64_t Bm = (opB == Op::NoTrans ? k : n);
    int64_t Bn = (opB == Op::NoTrans ? n : k);

    int64_t lda = Am;
    int64_t ldb = Bm;
    int64_t ldc = m;

    // todo: complex
    scalar_t alpha = 1.234;
    scalar_t beta = 4.321;

    scalar_t *A1 = nullptr;
    scalar_t *B1 = nullptr;
    scalar_t *C1 = nullptr;
    scalar_t *C2 = nullptr;

    int64_t seed_a[] = {0, 1, 0, 0};
    A1 = new scalar_t[ lda*An ];
    lapack::larnv(1, seed_a, lda*An, A1);

    int64_t seed_b[] = {0, 0, 1, 0};
    B1 = new scalar_t[ ldb*Bn ];
    lapack::larnv(1, seed_b, ldb*Bn, B1);

    int64_t seed_c[] = {0, 0, 0, 1};
    C1 = new scalar_t[ ldc*n ];
    lapack::larnv(1, seed_c, ldc*n, C1);

    if (test) {
        if (mpi_rank == 0) {
            C2 = new scalar_t[ ldc*n ];
            memcpy(C2, C1, sizeof(scalar_t)*ldc*n);
        }
    }

    slate::Matrix<scalar_t> A(Am, An, A1, lda, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> B(Bm, Bn, B1, ldb, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> C( m,  n, C1, ldc, nb, p, q, MPI_COMM_WORLD);

    if (opA == Op::Trans)
        A = transpose( A );
    else if (opA == Op::ConjTrans)
        A = conj_transpose( A );

    if (opB == Op::Trans)
        B = transpose( B );
    else if (opB == Op::ConjTrans)
        B = conj_transpose( B );

    assert( A.mt() == C.mt() );
    assert( B.nt() == C.nt() );
    assert( A.nt() == B.mt() );

    if (verbose && mpi_rank == 0) {
        printf( "alpha = %.4f + %.4fi;\n"
                "beta  = %.4f + %.4fi;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        print( "A1", Am, An, A1, lda );
        print( "A",  A );
        print( "B1", Bm, Bn, B1, ldb );
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
            slate::gemm<slate::Target::HostTask>(
                alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostNest:
            slate::gemm<slate::Target::HostNest>(
                alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostBatch:
            slate::gemm<slate::Target::HostBatch>(
                alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::Devices:
            slate::gemm<slate::Target::Devices>(
                alpha, A, B, beta, C, {{slate::Option::Lookahead, lookahead}});
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
        double ops = 2.0*n*n*n;
        double gflops = ops/time/1e9;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        C.gather(C1, ldc);

        if (mpi_rank == 0) {
            blas::gemm(blas::Layout::ColMajor,
                       opA, opB,
                       m, n, k,
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
    if (argc < 10 && mpi_rank == 0) {
        printf("Usage: %s {notrans,trans,conj} {notrans,trans,conj} m n k nb p q lookahead [HostTask|HostNest|HostBatch|Devices] [s|d|c|z] [test] [verbose] [trace]\n"
               "For opA, opB, only the first letter is used.\n", argv[0]);
        return EXIT_FAILURE;
    }

    int arg = 1;
    blas::Op opA = blas::char2op( argv[arg][0] );  ++arg;
    blas::Op opB = blas::char2op( argv[arg][0] );  ++arg;
    int64_t m  = atol(argv[arg]);  ++arg;
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
            test_gemm< float >( opA, opB, m, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'd':
            test_gemm< double >( opA, opB, m, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'c':
            test_gemm< std::complex<float> >( opA, opB, m, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'z':
            test_gemm< std::complex<double> >( opA, opB, m, n, k, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        default:
            printf( "unknown datatype: %c\n", datatype );
            break;
    }

    //--------------------
    MPI_Finalize();
    return EXIT_SUCCESS;
}
