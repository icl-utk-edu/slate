
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
void test_trmm(
    blas::Side side, blas::Uplo uplo, blas::Op opA, blas::Op opB, blas::Diag diag,
    int64_t m, int64_t n, int64_t nb, int p, int q, int64_t lookahead,
    slate::Target target, bool test, bool verbose, bool trace )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::Op;
    using blas::real;
    using blas::imag;
    using blas::conj;
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
        printf( "side=%c, uplo=%c, opA=%c, obB=%c, diag=%c, m=%lld, n=%lld, nb=%lld, p=%d, q=%d, lookahead=%lld, target=%d\n",
                char(side), char(uplo), char(opA), char(opB), char(diag), lld(m), lld(n), lld(nb), p, q, lld(lookahead), int(target) );
    }

    // for now, trmm requires full tiles
    assert(m % nb == 0);
    assert(n % nb == 0);

    // setup so op(B) is m-by-n
    int64_t An  = (side == blas::Side::Left ? m : n);
    int64_t Bm  = (opB == Op::NoTrans ? m : n);
    int64_t Bn  = (opB == Op::NoTrans ? n : m);
    int64_t lda = An;
    int64_t ldb = Bm;

    scalar_t alpha;
    int64_t seed[] = {0, 1, 2, 3};
    lapack::larnv(1, seed, 1, &alpha);

    scalar_t *A1 = nullptr;
    scalar_t *B1 = nullptr;
    scalar_t *B2 = nullptr;

    A1 = new scalar_t[ lda*An ];
    lapack::larnv(1, seed, lda*An, A1);

    // set unused data to nan
    scalar_t nan_ = nan("");
    if (uplo == blas::Uplo::Lower) {
        lapack::laset( lapack::MatrixType::Upper, An-1, An-1, nan_, nan_,
                       &A1[ 0 + 1*lda ], lda );
    }
    else {
        lapack::laset( lapack::MatrixType::Lower, An-1, An-1, nan_, nan_,
                       &A1[ 1 + 0*lda ], lda );
    }

    B1 = new scalar_t[ ldb*Bn ];
    lapack::larnv(1, seed, ldb*Bn, B1);

    if (test) {
        if (mpi_rank == 0) {
            B2 = new scalar_t[ ldb*Bn ];
            memcpy(B2, B1, sizeof(scalar_t)*ldb*Bn);
        }
    }

    slate::TriangularMatrix<scalar_t> A(uplo,
                                        An, A1, lda,
                                        nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> B(Bm, Bn, B1, ldb, nb, p, q, MPI_COMM_WORLD);

    if (opA == Op::Trans)
        A = transpose( A );
    else if (opA == Op::ConjTrans)
        A = conj_transpose( A );

    if (opB == Op::Trans)
        B = transpose( B );
    else if (opB == Op::ConjTrans)
        B = conj_transpose( B );

    if (verbose && mpi_rank == 0) {
        printf( "alpha = %.4f + %.4fi;\n",
                real(alpha), imag(alpha) );
        print( "A1", An, An, A1, lda );
        print( "A",  A );
        print( "B1", Bm, Bn, B1, ldb );
        print( "B",  B );
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
            slate::trmm<slate::Target::HostTask>(
                side, diag,
                alpha, A, B, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostNest:
            slate::trmm<slate::Target::HostNest>(
                side, diag,
                alpha, A, B, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::HostBatch:
            slate::trmm<slate::Target::HostBatch>(
                side, diag,
                alpha, A, B, {{slate::Option::Lookahead, lookahead}});
            break;
        case slate::Target::Devices:
            slate::trmm<slate::Target::Devices>(
                side, diag,
                alpha, A, B, {{slate::Option::Lookahead, lookahead}});
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
        print( "B1res", Bm, Bn, B1, ldb );
        print( "Bres", B );
    }

    assert( B.mt() == slate::ceildiv( m, nb ));
    assert( B.nt() == slate::ceildiv( n, nb ));

    //--------------
    // Print GFLOPS.
    if (mpi_rank == 0) {
        double ops = blas::Gflop<scalar_t>::trmm( side, m, n );
        double gflops = ops/time;
        printf("\t%.0f GFLOPS\n", gflops);
        fflush(stdout);
    }

    //------------------
    // Test correctness.
    if (test) {
        B.gather(B1, ldb);

        if (mpi_rank == 0) {
            if (opB == Op::NoTrans) {
                blas::trmm(blas::Layout::ColMajor,
                           side, uplo, opA, diag,
                           Bm, Bn,
                           alpha, A1, lda,
                                  B2, ldb);
            }
            else {
                // transposed B: swap left <=> right
                blas::Side side2 = (side == blas::Side::Left
                        ? blas::Side::Right : blas::Side::Left);

                blas::Op op2;
                if (opA == Op::NoTrans)
                    op2 = opB;
                else if (opA == opB || A.is_real)
                    op2 = Op::NoTrans;
                else
                    throw std::exception();

                if (opB == Op::ConjTrans)
                    alpha = conj(alpha);

                blas::trmm(blas::Layout::ColMajor,
                           side2, uplo, op2, diag,
                           Bm, Bn,
                           alpha, A1, lda,
                                  B2, ldb);
            }

            if (verbose && mpi_rank == 0) {
                print( "Bref", Bm, Bn, B2, ldb );
            }
            if (verbose)
                slate::Debug::diffLapackMatrices(Bm, Bn, B1, ldb, B2, ldb, nb, nb);

            blas::axpy((size_t)ldb*Bn, -1.0, B1, 1, B2, 1);
            real_t norm =
                lapack::lange(lapack::Norm::Fro, Bm, Bn, B1, ldb);

            real_t error =
                lapack::lange(lapack::Norm::Fro, Bm, Bn, B2, ldb);

            if (norm != 0)
                error /= norm;

            real_t eps = std::numeric_limits< real_t >::epsilon();
            bool okay = (error < 50*eps);
            printf("\t%.2e error, %s\n", error, okay ? "ok" : "failed");

            delete[] B2;
            B2 = nullptr;
        }
    }
    delete[] B1;
    B1 = nullptr;
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
    if (argc < 11 && mpi_rank == 0) {
        printf("Usage: %s {Left,Right} {Upper,Lower} {Notrans,Trans,Conjtrans} {Notrans,Trans,Conjtrans} {Nonunit,Unit} m n nb p q lookahead [HostTask|HostNest|HostBatch|Devices] [s|d|c|z] [test] [verbose] [trace]\n"
               "For side, uplo, opA, opB, diag, only the first letter is used.\n", argv[0]);
        return EXIT_FAILURE;
    }

    int arg = 1;
    blas::Side side  = blas::char2side( argv[arg][0] ); ++arg;
    blas::Uplo uplo  = blas::char2uplo( argv[arg][0] ); ++arg;
    blas::Op   opA   = blas::char2op  ( argv[arg][0] ); ++arg;
    blas::Op   opB   = blas::char2op  ( argv[arg][0] ); ++arg;
    blas::Diag diag  = blas::char2diag( argv[arg][0] ); ++arg;
    int64_t m  = atol(argv[arg]); ++arg;
    int64_t n  = atol(argv[arg]); ++arg;
    int64_t nb = atol(argv[arg]); ++arg;
    int64_t p  = atol(argv[arg]); ++arg;
    int64_t q  = atol(argv[arg]); ++arg;
    int64_t lookahead = atol(argv[arg]); ++arg;

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
            test_trmm< float >( side, uplo, opA, opB, diag, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'd':
            test_trmm< double >( side, uplo, opA, opB, diag, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'c':
            test_trmm< std::complex<float> >( side, uplo, opA, opB, diag, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        case 'z':
            test_trmm< std::complex<double> >( side, uplo, opA, opB, diag, m, n, nb, p, q, lookahead, target, test, verbose, trace );
            break;
        default:
            printf( "unknown datatype: %c\n", datatype );
            break;
    }

    //--------------------
    MPI_Finalize();
    return EXIT_SUCCESS;
}
