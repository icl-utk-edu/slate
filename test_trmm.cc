
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
int main (int argc, char *argv[])
{
    if (argc < 11) {
        printf("Usage: %s side{l,r} uplo{u,l} op{n,t,c} diag{n,u} m n nb p q lookahead [test] [verbose] [trace]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int arg = 1;
    blas::Side side  = blas::char2side( argv[arg][0] ); arg += 1;
    blas::Uplo uplo  = blas::char2uplo( argv[arg][0] ); arg += 1;
    blas::Op   op    = blas::char2op  ( argv[arg][0] ); arg += 1;
    blas::Diag diag  = blas::char2diag( argv[arg][0] ); arg += 1;
    int64_t m  = atol(argv[arg]); arg += 1;
    int64_t n  = atol(argv[arg]); arg += 1;
    int64_t nb = atol(argv[arg]); arg += 1;
    int64_t p  = atol(argv[arg]); arg += 1;
    int64_t q  = atol(argv[arg]); arg += 1;
    int64_t lookahead = atol(argv[arg]); arg += 1;
    bool test    = argc > arg && std::string(argv[arg]) == "test";    arg += 1;
    bool verbose = argc > arg && std::string(argv[arg]) == "verbose"; arg += 1;
    bool trace   = argc > arg && std::string(argv[arg]) == "trace";   arg += 1;

    printf( "side=%c, uplo=%c, op=%c, diag=%c, m=%lld, n=%lld, nb=%lld, p=%lld, q=%lld, lookahead=%lld\n",
            char(side), char(uplo), char(op), char(diag), m, n, nb, p, q, lookahead );
    // for now, trmm requires full tiles
    assert(m % nb == 0);
    assert(n % nb == 0);

    int64_t An  = (side == blas::Side::Left ? m : n);
    int64_t lda = An;
    int64_t ldb = m;

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

    int64_t seed_a[] = {0, 1, 0, 3};
    A1 = new double[ lda*An ];
    lapack::larnv(1, seed_a, lda*An, A1);

    // set unused data to nan
    if (uplo == blas::Uplo::Lower) {
        for (int j = 0; j < An; ++j)
            for (int i = 0; i < j && i < An; ++i)  // upper
                A1[ i + j*lda ] = nan("");
    }
    else {
        for (int j = 0; j < An; ++j)
            for (int i = j+1; i < An; ++i)  // lower
                A1[ i + j*lda ] = nan("");
    }

    int64_t seed_c[] = {0, 0, 0, 1};
    B1 = new double[ ldb*n ];
    lapack::larnv(1, seed_c, ldb*n, B1);

    if (test) {
        if (mpi_rank == 0) {
            B2 = new double[ ldb*n ];
            memcpy(B2, B1, sizeof(double)*ldb*n);
        }
    }

    slate::TriangularMatrix<double> A(uplo,
                                      An, A1, lda,
                                      nb, p, q, MPI_COMM_WORLD);
    if (op == blas::Op::Trans)
        A = transpose( A );
    else if (op == blas::Op::ConjTrans)
        A = conj_transpose( A );
    slate::Matrix<double> B(m, n, B1, ldb, nb, p, q, MPI_COMM_WORLD);
    if (verbose && mpi_rank == 0) {
        printf( "A = " ); print( An, An, A1, lda );
        printf( "A = " ); print( A );
        printf( "B = " ); print( m, n, B1, ldb );
        printf( "B = " ); print( B );
    }
    if (trace)
        slate::trace::Trace::on();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double start = omp_get_wtime();
    slate::trmm<slate::Target::HostTask>(
        side, diag,
        alpha, A, B, {{slate::Option::Lookahead, lookahead}});

    MPI_Barrier(MPI_COMM_WORLD);
    double time = omp_get_wtime() - start;

    if (trace)
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
        B.gather(B1, ldb);

        if (mpi_rank == 0) {
            blas::trmm(blas::Layout::ColMajor,
                       side, uplo, op, diag,
                       m, n,
                       alpha, A1, lda,
                              B2, ldb);

            if (verbose && mpi_rank == 0) {
                printf( "Bresult = " ); print( B );
                printf( "Bref = "    ); print( m, n, B2, ldb );
            }
            if (verbose)
                slate::Debug::diffLapackMatrices(m, n, B1, ldb, B2, ldb, nb, nb);

            blas::axpy((size_t)ldb*n, -1.0, B1, 1, B2, 1);
            double norm =
                lapack::lange(lapack::Norm::Fro, m, n, B1, ldb);

            double error =
                lapack::lange(lapack::Norm::Fro, m, n, B2, ldb);

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
