#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
// Similar to ScaLAPACK numroc (number of rows or columns).
int64_t localRowsCols(int64_t n, int64_t nb, int iproc, int mpi_size);

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmtr_he2hb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::conj;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Side side = params.side();
    slate::Op trans = params.trans();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    //params.gflops();

    if (! run)
        return;

    //==================================================
    // quick returns:
    //==================================================

    // todo: implement none-ScaLAPACK layout.
    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported.\n");
        return;
    }
    // no formula currently
    if (trans == slate::Op::Trans) {
        printf("skipping: slate::Op::NoTrans or slate::Op::ConjTrans.\n");
        return;
    }

    // Requires a square processing grid.
    assert(p == q);
    // todo: Only lower for now because he2hb doesn't implement Uplo::Upper.
    assert(uplo == slate::Uplo::Lower);

    // Local values
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // MPI variables
    int mpi_rank, mpi_size;
    slate_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert(p*q <= mpi_size);

    int myrow = mpi_rank % p;
    int mycol = mpi_rank / p;

    // figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(n, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lld    = mlocal;
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };

    // matrix A
    std::vector<scalar_t> A_data(lld*nlocal);
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());
    // Create SLATE matrices from the ScaLAPACK layouts.
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, A_data.data(), lld, nb, p, q, MPI_COMM_WORLD);
    // Copy test data for check.
    slate::HermitianMatrix<scalar_t> A_ref(uplo, n, nb, p, q, MPI_COMM_WORLD);
    A_ref.insertLocalTiles();
    slate::copy(A, A_ref);
    if (verbose > 1) {
        print_matrix("A", A);
    }

    slate::TriangularFactors<scalar_t> T;

    slate::he2hb(A, T, {{slate::Option::Target, target}});

    if (verbose > 1) {
        print_matrix("A_factored", A);
        print_matrix("T_local"   , T[ 0 ]);
        print_matrix("T_reduce"  , T[ 1 ]);
    }

    // matrix B
    std::vector<scalar_t> B_data(lld*nlocal);
    lapack::larnv(idist, iseed, B_data.size(), B_data.data());
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                    n, n, B_data.data(), lld, nb, p, q, MPI_COMM_WORLD);
    if (verbose > 1) {
        print_matrix("B", B);
    }

    // todo
    //double gflop = lapack::Gflop<scalar_t>::unmtr_he2hb(n, n);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = testsweeper::get_wtime();

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::unmtr_he2hb(side, uplo, trans, A, T, B, {
        {slate::Option::Target, target}
    });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time_tst;
    // params.gflops() = gflop / time_tst;
}

// -----------------------------------------------------------------------------
void test_unmtr_he2hb(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmtr_he2hb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmtr_he2hb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmtr_he2hb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmtr_he2hb_work<std::complex<double>> (params, run);
            break;
    }
}
