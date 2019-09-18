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
int64_t localRowsCols(int64_t n, int64_t nb, int iproc, int mpi_size)
{
    int64_t nblocks = n / nb;
    int64_t num = (nblocks / mpi_size) * nb;
    int64_t extra_blocks = nblocks % mpi_size;
    if (iproc < extra_blocks) {
        // extra full blocks
        num += nb;
    }
    else if (iproc == extra_blocks) {
        // last partial block
        num += n % nb;
    }
    return num;
}

//------------------------------------------------------------------------------
template <typename scalar_t> void test_ge2tb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    //using llong = long long;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
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

    // Local values
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // MPI variables
    int mpi_rank, mpi_size;
    slate_mpi_call(
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert(p*q <= mpi_size);

    int myrow = mpi_rank % p;
    int mycol = mpi_rank / p;

bool debug = verbose;
if (debug) printf( "rank %2d, init A\n", mpi_rank );
    // matrix A, figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(m, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lldA   = mlocal;
    std::vector<scalar_t> A_data(lldA*nlocal);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());

if (debug) printf( "rank %2d, init A (2)\n", mpi_rank );
    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        A = slate::Matrix<scalar_t>(m, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin2target(origin));
        // todo: need ScaLAPACK descriptor for copy. hmpf!
        //copy(A_data.data(), descA_tst, A);
        assert(false);
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);
    }
    slate::TriangularFactors<scalar_t> TU, TV;

    if (verbose > 1) {
        print_matrix("A", A);
    }

if (debug) printf( "rank %2d, init A_ref\n", mpi_rank );
    // Copy test data for check.
    slate::Matrix<scalar_t> A_ref(m, n, nb, p, q, MPI_COMM_WORLD);
    A_ref.insertLocalTiles();
    slate::copy(A, A_ref);

    // todo
    //double gflop = lapack::Gflop<scalar_t>::ge2tb(m, n);

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = libtest::get_wtime();

if (debug) printf( "rank %2d, ge2tb\n", mpi_rank );
    //==================================================
    // Run SLATE test.
    //==================================================
    slate::ge2tb(A, TU, TV, {
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });
if (debug) printf( "rank %2d, ge2tb done\n", mpi_rank );

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = libtest::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time_tst;
    //params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        print_matrix("A_factored", A);
        print_matrix("TUlocal",  TU[0]);
        print_matrix("TUreduce", TU[1]);
        print_matrix("TVlocal",  TV[0]);
        print_matrix("TVreduce", TV[1]);
    }

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || UBV^H - A ||_1
        //     ------------------- < tol * epsilon
        //      || A ||_1 * m
        //
        //==================================================

if (debug) printf( "rank %2d, A norm\n", mpi_rank );
        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, A_ref);

if (debug) printf( "rank %2d, copy B\n", mpi_rank );
        // Zero out B, then copy band matrix B from A.
        slate::Matrix<scalar_t> B = A.emptyLike();
        B.insertLocalTiles();
        set(zero, B);
        int64_t min_mtnt = std::min(A.mt(), A.nt());
        for (int64_t i = 0; i < min_mtnt; ++i) {
            if (B.tileIsLocal(i, i)) {
                // diagonal tile
                auto Aii = A(i, i);
                auto Bii = B(i, i);
                Aii.uplo(slate::Uplo::Upper);
                Bii.uplo(slate::Uplo::Upper);
                tzcopy(Aii, Bii);
            }
            if (i+1 < min_mtnt && B.tileIsLocal(i, i+1)) {
                // super-diagonal tile
                auto Aii1 = A(i, i+1);
                auto Bii1 = B(i, i+1);
                Aii1.uplo(slate::Uplo::Lower);
                Bii1.uplo(slate::Uplo::Lower);
                tzcopy(Aii1, Bii1);
            }
        }
        if (verbose > 1) {
            print_matrix("B", B);
        }

if (debug) printf( "rank %2d, unmqr\n", mpi_rank );
        // Form UB, where U's representation is in lower part of A and TU.
        slate::unmqr(slate::Side::Left, slate::Op::NoTrans, A, TU, B,
                     {{slate::Option::Target, target}});
        if (verbose > 1) {
            print_matrix("UB", B);
        }

if (debug) printf( "rank %2d, unmlq\n", mpi_rank );
        // Form (UB)V^H, where V's representation is above band in A and TV.
        auto Asub =  A.sub(0, A.mt()-1, 1, A.nt()-1);
        auto Bsub =  B.sub(0, B.mt()-1, 1, B.nt()-1);
        slate::TriangularFactors<scalar_t> TVsub = {
            TV[0].sub(0, TV[0].mt()-1, 1, TV[0].nt()-1),
            TV[1].sub(0, TV[1].mt()-1, 1, TV[1].nt()-1)
        };
        // Note V^H == Q, not Q^H.
        slate::unmlq(slate::Side::Right, slate::Op::NoTrans,
                     Asub, TVsub, Bsub,
                     {{slate::Option::Target, target}});
        if (verbose > 1) {
            print_matrix("UBV^H", B);
        }

if (debug) printf( "rank %2d, geadd\n", mpi_rank );
        // Form UBV^H - A, where A is in A_ref.
        slate::geadd(-one, A_ref, one, B);
        if (verbose > 1) {
            print_matrix("UBV^H - A", B);
        }

if (debug) printf( "rank %2d, error\n", mpi_rank );
        // Norm of backwards error: || UBV^H - A ||_1
        params.error() = slate::norm(slate::Norm::One, B) / (m * A_norm);
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
        params.okay() = (params.error() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_ge2tb(Params& params, bool run)
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ge2tb_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_ge2tb_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_ge2tb_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_ge2tb_work<std::complex<double>> (params, run);
            break;
    }
}
