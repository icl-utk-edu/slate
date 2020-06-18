#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_ge2tb_work(Params& params, bool run)
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

    origin = slate::Origin::ScaLAPACK;  // todo: for now

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

    const int myrow = whoismyrow(mpi_rank, p);
    const int mycol = whoismycol(mpi_rank, p);

    // matrix A, figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(m, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lldA   = mlocal;
    std::vector<scalar_t> A_data(lldA*nlocal);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());

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
    double time = testsweeper::get_wtime();

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::ge2tb(A, TU, TV, {
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

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

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, A_ref);

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

        // Form UB, where U's representation is in lower part of A and TU.
        slate::qr_multiply_by_q(
            slate::Side::Left, slate::Op::NoTrans, A, TU, B,
            {{slate::Option::Target, target}}
        );
        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::unmqr(slate::Side::Left, slate::Op::NoTrans, A, TU, B,
        //              {{slate::Option::Target, target}});

        if (verbose > 1) {
            print_matrix("UB", B);
        }

        // Form (UB)V^H, where V's representation is above band in A and TV.
        auto Asub =  A.sub(0, A.mt()-1, 1, A.nt()-1);
        auto Bsub =  B.sub(0, B.mt()-1, 1, B.nt()-1);
        slate::TriangularFactors<scalar_t> TVsub = {
            TV[0].sub(0, TV[0].mt()-1, 1, TV[0].nt()-1),
            TV[1].sub(0, TV[1].mt()-1, 1, TV[1].nt()-1)
        };

        // Note V^H == Q, not Q^H.
        slate::lq_multiply_by_q(
            slate::Side::Right, slate::Op::NoTrans, Asub, TVsub, Bsub,
            {{slate::Option::Target, target}}
        );
        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::unmlq(slate::Side::Right, slate::Op::NoTrans,
        //              Asub, TVsub, Bsub,
        //              {{slate::Option::Target, target}});

        if (verbose > 1) {
            print_matrix("UBV^H", B);
        }

        // Form UBV^H - A, where A is in A_ref.
        slate::geadd(-one, A_ref, one, B);
        if (verbose > 1) {
            print_matrix("UBV^H - A", B);
        }

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
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ge2tb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_ge2tb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_ge2tb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ge2tb_work<std::complex<double>> (params, run);
            break;
    }
}
