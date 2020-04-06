#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "aux/Debug.hh"
#include "grid_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hegst_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    int64_t itype = params.itype();
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Target target = params.target();
    slate::Origin origin = params.origin();

    params.time();
    params.ref_time();
    // params.gflops(); // todo
    // params.ref_gflops(); // todo

    origin = slate::Origin::ScaLAPACK;  // todo: for now

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) { // todo
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        // auto A = slate::HermitianMatrix<scalar_t>(
        //                          uplo, n, nb, p, q, MPI_COMM_WORLD);
        // A.insertLocalTiles(origin2target(origin));
        // todo: need ScaLAPACK descriptor for copy.
        //copy(A_data.data(), descA_tst, A);
        assert(false);
    }

    // MPI variables
    int mpi_rank;
    slate_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    int mpi_size;
    slate_mpi_call(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert(p*q <= mpi_size);

    const int myrow = whoismyrow(mpi_rank, p);
    const int mycol = whoismycol(mpi_rank, p);

    // Figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(n, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lld   = mlocal;
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };

    // Matrix A
    std::vector<scalar_t> A_data(lld*nlocal);
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, n, A_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Matrix A_ref
    std::vector<scalar_t> A_ref_data(lld*nlocal);
    A_ref_data = A_data;
    auto A_ref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, n, A_ref_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    if (verbose > 2) {
        print_matrix("A_ref", A_ref);
    }

    // Matrix B
    std::vector<scalar_t> B_data(lld*nlocal);
    lapack::larnv(idist, iseed, B_data.size(), B_data.data());
    auto B = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, n, B_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    // Make B positive-definite
    for (int64_t i = 0; i < B.mt(); ++i) {
        if (B.tileIsLocal(i, i)) {
            auto Bii = B(i, i);
            for (int64_t ii = 0; ii < std::min(Bii.mb(), Bii.nb()); ++ii) {
                Bii.at(ii, ii) = std::abs(Bii.at(ii, ii)) + n;
            }
        }
    }

    if (verbose > 1) {
        print_matrix("B", B);
    }

    // Factorize B
    slate::potrf(B, {{slate::Option::Lookahead, lookahead},
                     {slate::Option::Target,    target}});

    if (verbose > 2) {
        print_matrix("B_factored", B);
    }

    if (! ref_only) {
        // todo
        //double gflop = lapack::Gflop<scalar_t>::hegst(n);

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
        slate::hegst(itype, A, B, {{slate::Option::Lookahead, lookahead},
                                   {slate::Option::Target,    target}});
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
            print_matrix("A_hegst", A);
        }
    }

    if (check || ref) {
        real_t A_norm = slate::norm(slate::Norm::One, A_ref);

        int ictxt;
        Cblacs_get(-1, 0, &ictxt);
        Cblacs_gridinit(&ictxt, "Col", p, q);

        const int izero = 0;
        int descA[9], descB[9], info;
        scalapack_descinit(
            descA, n, n, nb, nb, izero, izero, ictxt, mlocal, &info);
        slate_assert(info == 0);
        scalapack_descinit(
            descB, n, n, nb, nb, izero, izero, ictxt, mlocal, &info);
        slate_assert(info == 0);
        const int64_t ione = 1;
        double scale;

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================

        slate_mpi_call(MPI_Barrier(MPI_COMM_WORLD));
        double time = testsweeper::get_wtime();

        scalapack_phegst(itype, uplo2str(uplo), n,
            A_ref_data.data(), ione, ione, descA,
            B_data.data(),     ione, ione, descB,
            &scale, &info);
        slate_assert(info == 0);

        slate_mpi_call(MPI_Barrier(MPI_COMM_WORLD));
        double time_ref = testsweeper::get_wtime() - time;

        params.ref_time() = time_ref;
        // params.ref_gflops() = gflop / time_ref;

        if (verbose > 1) {
            print_matrix("A_ref_hegst", A_ref);
        }

        slate_set_num_blas_threads(saved_num_threads);

        if (! ref_only) {
            // Local operation: error = A_ref - A
            blas::axpy(
                A_ref_data.size(), scalar_t(-1.0),
                A_data.data(), 1,
                A_ref_data.data(), 1);

            params.error() = slate::norm(slate::Norm::One, A_ref) / (n * A_norm);
            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
            params.okay() = (params.error() <= tol);
        }
        Cblacs_gridexit(ictxt);
    }
}

// -----------------------------------------------------------------------------
void test_hegst(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hegst_work<float>(params, run);
            break;

        case testsweeper::DataType::Double:
            test_hegst_work<double>(params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hegst_work<std::complex<float>>(params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hegst_work<std::complex<double>>(params, run);
            break;
    }
}
