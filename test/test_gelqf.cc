#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_gelqf_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    // using llong = long long;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // Local values
    const scalar_t zero = 0;
    const scalar_t one = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9], descLQ_tst[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(m, nb, myrow, 0, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, 0, npcol);
    scalapack_descinit(descA_tst, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix LQ, for checking result
    std::vector<scalar_t> LQ_tst(1);
    scalapack_descinit(descLQ_tst, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
    slate_assert(info == 0);

    // tau vector for ScaLAPACK
    int64_t ltau = scalapack_numroc(std::min(m, n), nb, mycol, 0, npcol);
    std::vector<scalar_t> tau(ltau);

    // workspace for ScaLAPACK
    int64_t lwork;
    std::vector<scalar_t> work(1);

    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }
    slate::TriangularFactors<scalar_t> T;

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    slate::Matrix<scalar_t> Aref;
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, m, n, nb, nb, 0, 0, ictxt, mlocA, &info);
        slate_assert(info == 0);

        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &A_ref[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    double gflop = lapack::Gflop<scalar_t>::gelqf(m, n);

    if (! ref_only) {
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
        slate::lq_factor(A, T, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target},
            {slate::Option::MaxPanelThreads, panel_threads},
            {slate::Option::InnerBlocking, ib}
        });

        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::gelqf(A, T, {
        //     {slate::Option::Lookahead, lookahead},
        //     {slate::Option::Target, target},
        //     {slate::Option::MaxPanelThreads, panel_threads},
        //     {slate::Option::InnerBlocking, ib}
        // });

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;

        if (verbose > 1) {
            print_matrix("A_factored", A);
            print_matrix("Tlocal",  T[0]);
            print_matrix("Treduce", T[1]);
        }
    }

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || LQ - A ||_1
        //     ---------------- < tol * epsilon
        //      || A ||_1 * m
        //
        //==================================================

        if (origin != slate::Origin::ScaLAPACK) {
            // Copy SLATE result back from GPU or CPU tiles.
            copy(A, &A_tst[0], descA_tst);
        }

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        // Zero out LQ, then copy L, stored in lower triangle of A_tst.
        // todo: replace with slate set/copy functions.
        LQ_tst = std::vector<scalar_t>(A_tst.size(), zero);
        scalapack_placpy("Lower", m, std::min(m, n),
                         &A_tst[0], 1, 1, descA_tst,
                         &LQ_tst[0], 1, 1, descLQ_tst);

        // Alternatively, copy all of A_tst to LQ, then zero out above diagonal.
        //LQ_tst = A_tst;
        //scalapack_plaset("Upper", m-1, n, zero, zero,
        //                 &LQ_tst[0], 1, 2, descLQ_tst);

        auto LQ = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &LQ_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);

        if (verbose > 1) {
            print_matrix("L", LQ);
        }

        // Form LQ, where Q's representation is in A and T, and L is in LQ.
        slate::lq_multiply_by_q(
            slate::Side::Right, slate::Op::NoTrans, A, T, LQ,
            {{slate::Option::Target, target}}
        );
        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::unmlq(slate::Side::Right, slate::Op::NoTrans, A, T, LQ, {
        //     {slate::Option::Target, target}
        // });

        if (verbose > 1) {
            print_matrix("LQ", LQ);
        }

        // Form LQ - A, where A is in Aref.
        // todo: slate::geadd(-one, Aref, LQ);
        // using axpy assumes A_ref and LQ_tst have same lda.
        blas::axpy(LQ_tst.size(), -one, &A_ref[0], 1, &LQ_tst[0], 1);

        if (verbose > 1) {
            print_matrix("LQ - A", LQ);
        }

        // Norm of backwards error: || LQ - A ||_1
        real_t R_norm = slate::norm(slate::Norm::One, LQ);

        double residual = R_norm / (m*A_norm);
        params.error() = residual;
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref) {
        // A comparison with a reference routine from ScaLAPACK for timing only

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);
        int64_t info_ref = 0;

        // query for workspace size
        scalar_t dummy;
        scalapack_pgelqf(m, n, &A_ref[0], 1, 1, descA_ref, tau.data(),
                         &dummy, -1, &info_ref);
        lwork = int64_t( real( dummy ) );
        work.resize(lwork);

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        double time = testsweeper::get_wtime();
        scalapack_pgelqf(m, n, &A_ref[0], 1, 1, descA_ref, tau.data(),
                         work.data(), lwork, &info_ref);
        slate_assert(info_ref == 0);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        params.ref_time() = time_ref;
        params.ref_gflops() = gflop / time_ref;

        slate_set_num_blas_threads(saved_num_threads);
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_gelqf(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gelqf_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gelqf_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gelqf_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gelqf_work<std::complex<double>> (params, run);
            break;
    }
}
