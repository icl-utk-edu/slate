#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_getri_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
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
    int verbose = params.verbose(); SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9];
    int descC_chk[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // scalapack matrix A_tst, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::Matrix<scalar_t> A0(n, n, nb, p, q, MPI_COMM_WORLD);

    // Setup SLATE matrix A based on scalapack matrix/data in A_tst
    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(n, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(n, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    // Create pivot structure to store pivots after factoring
    slate::Pivots pivots;

    // if check (or ref) is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);
    }

    // If check is required: record the norm(A original)
    real_t A_norm = 0.0;
    if (check) A_norm = slate::norm(slate::Norm::One, A);

    // initialize C_chk; space to hold A*inv(A); also used for out-of-place algorithm
    std::vector<scalar_t> C_chk;
    C_chk = A_tst;
    scalapack_descinit(descC_chk, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);

    // C will be used as storage for out-of-place algorithm
    slate::Matrix<scalar_t> C;
    // todo: Select correct times to use out-of-place getri, currently always use
    if (params.routine == "getriOOP") {
        // setup SLATE matrix C based on scalapack matrix/data in C_chk
        if (origin != slate::Origin::ScaLAPACK) {
            // Copy local ScaLAPACK data to GPU or CPU tiles.
            slate::Target origin_target = origin2target(origin);
            C = slate::Matrix<scalar_t>(n, n, nb, nprow, npcol, MPI_COMM_WORLD);
            C.insertLocalTiles(origin_target);
            copy(&C_chk[0], descC_chk, C);
        }
        else {
            // Create SLATE matrix from the ScaLAPACK layouts
            C = slate::Matrix<scalar_t>::fromScaLAPACK(n, n, &C_chk[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        }
    }

    // the timing includes getrf and getri
    double gflop = lapack::Gflop<scalar_t>::getrf(n, n) + lapack::Gflop<scalar_t>::getri(n);

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
        // factor then invert; measure time for both
        slate::lu_factor(A, pivots, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target},
            {slate::Option::MaxPanelThreads, panel_threads},
            {slate::Option::InnerBlocking, ib}
        });

        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::getrf(A, pivots, {
        //     {slate::Option::Lookahead, lookahead},
        //     {slate::Option::Target, target},
        //     {slate::Option::MaxPanelThreads, panel_threads},
        //     {slate::Option::InnerBlocking, ib}
        // });

        if (params.routine == "getri") {
            // call in-place inversion
            slate::lu_inverse_using_factor(A, pivots, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getri(A, pivots, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }
        else if (params.routine == "getriOOP") {
            // Call the out-of-place version; on exit, C = inv(A), A unchanged
            slate::lu_inverse_using_factor_out_of_place(A, pivots, C, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target},
                {slate::Option::MaxPanelThreads, panel_threads},
                {slate::Option::InnerBlocking, ib}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getri(A, pivots, C, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target},
            //     {slate::Option::MaxPanelThreads, panel_threads},
            //     {slate::Option::InnerBlocking, ib}
            // });
        }

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;
    }

    if (check) {
        //==================================================
        // Check  || I - inv(A)*A || / ( || A || * N ) <=  tol * eps

        if (origin != slate::Origin::ScaLAPACK) {
            // Copy SLATE result back from GPU or CPU tiles.
            copy(A, &A_tst[0], descA_tst);
            copy(C, &C_chk[0], descC_chk);
        }

        // Copy inv(A) from oop vector storage C_chk to expected location A_tst
        // After this, A_tst contains the inv(A)
        if (params.routine == "getriOOP") {
            A_tst = C_chk;
        }

        // For check make C_chk a identity matrix to check the result of multiplying A and A_inv
        scalar_t zero = 0.0; scalar_t one = 1.0;
        scalapack_plaset("All", n, n, zero, one, &C_chk[0], ione, ione, descC_chk);

        // C_chk has been setup as an identity matrix; C_chk = C_chk - inv(A)*A
        scalar_t alpha = -1.0; scalar_t beta = 1.0;
        scalapack_pgemm("notrans", "notrans", n, n, n, alpha,
                        &A_tst[0], ione, ione, descA_tst,
                        &A_ref[0], ione, ione, descA_ref, beta,
                        &C_chk[0], ione, ione, descC_chk);

        // Norm of C_chk ( = I - inv(A) * A )
        std::vector<real_t> worklange(n);
        real_t C_norm = scalapack_plange("One", n, n, &C_chk[0], ione, ione, descC_chk, &worklange[0]);

        real_t A_inv_norm = scalapack_plange("One", n, n, &A_tst[0], ione, ione, descA_tst, &worklange[0]);

        double residual = C_norm / (A_norm * n * A_inv_norm);
        params.error() = residual;

        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref) {
        // todo: call to reference getri from ScaLAPACK not implemented
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_getri(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_getri_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_getri_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_getri_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_getri_work<std::complex<double>> (params, run);
            break;
    }
}
