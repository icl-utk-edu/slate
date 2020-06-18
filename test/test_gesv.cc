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
void test_gesv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Op trans = slate::Op::NoTrans;
    if (params.routine == "getrs" || params.routine == "getrs_nopiv")
        trans = params.trans();

    int64_t m;
    if (params.routine == "getrf" || params.routine == "getrf_nopiv")
        m = params.dim.m();
    else
        m = params.dim.n();  // square, n-by-n

    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
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
    bool nonuniform_nb = params.nonuniform_nb() == 'y';
    int verbose = params.verbose(); SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (params.routine == "gesvMixed") {
        params.iters();
    }
    if (! run)
        return;

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (nonuniform_nb) {
        if (ref || origin == slate::Origin::ScaLAPACK) {
            if (mpi_rank == 0) {
                printf("Unsupported to test nonuniform tile size using scalapack\n");
            }
        }
        params.ref() = 'n';
        params.origin() = slate::Origin::Host;
        ref = false;
        origin = slate::Origin::Host;
    }

    if (params.routine == "gesvMixed") {
        if (! std::is_same<real_t, double>::value) {
            if (mpi_rank == 0) {
                printf("Unsupported mixed precision\n");
            }
            return;
        }
    }

    slate::Options const opts =  {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    };

    // Local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descB_tst[9];
    int iam = 0, nprocs = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(m, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, m, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);

    // allocate ipiv locally
    size_t ipiv_size = (size_t)(lldA + nb);
    std::vector<int> ipiv_tst(ipiv_size);


    // To generate matrix with non-uniform tile size using the Lambda constructor
    std::function< int64_t (int64_t j) >
    tileNb = [n, nb](int64_t j)
    {
        // for non-uniform tile size
        return (j % 2 != 0 ? nb/2 : nb);
    };

    // 2D block column cyclic
    std::function< int (std::tuple<int64_t, int64_t> ij) >
    tileRank = [p, q](std::tuple<int64_t, int64_t> ij)
    {
        int64_t i = std::get<0>(ij);
        int64_t j = std::get<1>(ij);
        return int(i%p + (j%q)*p);
    };

    // 1D block row cyclic
    int num_devices_ = 0;//num_devices;
    std::function< int (std::tuple<int64_t, int64_t> ij) >
    tileDevice = [num_devices_](std::tuple<int64_t, int64_t> ij)
    {
        int64_t i = std::get<0>(ij);
        return int(i)%num_devices_;
    };


    slate::Matrix<scalar_t> A, B, X;
    std::vector<scalar_t> X_tst;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        if (nonuniform_nb) {
            A = slate::Matrix<scalar_t>(m, n, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>(n, nrhs, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
        }
        else {
            A = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>(n, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
        }
        A.insertLocalTiles(origin_target);
        B.insertLocalTiles(origin_target);

        if (params.routine == "gesvMixed") {
            if (std::is_same<real_t, double>::value) {
                X_tst.resize(lldB*nlocB);
                if (nonuniform_nb) {
                    X = slate::Matrix<scalar_t>(n, nrhs, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
                }
                else {
                    X = slate::Matrix<scalar_t>(n, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
                }
                X.insertLocalTiles(origin_target);
            }
        }
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(m, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(n, nrhs, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
        if (params.routine == "gesvMixed") {
            if (std::is_same<real_t, double>::value) {
                X_tst.resize(lldB*nlocB);
                X = slate::Matrix<scalar_t>::fromScaLAPACK(n, nrhs, &X_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
            }
        }
    }

    slate::Pivots pivots;

    slate::generate_matrix(params.matrix, A);
    slate::generate_matrix(params.matrix, B);

    if (ref && ! nonuniform_nb) {
        copy(A, &A_tst[0], descA_tst);
        copy(B, &B_tst[0], descB_tst);
    }

    slate::Matrix<scalar_t> Aref, Bref;
    if (check) {
        if (nonuniform_nb) {
            Aref = slate::Matrix<scalar_t>(m, n, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
            Bref = slate::Matrix<scalar_t>(n, nrhs, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
        }
        else {
            Aref = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
            Bref = slate::Matrix<scalar_t>(n, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
        }
        Aref.insertLocalTiles(origin2target(origin));
        Bref.insertLocalTiles(origin2target(origin));

        copy(A, Aref);
        copy(B, Bref);
    }

    // if check/ref is required, copy test data
    std::vector<scalar_t> A_ref, B_ref, B_orig;
    std::vector<int> ipiv_ref;
    if (check || ref) {
        A_ref = A_tst;
        B_ref = B_tst;
        if (check && ref)
            B_orig = B_tst;
        ipiv_ref.resize(ipiv_tst.size());
    }

    int iters = 0;

    double gflop;
    if (params.routine == "getrf" || params.routine == "getrf_nopiv")
        gflop = lapack::Gflop<scalar_t>::getrf(m, n);
    else if (params.routine == "getrs" || params.routine == "getrs_nopiv")
        gflop = lapack::Gflop<scalar_t>::getrs(n, nrhs);
    else
        gflop = lapack::Gflop<scalar_t>::gesv(n, nrhs);

    if (! ref_only) {
        if (params.routine == "getrs") {
            // Factor matrix A.
            slate::lu_factor(A, pivots, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrf(A, pivots, opts);
        }

        if (params.routine == "getrs_nopiv") {
            // Factor matrix A.
            slate::lu_factor_nopiv(A, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrf_nopiv(A, opts);
        }

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time = testsweeper::get_wtime();

        //==================================================
        // Run SLATE test.
        // One of:
        // getrf: Factor PA = LU.
        // getrs: Solve AX = B after factoring A above.
        // gesv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "getrf") {
            slate::lu_factor(A, pivots, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrf(A, pivots, opts);
        }
        else if (params.routine == "getrs") {
            auto opA = A;
            if (trans == slate::Op::Trans)
                opA = transpose(A);
            else if (trans == slate::Op::ConjTrans)
                opA = conjTranspose(A);

            slate::lu_solve_using_factor(opA, pivots, B, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrs(opA, pivots, B, opts);
        }
        else if (params.routine == "gesv") {
            slate::lu_solve(A, B, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::gesv(A, pivots, B, opts);
        }
        else if (params.routine == "gesvMixed") {
            if (std::is_same<real_t, double>::value) {
                slate::gesvMixed(A, pivots, B, X, iters, opts);
            }
        }
        else if (params.routine == "getrf_nopiv") {
            slate::lu_factor_nopiv(A, opts);
            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrf_nopiv(A, opts);
        }
        else if (params.routine == "getrs_nopiv") {
            auto opA = A;
            if (trans == slate::Op::Trans)
                opA = transpose(A);
            else if (trans == slate::Op::ConjTrans)
                opA = conjTranspose(A);

            slate::lu_solve_using_factor_nopiv(opA, B, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrs_nopiv(opA, B, opts);
        }
        else if (params.routine == "gesv_nopiv") {
            slate::lu_solve_nopiv(A, B, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::gesv_nopiv(A, B, opts);
        }
        else {
            slate_error("Unknown routine!");
        }

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        if (params.routine == "gesvMixed") {
            params.iters() = iters;
        }

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;
    }

    if (check) {
        //==================================================
        // Test results by checking the residual
        //
        //           || B - AX ||_1
        //     --------------------------- < tol * epsilon
        //      || A ||_1 * || X ||_1 * N
        //
        //==================================================

        if (params.routine == "getrf") {
            // Solve AX = B.
            slate::getrs(A, pivots, B, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrs(A, pivots, B, opts);
        }
        if (params.routine == "getrf_nopiv") {
            // Solve AX = B.
            slate::lu_solve_using_factor_nopiv(A, B, opts);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::getrs_nopiv(A, B, opts);
        }

        // Norm of updated-rhs/solution matrix: || X ||_1
        real_t X_norm;
        if (params.routine == "gesvMixed")
            X_norm = slate::norm(slate::Norm::One, X);
        else
            X_norm = slate::norm(slate::Norm::One, B);

        // Norm of original A matrix
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        // Apply transpose operations to the A matrix
        slate::Matrix<scalar_t> opAref;
        if (trans == slate::Op::Trans)
            opAref = slate::transpose(Aref);
        else if (trans == slate::Op::ConjTrans)
            opAref = slate::conj_transpose(Aref);
        else
            opAref = Aref;

        // B_ref -= op(Aref)*B_tst
        if (params.routine == "gesvMixed") {
            if (std::is_same<real_t, double>::value)
                slate::multiply(scalar_t(-1.0), opAref, X, scalar_t(1.0), Bref);
                //---------------------
                // Using traditional BLAS/LAPACK name
                // slate::gemm(scalar_t(-1.0), opAref, X, scalar_t(1.0), Bref);
        }
        else {
            slate::multiply(scalar_t(-1.0), opAref, B, scalar_t(1.0), Bref);
            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::gemm(scalar_t(-1.0), opAref, B, scalar_t(1.0), Bref);
        }

        // Norm of residual: || B - AX ||_1
        real_t R_norm = slate::norm(slate::Norm::One, Bref);
        double residual = R_norm / (n*A_norm*X_norm);
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

        // restore B_ref
        B_ref = B_orig;
        int descB_ref[9];
        scalapack_descinit(descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
        slate_assert(info == 0);

        // ScaLAPACK descriptor for the reference matrix
        int descA_ref[9];
        scalapack_descinit(descA_ref, m, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);

        if (params.routine == "getrs" || params.routine == "getrs_nopiv") {
            // Factor matrix A.
            scalapack_pgetrf(m, n, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &info_ref);
            slate_assert(info_ref == 0);
        }

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        double time = testsweeper::get_wtime();
        if (params.routine == "getrf" || params.routine == "getrf_nopiv") {
            scalapack_pgetrf(m, n, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &info_ref);
        }
        else if (params.routine == "getrs" || params.routine == "getrs_nopiv") {
            scalapack_pgetrs(op2str(trans), n, nrhs, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &B_ref[0], ione, ione, descB_ref, &info_ref);
        }
        else {
            scalapack_pgesv(n, nrhs, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &B_ref[0], ione, ione, descB_ref, &info_ref);
        }
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
void test_gesv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gesv_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gesv_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gesv_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gesv_work<std::complex<double>> (params, run);
            break;
    }
}
