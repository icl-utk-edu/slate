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
template <typename scalar_t> void test_gesv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Op trans = slate::Op::NoTrans;
    if (params.routine == "getrs")
        trans = params.trans();

    int64_t m;
    if (params.routine == "getrf")
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
    int verbose = params.verbose(); SLATE_UNUSED(verbose);
    int matrix = params.matrix();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

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

    if (params.routine == "gesvMixed") {
        if (! std::is_same<real_t, double>::value) {
            if (mpi_rank == 0) {
                printf("Unsupported mixed precision\n");
            }
            return;
        }
    }

    // Local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9];
    int descB_tst[9], descB_ref[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

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
    scalapack_pplrnt(&A_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], n, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // allocate ipiv locally
    size_t ipiv_size = (size_t)(lldA + nb);
    std::vector<int> ipiv_tst(ipiv_size);

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::Matrix<scalar_t> A0(m, n, nb, p, q, MPI_COMM_WORLD);

    slate::Matrix<scalar_t> A, B, X;
    std::vector<scalar_t> X_tst;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        B = slate::Matrix<scalar_t>(n, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);
        copy(&B_tst[0], descB_tst, B);

        if (params.routine == "gesvMixed") {
            if (std::is_same<real_t, double>::value) {
                X_tst.resize(lldB*nlocB);
                X = slate::Matrix<scalar_t>(n, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
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

    if (matrix == 1) {
        // Make A diagonally dominant to avoid pivoting.
        printf("diag dominant\n");
        for (int k = 0; k < std::min(A.mt(), A.nt()); ++k) {
            auto T = A(k, k);
            for (int i = 0; i < T.nb(); ++i) {
                T.at(i, i) += n;
            }
        }
    }

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    std::vector<scalar_t> B_ref;
    std::vector<scalar_t> B_orig;
    std::vector<int> ipiv_ref;
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, m, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);

        B_ref = B_tst;
        scalapack_descinit(descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
        slate_assert(info == 0);

        if (check && ref)
            B_orig = B_tst;

        ipiv_ref.resize(ipiv_tst.size());
    }

    int iters = 0;

    double gflop;
    if (params.routine == "getrf")
        gflop = lapack::Gflop<scalar_t>::getrf(m, n);
    else if (params.routine == "getrs")
        gflop = lapack::Gflop<scalar_t>::getrs(n, nrhs);
    else
        gflop = lapack::Gflop<scalar_t>::gesv(n, nrhs);

    if (! ref_only) {
        if (params.routine == "getrs") {
            // Factor matrix A.
            slate::getrf(A, pivots, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target},
                {slate::Option::MaxPanelThreads, panel_threads},
                {slate::Option::InnerBlocking, ib}
            });
        }

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time = libtest::get_wtime();

        //==================================================
        // Run SLATE test.
        // One of:
        // getrf: Factor PA = LU.
        // getrs: Solve AX = B after factoring A above.
        // gesv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "getrf") {
            slate::getrf(A, pivots, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target},
                {slate::Option::MaxPanelThreads, panel_threads},
                {slate::Option::InnerBlocking, ib}
            });
        }
        else if (params.routine == "getrs") {
            auto opA = A;
            if (trans == slate::Op::Trans)
                opA = transpose(A);
            else if (trans == slate::Op::ConjTrans)
                opA = conj_transpose(A);

            slate::getrs(opA, pivots, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }
        else if (params.routine == "gesv") {
            slate::gesv(A, pivots, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target},
                {slate::Option::MaxPanelThreads, panel_threads},
                {slate::Option::InnerBlocking, ib}
            });
        }
        else if (params.routine == "gesvMixed") {
            if (std::is_same<real_t, double>::value) {
                slate::gesvMixed(A, pivots, B, X, iters, {
                    {slate::Option::Lookahead, lookahead},
                    {slate::Option::Target, target},
                    {slate::Option::MaxPanelThreads, panel_threads},
                    {slate::Option::InnerBlocking, ib}
                });
            }
        }
        else {
            slate_error("Unknown routine!");
        }

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = libtest::get_wtime() - time;

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
            slate::getrs(A, pivots, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }

        if (origin != slate::Origin::ScaLAPACK) {
            // Copy SLATE result back from GPU or CPU tiles.
            if (params.routine == "gesvMixed") {
                if (std::is_same<real_t, double>::value) {
                    copy(X, &X_tst[0], descB_tst);
                }
            }
            else {
                copy(B, &B_tst[0], descB_tst);
            }
        }

        // allocate work space
        std::vector<real_t> worklangeA(std::max(mlocA, nlocA));
        std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

        // Norm of original matrix: || A ||_1
        real_t A_norm = scalapack_plange("1", m, n, &A_ref[0], ione, ione, descA_ref, &worklangeA[0]);
        // Norm of updated rhs matrix: || X ||_1
        real_t X_norm = scalapack_plange("1", n, nrhs, &B_tst[0], ione, ione, descB_tst, &worklangeB[0]);

        // B_ref -= op(Aref)*B_tst
        if (params.routine == "gesvMixed") {
            if (std::is_same<real_t, double>::value) {
                scalapack_pgemm(op2str(trans), "notrans",
                                n, nrhs, n,
                                scalar_t(-1.0),
                                &A_ref[0], ione, ione, descA_ref,
                                &X_tst[0], ione, ione, descB_tst,
                                scalar_t(1.0),
                                &B_ref[0], ione, ione, descB_ref);
            }
        }
        else {
            scalapack_pgemm(op2str(trans), "notrans",
                            n, nrhs, n,
                            scalar_t(-1.0),
                            &A_ref[0], ione, ione, descA_ref,
                            &B_tst[0], ione, ione, descB_tst,
                            scalar_t(1.0),
                            &B_ref[0], ione, ione, descB_ref);
        }

        // Norm of residual: || B - AX ||_1
        real_t R_norm = scalapack_plange("1", n, nrhs, &B_ref[0], ione, ione, descB_ref, &worklangeB[0]);
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

        if (check) {
            // restore B_ref
            B_ref = B_orig;
            scalapack_descinit(descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
            slate_assert(info == 0);
        }

        if (params.routine == "getrs") {
            // Factor matrix A.
            scalapack_pgetrf(m, n, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &info_ref);
            slate_assert(info_ref == 0);
        }

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        double time = libtest::get_wtime();
        if (params.routine == "getrf") {
            scalapack_pgetrf(m, n, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &info_ref);
        }
        else if (params.routine == "getrs") {
            scalapack_pgetrs(op2str(trans), n, nrhs, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &B_ref[0], ione, ione, descB_ref, &info_ref);
        }
        else {
            scalapack_pgesv(n, nrhs, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], &B_ref[0], ione, ione, descB_ref, &info_ref);
        }
        slate_assert(info_ref == 0);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

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
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gesv_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_gesv_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_gesv_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_gesv_work<std::complex<double>> (params, run);
            break;
    }
}
