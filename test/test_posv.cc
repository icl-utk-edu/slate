#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "aux/Debug.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_posv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose(); SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::Dist dev_dist = params.dev_dist();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (params.routine == "posvMixed") {
        params.iters();
    }

    if (! run)
        return;

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (target != slate::Target::Devices && dev_dist != slate::Dist::Col) {
        if (mpi_rank == 0)
            printf("skipping: dev_dist = Row applies only to target devices\n");
        return;
    }

    if (params.routine == "posvMixed") {
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
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplghe(&A_tst[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], n, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);


    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::HermitianMatrix<scalar_t> A0(uplo, n, nb, p, q, MPI_COMM_WORLD);

    slate::HermitianMatrix<scalar_t> A;
    slate::Matrix<scalar_t> B, X;
    std::vector<scalar_t> X_tst;
    if (origin != slate::Origin::ScaLAPACK) {
        if (dev_dist == slate::Dist::Row && target == slate::Target::Devices) {
            // slate_assert(target == slate::Target::Devices);
            // todo: doesn't work when lookahead is greater than 2
            // slate_assert(lookahead < 3);
            // std::function<int64_t (int64_t i)> tileMb = [nrhs, nb] (int64_t i)
            //    { return (i + 1)*mb > nrhs ? nrhs%mb : mb; };
            std::function<int64_t (int64_t j)> tileNb = [n, nb] (int64_t j)
                { return (j + 1)*nb > n ? n%nb : nb; };

            std::function<int (std::tuple<int64_t, int64_t> ij)>
            tileRank = [nprow, npcol](std::tuple<int64_t, int64_t> ij) {
                int64_t i = std::get<0>(ij);
                int64_t j = std::get<1>(ij);
                return int(i%nprow + (j%npcol)*nprow);
            };

            int num_devices = 0;
            cudaGetDeviceCount(&num_devices);
            slate_assert(num_devices > 0);

            std::function<int (std::tuple<int64_t, int64_t> ij)>
            tileDevice = [nprow, num_devices](std::tuple<int64_t, int64_t> ij) {
                int64_t i = std::get<0>(ij);
                return int(i/nprow)%num_devices;
            };

            A = slate::HermitianMatrix<scalar_t>(
                uplo, n, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
            B = slate::Matrix<scalar_t>(
                n, nrhs, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
        }
        else {
            // A
            A = slate::HermitianMatrix<scalar_t>(
                    uplo, n, nb, nprow, npcol, MPI_COMM_WORLD);
            // B
            B = slate::Matrix<scalar_t>(
                    n, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
        }

        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        B.insertLocalTiles(origin_target);
        copy(&B_tst[0], descB_tst, B);

        if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value) {
                X_tst.resize(lldB*nlocB);
                X = slate::Matrix<scalar_t>(n, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
                X.insertLocalTiles(origin_target);
            }
        }
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(n, nrhs, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
        if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value) {
                X_tst.resize(lldB*nlocB);
                X = slate::Matrix<scalar_t>::fromScaLAPACK(n, nrhs, &X_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
            }
        }
    }

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    std::vector<scalar_t> B_ref;
    std::vector<scalar_t> B_orig;
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);

        B_ref = B_tst;
        scalapack_descinit(descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
        slate_assert(info == 0);

        if (check && ref)
            B_orig = B_tst;
    }

    int iters = 0;

    double gflop;
    if (params.routine == "potrf")
        gflop = lapack::Gflop<scalar_t>::potrf(n);
    else if (params.routine == "potrs")
        gflop = lapack::Gflop<scalar_t>::potrs(n, nrhs);
    else
        gflop = lapack::Gflop<scalar_t>::posv(n, nrhs);

    if (! ref_only) {
        if (params.routine == "potrs") {
            // Factor matrix A.
            slate::chol_factor(A, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::potrf(A, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
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
        // potrf: Factor A = LL^H or A = U^H U.
        // potrs: Solve AX = B, after factoring A above.
        // posv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "potrf") {
            slate::chol_factor(A, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::potrf(A, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }
        else if (params.routine == "potrs") {
            slate::chol_solve_using_factor(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::potrs(A, B, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }
        else if (params.routine == "posv") {
            slate::chol_solve(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::posv(A, B, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }
        else if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value) {
                slate::posvMixed(A, B, X, iters, {
                    {slate::Option::Lookahead, lookahead},
                    {slate::Option::Target, target}
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
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        if (params.routine == "posvMixed") {
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

        if (params.routine == "potrf") {
            // Solve AX = B.
            slate::chol_solve_using_factor(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::potrs(A, B, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }

        // SLATE matrix wrappers for the reference data
        auto Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
            uplo, n, &A_ref[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        auto Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
            n, nrhs, &B_ref[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aref);

        // Norm of updated-rhs/solution matrix: || X ||_1
        real_t X_norm;
        if (params.routine == "posvMixed")
            X_norm = slate::norm(slate::Norm::One, X);
        else
            X_norm = slate::norm(slate::Norm::One, B);

        // B_ref -= Aref*B_tst
        if (params.routine == "posvMixed") {
            if (std::is_same<real_t, double>::value)
                slate::multiply(scalar_t(-1.0), Aref, X, scalar_t(1.0), Bref);

                //---------------------
                // Using traditional BLAS/LAPACK name
                // slate::hemm(slate::Side::Left, scalar_t(-1.0), Aref, X, scalar_t(1.0), Bref);
        }
        else {
            slate::multiply(scalar_t(-1.0), Aref, B, scalar_t(1.0), Bref);

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::hemm(slate::Side::Left, scalar_t(-1.0), Aref, B, scalar_t(1.0), Bref);
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

        if (check) {
            // restore B_ref
            B_ref = B_orig;
            scalapack_descinit(descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
            slate_assert(info == 0);
        }

        if (params.routine == "potrs") {
            // Factor matrix A.
            scalapack_ppotrf(uplo2str(uplo), n, &A_ref[0], ione, ione, descA_ref, &info);
            slate_assert(info == 0);
        }

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        double time = testsweeper::get_wtime();
        if (params.routine == "potrf") {
            scalapack_ppotrf(uplo2str(uplo), n, &A_ref[0], ione, ione, descA_ref, &info);
        }
        else if (params.routine == "potrs") {
            scalapack_ppotrs(uplo2str(uplo), n, nrhs, &A_ref[0], ione, ione, descA_ref, &B_ref[0], ione, ione, descB_ref, &info);
        }
        else {
            scalapack_pposv(uplo2str(uplo), n, nrhs, &A_ref[0], ione, ione, descA_ref, &B_ref[0], ione, ione, descB_ref, &info);
        }
        slate_assert(info == 0);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        params.ref_time() = time_ref;
        params.ref_gflops() = gflop / time_ref;

        slate_set_num_blas_threads(saved_num_threads);

        if (verbose > 2) {
            if (origin == slate::Origin::ScaLAPACK) {
                slate::Debug::diffLapackMatrices<scalar_t>(n, n, &A_tst[0], lldA, &A_ref[0], lldA, nb, nb);
                if (params.routine != "potrf") {
                    slate::Debug::diffLapackMatrices<scalar_t>(n, nrhs, &B_tst[0], lldB, &B_ref[0], lldB, nb, nb);
                }
            }
        }
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_posv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_posv_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_posv_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_posv_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_posv_work<std::complex<double>> (params, run);
            break;
    }
}
