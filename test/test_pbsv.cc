#include "slate/slate.hh"
#include "slate/HermitianBandMatrix.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_pbsv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using llong = long long;

    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t kd = params.kd();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
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
    //params.ref_time();
    //params.ref_gflops();

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported\n");
        return;
    }

    // Local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descB_tst[9], descB_ref[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    scalapack_descinit(
      descB_tst, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(
      &B_tst[0], n, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, nrhs, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    auto A = slate::HermitianBandMatrix<scalar_t>(
              uplo, n, kd, nb, p, q, MPI_COMM_WORLD);
    auto Aorig = slate::HermitianBandMatrix<scalar_t>(
                  uplo, n, kd, nb, p, q, MPI_COMM_WORLD);

    int64_t kdt = slate::ceildiv(kd, nb);
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t jb = A.tileNb(j);
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            int64_t ib = A.tileMb(i);
            if (A.tileIsLocal(i, j)) {
                if ((A.uplo() == slate::Uplo::Lower && i <= j + kdt && j <= i) ||
                    (A.uplo() == slate::Uplo::Upper && i >= j - kdt && j >= i)) {
                    A.tileInsert(i, j);
                    Aorig.tileInsert(i, j);
                    auto T = A(i, j);
                    lapack::larnv(2, iseeds, T.size(), T.data());
                    for (int64_t tj = jj; tj < jj + T.nb(); ++tj) {
                        for (int64_t ti = ii; ti < ii + T.mb(); ++ti) {
                            if ((A.uplo() == slate::Uplo::Lower && -kd     > tj - ti) ||
                                (A.uplo() == slate::Uplo::Upper && tj - ti > kd     )) {
                                T.at(ti - ii, tj - jj) = 0;
                            }
                        }
                    }
                    if (i == j) {
                        for (int64_t tj = jj; tj < jj + T.nb(); ++tj) {
                            for (int64_t ti = ii; ti < ii + T.mb(); ++ti) {
                                if ((ti - ii) == (tj - jj)) {
                                    T.at(ti - ii, tj - jj) = std::abs(T.at(ti - ii, tj - jj)) + n;
                                }
                            }
                        }
                    }
                    auto T2 = Aorig(i, j);
                    lapack::lacpy(lapack::MatrixType::General, T.mb(), T.nb(),
                                  T.data(), T.stride(),
                                  T2.data(), T2.stride());
                }
            }
            ii += ib;
        }
        jj += jb;
    }

    if (verbose > 1) {
        printf("%% rank %d A kd %lld\n", A.mpiRank(), llong( A.bandwidth()));
        print_matrix("A", A);
        print_matrix("B", B);
    }


    // if check is required, copy test data and create a descriptor for it
    slate::Matrix<scalar_t> Bref;
    std::vector<scalar_t> B_ref;
    if (check || ref) {
        B_ref = B_tst;
        scalapack_descinit(
          descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
        slate_assert(info == 0);

        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, nrhs, &B_ref[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    double gflop;
    if (params.routine == "pbtrf")
        gflop = lapack::Gflop<scalar_t>::pbtrf(n, kd);
    else if (params.routine == "potrs")
        gflop = lapack::Gflop<scalar_t>::pbtrs(n, nrhs, kd);
    else
        gflop = lapack::Gflop<scalar_t>::pbsv(n, nrhs, kd);

    if (! ref_only) {
        if (params.routine == "pbtrs") {
            // Factor matrix A.
            slate::chol_factor(A, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::pbtrf(A, {
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
        // pbtrf: Factor A = LL^U or A = U^H U.
        // pbtrs: Solve AX = B, after factoring A above.
        // pbsv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "pbtrf") {
            slate::chol_factor(A, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::pbtrf(A, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }
        else if (params.routine == "pbtrs") {
            slate::chol_solve_using_factor(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::pbtrs(A, B, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }
        else {
            slate::chol_solve(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            // slate::pbsv(A, B, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace)
            slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;

        if (verbose > 1) {
            printf("%% rank %d A2 kd %lld\n",
                  A.mpiRank(), llong( A.bandwidth( )));
            print_matrix("A2", A);
            print_matrix("B2", B);
            printf( "nb = %lld;\n", llong( nb ) );
        }
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

        // LAPACK (dget02) does
        // max_j || A * x_j - b_j ||_1 / (|| A ||_1 * || x_j ||_1).
        // No N?

        if (params.routine == "pbtrf") {
            // Solve AX = B.
            slate::chol_solve_using_factor(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::pbtrs(A, B, {
            //     {slate::Option::Lookahead, lookahead},
            //     {slate::Option::Target, target}
            // });
        }

        // allocate work space
        std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aorig);
        // Norm of updated rhs matrix: || X ||_1
        real_t X_norm = scalapack_plange("1", n, nrhs, &B_tst[0], ione, ione, descB_tst, &worklangeB[0]);

        // B_ref -= Aref*B_tst
        slate::multiply(scalar_t(-1.0), Aorig, B, scalar_t(1.0), Bref);
        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::hbmm(blas::Side::Left, scalar_t(-1.0), Aorig, B, scalar_t(1.0), Bref);

        // Norm of residual: || B - AX ||_1
        real_t R_norm = scalapack_plange("1", n, nrhs, &B_ref[0], ione, ione, descB_ref, &worklangeB[0]);
        double residual = R_norm / (n*A_norm*X_norm);
        params.error() = residual;

        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);

        if (verbose > 0) {
            printf("Anorm = %.4e; Xnorm = %.4e; Rnorm = %.4e; error = %.4e;\n",
                   A_norm, X_norm, R_norm, residual);
        }
        if (verbose > 1) {
            print_matrix("Residual", n, nrhs, &B_ref[0], lldB, p, q, MPI_COMM_WORLD);
        }
    }
    // todo: reference solution requires setting up band matrix in ScaLAPACK's
    // band storage format.

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_pbsv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_pbsv_work<float>(params, run);
            break;

        case testsweeper::DataType::Double:
            test_pbsv_work<double>(params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_pbsv_work<std::complex<float>>(params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_pbsv_work<std::complex<double>>(params, run);
            break;
    }
}
