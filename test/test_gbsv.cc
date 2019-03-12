#include "slate/slate.hh"
#include "slate/BandMatrix.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t> void test_gbsv_work(Params& params, bool run)
{
    using blas::max;
    using blas::real;
    using real_t = blas::real_type<scalar_t>;
    using lld = long long;

    // get & mark input values
    slate::Op trans = slate::Op::NoTrans;
    if (params.routine == "gbtrs")
        trans = params.trans();

    int64_t m;
    if (params.routine == "gbtrf")
        m = params.dim.m();
    else
        m = params.dim.n();  // square, n-by-n

    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
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
    slate::Target target = char2target(params.target());  // TODO: enum

    // mark non-standard output values
    params.time();
    //params.gflops();
    //params.ref_time();
    //params.ref_gflops();

    if (! run)
        return;

    // Local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descB_tst[9], descB_ref[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
    assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], n, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, nrhs, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    auto A     = slate::BandMatrix<scalar_t>(m, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    auto Aorig = slate::BandMatrix<scalar_t>(m, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    slate::Pivots pivots;

    int64_t klt = slate::ceildiv(kl, nb);
    int64_t kut = slate::ceildiv(ku, nb);
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (i >= j - kut && i <= j + klt) {
                A.tileInsert(i, j);
                Aorig.tileInsert(i, j);
                auto T = A(i, j);
                lapack::larnv(2, iseeds, T.size(), T.data());
                for (int64_t tj = 0; tj < T.nb(); ++tj) {
                    for (int64_t ti = 0; ti < T.mb(); ++ti) {
                        int64_t j_i = (jj + tj) - (ii + ti);
                        if (-kl > j_i || j_i > ku) {
                            T.at(ti, tj) = 0;
                        }
                    }
                }
                auto T2 = Aorig(i, j);
                lapack::lacpy(lapack::MatrixType::General, T.mb(), T.nb(),
                              T.data(), T.stride(),
                              T2.data(), T2.stride());
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    if (verbose > 1) {
        printf("%% rank %d A kl %lld, ku %lld\n",
               A.mpiRank(), (lld) A.lowerBandwidth(), (lld) A.upperBandwidth());
        print_matrix("A", A);
    }

    // if check is required, copy test data and create a descriptor for it
    slate::Matrix<scalar_t> Bref;
    std::vector<scalar_t> B_ref;
    if (check || ref) {
        B_ref = B_tst;
        scalapack_descinit(descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
        assert(info == 0);

        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   n, nrhs, &B_ref[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    // todo: gflops formula for band.
    //double gflop;
    //if (params.routine == "getrf")
    //    gflop = lapack::Gflop<scalar_t>::getrf(m, n);
    //else if (params.routine == "getrs")
    //    gflop = lapack::Gflop<scalar_t>::getrs(n, nrhs);
    //else
    //    gflop = lapack::Gflop<scalar_t>::gesv(n, nrhs);

    if (! ref_only) {
        if (params.routine == "gbtrs") {
            // Factor matrix A.
            slate::gbtrf(A, pivots, {
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
        // gbtrf: Factor PA = LU.
        // gbtrs: Solve AX = B, after factoring A above.
        // gbsv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "gbtrf") {
            slate::gbtrf(A, pivots, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target},
                {slate::Option::MaxPanelThreads, panel_threads},
                {slate::Option::InnerBlocking, ib}
            });
        }
        else if (params.routine == "gbtrs") {
            auto opA = A;
            if (trans == slate::Op::Trans)
                opA = transpose(A);
            else if (trans == slate::Op::ConjTrans)
                opA = conj_transpose(A);

            slate::gbtrs(opA, pivots, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }
        else {
            slate::gbsv(A, pivots, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target},
                {slate::Option::MaxPanelThreads, panel_threads},
                {slate::Option::InnerBlocking, ib}
            });
        }

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = libtest::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
        ///params.gflops() = gflop / time_tst;

        if (verbose > 1) {
            printf("%% rank %d A2 kl %lld, ku %lld\n",
                   A.mpiRank(), (lld) A.lowerBandwidth(), (lld) A.upperBandwidth());
            print_matrix("A2", A);
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

        if (params.routine == "gbtrf") {
            // Solve AX = B.
            slate::gbtrs(A, pivots, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }

        // allocate work space
        std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, Aorig);
        // Norm of updated rhs matrix: || X ||_1
        real_t X_norm = scalapack_plange("1", n, nrhs, &B_tst[0], ione, ione, descB_tst, &worklangeB[0]);

        // B_ref -= op(Aref)*B_tst
        auto opAorig = Aorig;
        if (trans == slate::Op::Trans)
            opAorig = transpose(Aorig);
        else if (trans == slate::Op::ConjTrans)
            opAorig = conj_transpose(Aorig);
        slate::gbmm(scalar_t(-1.0), opAorig, B, scalar_t(1.0), Bref);

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

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_gbsv(Params& params, bool run)
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbsv_work<float>(params, run);
            break;

        case libtest::DataType::Double:
            test_gbsv_work<double>(params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_gbsv_work<std::complex<float>>(params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_gbsv_work<std::complex<double>>(params, run);
            break;
    }
}
