#include "slate.hh"
#include "slate_BandMatrix.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"
#include "band_utils.hh"

#include "slate_mpi.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads(int nt);
inline int slate_set_num_blas_threads(const int nt) { return MKL_Set_Num_Threads(nt); }
#else
inline int slate_set_num_blas_threads(const int nt) { return -1; }
#endif

//------------------------------------------------------------------------------
template <typename scalar_t> void test_gbtrs_work(Params& params, bool run)
{
    using blas::max;
    using blas::real;
    using real_t = blas::real_type<scalar_t>;
    using lld = long long;

    // get & mark input values
    blas::Op trans = params.trans.value();
    int64_t m = params.dim.n();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t kl = params.kl.value();
    int64_t ku = params.ku.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t nb = params.nb.value();
    int64_t lookahead = params.lookahead.value();
    int64_t panel_threads = params.panel_threads.value();
    lapack::Norm norm = params.norm.value();
    bool ref_only = params.ref.value() == 'o';
    bool ref = params.ref.value() == 'y' || ref_only;
    bool check = params.check.value() == 'y' && ! ref_only;
    bool trace = params.trace.value() == 'y';
    int verbose = params.verbose.value();
    int matrix = params.matrix.value();
    slate::Target target = char2target(params.target.value());

    // mark non-standard output values
    params.time.value();
    //params.gflops.value();
    //params.ref_time.value();
    //params.ref_gflops.value();

    if (!run)
        return;

    int64_t Am = m;
    int64_t An = n;
    int64_t Bm = n;
    int64_t Bn = nrhs;

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
    assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

    auto A = slate::BandMatrix<scalar_t>(
                 n, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    auto Aorig = slate::BandMatrix<scalar_t>(
                     n, n, kl, ku, nb, p, q, MPI_COMM_WORLD);

    std::vector<int64_t> iseeds = { myrow, mycol, 2, 3 };
    std::vector<int64_t> iseeds_save = iseeds;
    initializeRandom(2, iseeds.data(), A);
    // generate copy of A for checks
    iseeds = iseeds_save;
    initializeRandom(2, iseeds.data(), Aorig);

    if (matrix == 1) {
        // Make A and Aorig diagonally dominant to avoid pivoting.
        printf("diag dominant\n");
        for (int k = 0; k < std::min(A.mt(), A.nt()); ++k) {
            auto T = A(k, k);
            auto T2 = Aorig(k, k);
            for (int i = 0; i < T.nb(); ++i) {
                T.at(i, i) += n;
                T2.at(i, i) += n;
            }
        }
    }

    if (verbose > 1) {
        printf("%% rank %d A kl %lld, ku %lld\n",
               A.mpiRank(), (lld) A.lowerBandwidth(), (lld) A.upperBandwidth());
        print_matrix("A", A);
        print_matrix("Aorig", Aorig);
        print_matrix("B", B);
    }

    // if check is required, copy test data and create a descriptor for it
    slate::Matrix<scalar_t> Bref;
    std::vector<scalar_t> B_ref;
    if (check || ref) {
        B_ref = B_tst;
        scalapack_descinit(descB_ref, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
        assert(info == 0);

        Bref = slate::Matrix<scalar_t>::fromScaLAPACK(
                   Bm, Bn, &B_ref[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
    }
    double gflop = lapack::Gflop<scalar_t>::getrs(n, nrhs);

    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        // run test
        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }

        slate::Pivots pivots;

        // factor matrix A
        slate::gbtrf(A, pivots, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target},
            {slate::Option::MaxPanelThreads, panel_threads}
        });

        auto opA = A;
        if (trans == blas::Op::Trans)
            opA = transpose(A);
        else if (trans == blas::Op::ConjTrans)
            opA = conj_transpose(A);

        double time = libtest::get_wtime();

        //============================================================
        // Run SLATE test.
        // Solve op(A) X = B, after factoring A above.
        //============================================================
        slate::gbtrs(opA, pivots, B, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });

        MPI_Barrier(MPI_COMM_WORLD);
        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = libtest::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time.value() = time_tst;
        //params.gflops.value() = gflop / time_tst;

        if (verbose > 1) {
            printf("%% rank %d A2 kl %lld, ku %lld\n",
                   A.mpiRank(), (lld) A.lowerBandwidth(), (lld) A.upperBandwidth());
            print_matrix("A2", A);
            print_matrix("B2", B);

            printf("ipiv = [\n");
            for (size_t i = 0; i < pivots.size(); ++i) {
                auto p = pivots[i];
                for (size_t j = 0; j < p.size(); ++j) {
                    printf("%3lld  %% %lld, %lld\n",
                           (lld)((i + p[j].tileIndex())*nb + p[j].elementOffset() + 1),
                           (lld) p[j].tileIndex(), (lld) p[j].elementOffset());
                }
                if (i < pivots.size() - 1)
                    printf("\n");
            }
            printf("];\n");
        }
    }

    if (check) {
        // check residual for accuracy

        //================================================================
        // Test results by checking the residual
        //
        //                      || B - AX ||_I
        //                --------------------------- < epsilon
        //                 || A ||_I * || X ||_I * N
        //
        //================================================================

        // LAPACK (dget02) does
        // max_j || A * x_j - b_j ||_1 / (|| A ||_1 * || x_j ||_1).
        // No N?

        // allocate work space
        std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

        // Norm of the orig matrix: || A ||_I
        real_t A_norm = slate::norm(norm, Aorig);
        // norm of updated rhs matrix: || X ||_I
        real_t X_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_tst[0], ione, ione, descB_tst, &worklangeB[0]);

        // B_ref -= op(Aorig)*B
        auto opAorig = Aorig;
        if (trans == blas::Op::Trans)
            opAorig = transpose(Aorig);
        else if (trans == blas::Op::ConjTrans)
            opAorig = conj_transpose(Aorig);
        slate::gbmm(scalar_t(-1.0), opAorig, B, scalar_t(1.0), Bref);

        // || B - AX ||_I
        real_t R_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_ref[0], ione, ione, descB_ref, &worklangeB[0]);
        double residual = R_norm / (n*A_norm*X_norm);
        params.error.value() = residual;

        if (verbose > 0) {
            printf("Anorm = %.4e; Xnorm = %.4e; Rnorm = %.4e; error = %.4e;\n",
                   A_norm, X_norm, R_norm, residual);
        }
        if (verbose > 1) {
            print_matrix("Residual", Bm, Bn, &B_ref[0], lldB, p, q, MPI_COMM_WORLD);
        }

        real_t tol = params.tol.value() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay.value() = (params.error.value() <= tol);
    }

    // todo: reference solution requires setting up band matrix in ScaLAPACK's
    // band storage format.

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_gbtrs(Params& params, bool run)
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbtrs_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_gbtrs_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_gbtrs_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_gbtrs_work<std::complex<double>> (params, run);
            break;
    }
}
