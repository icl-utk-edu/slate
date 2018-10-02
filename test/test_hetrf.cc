#include "slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"

#include "slate_mpi.hh"
#include "../test.hh"

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
template <typename scalar_t> void test_hetrf_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    //---------------------
    // get & mark input values
    slate::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t nb = params.nb.value();
    int64_t lookahead = params.lookahead.value();
    int64_t panel_threads = params.panel_threads.value();
    lapack::Norm norm = params.norm.value();
    bool check = params.check.value() == 'y';
    bool trace = params.trace.value() == 'y';
    slate::Target target = char2target(params.target.value());

    //---------------------
    // mark non-standard output values
    params.time.value();
    params.gflops.value();

    if (! run)
        return;

    int64_t Am = n;
    int64_t An = n;

    //---------------------
    // Local values
    const int izero = 0, ione = 1;

    //---------------------
    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    //---------------------
    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    //---------------------
    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplghe(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    //---------------------
    // Create SLATE matrix from the ScaLAPACK layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    slate::Pivots pivots;

    //---------------------
    // tridiagonal matrices
    int64_t kl = nb;
    int64_t ku = nb;
    slate::Pivots pivots2;
    auto T = slate::BandMatrix<scalar_t>(n, n, kl, ku, nb, p, q, MPI_COMM_WORLD);

    //---------------------
    // auxiliary matrices
    auto H = slate::Matrix<scalar_t> (n, n, nb, p, q, MPI_COMM_WORLD);

    //---------------------
    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    if (check) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    //---------------------
    // run test
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = libtest::get_wtime();
    slate::hetrf(A, pivots, T, pivots2, H, {
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads}
    });

    MPI_Barrier(MPI_COMM_WORLD);
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = libtest::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    //---------------------
    // compute and save timing/performance
    double gflop = lapack::Gflop<scalar_t>::potrf(n);
    params.time.value() = time_tst;
    params.gflops.value() = gflop / time_tst;

    if (check) {
        int64_t Bm = n;
        int64_t Bn = n;
        int descB_tst[9], descB_ref[9];
        std::vector<scalar_t> B_ref;

        // matrix B, figure out local size, allocate, create descriptor, initialize
        int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
        int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
        scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
        assert(info == 0);
        int64_t lldB = (int64_t)descB_tst[8];
        std::vector<scalar_t> B_tst(lldB*nlocB);
        scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

        B_ref.resize(B_tst.size());
        B_ref = B_tst;
        scalapack_descinit(descB_ref, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
        assert(info == 0);

        auto B = slate::Matrix<scalar_t>::fromScaLAPACK(Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

        // solve
        slate::hetrs(A, pivots, T, pivots2, B, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });

        // allocate work space
        std::vector<real_t> worklangeA(std::max({mlocA, nlocA}));
        std::vector<real_t> worklangeB(std::max({mlocB, nlocB}));

        // Norm of the orig matrix: || A ||_I
        real_t A_norm = scalapack_plange(norm2str(norm), Am, An, &A_ref[0], ione, ione, descA_ref, &worklangeA[0]);
        // norm of updated rhs matrix: || X ||_I
        real_t X_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_tst[0], ione, ione, descB_tst, &worklangeB[0]);

        // B_ref -= Aref*B_tst
        scalapack_phemm("Left", "Lower",
                        Bm, Bn,
                        scalar_t(-1.0),
                        &A_ref[0], ione, ione, descA_ref,
                        &B_tst[0], ione, ione, descB_tst,
                        scalar_t(1.0),
                        &B_ref[0], ione, ione, descB_ref);

        // || B - AX ||_I
        real_t R_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_ref[0], ione, ione, descB_ref, &worklangeB[0]);

        double residual = R_norm / (n*A_norm*X_norm);
        params.error.value() = residual;

        real_t tol = params.tol.value() * 0.5 * std::numeric_limits<real_t>::epsilon();
        params.okay.value() = (params.error.value() <= tol);
    }

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_hetrf(Params& params, bool run)
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hetrf_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_hetrf_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_hetrf_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_hetrf_work<std::complex<double>> (params, run);
            break;
    }
}
