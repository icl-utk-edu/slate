#include "slate.hh"
#include "slate_BandMatrix.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"

#include "slate_mpi.hh"

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
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Target target = char2target(params.target());  // TODO: enum

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

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
    int64_t mlocB = scalapack_numroc(n,    nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
    assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], n, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 1);

    /// // allocate ipiv locally
    /// size_t ipiv_size = (size_t) (lldA + nb);
    /// std::vector<int> ipiv_tst(ipiv_size);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 n, nrhs, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    auto A = slate::BandMatrix<scalar_t>(
                 n, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    auto A_ref = slate::BandMatrix<scalar_t>(
                     n, n, kl, ku, nb, p, q, MPI_COMM_WORLD);
    slate::Pivots pivots;

    int64_t klt = slate::ceildiv(kl, nb);
    int64_t kut = slate::ceildiv(ku, nb);
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (i >= j - kut && i <= j + klt) {
                A.tileInsert(i, j);
                A_ref.tileInsert(i, j);
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
                auto T2 = A_ref(i, j);
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

    // TODO: keep copy of A and B for residual check B - AX. Or regenerate A.
    /// // if check is required, copy test data and create a descriptor for it
    /// std::vector<scalar_t> A_ref;
    /// std::vector<int> ipiv_ref;
    /// if (check || ref) {
    ///     A_ref = A_tst;
    ///     scalapack_descinit(descA_ref, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    ///     assert(info == 0);
    ///     ipiv_ref.resize(ipiv_tst.size());
    /// }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = libtest::get_wtime();

    //==================================================
    // Run SLATE test.
    // Solve AX = B, including factoring A.
    //==================================================
    slate::gbsv(A, pivots, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = libtest::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    ///double gflop = lapack::Gflop<scalar_t>::gbsv(m, n);
    params.time() = time_tst;
    ///params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        printf("%% rank %d A2 kl %lld, ku %lld\n",
               A.mpiRank(), (lld) A.lowerBandwidth(), (lld) A.upperBandwidth());
        print_matrix("A2", A);
    }

    /// if (check || ref) {
    ///     // A comparison with a reference routine from ScaLAPACK
    ///
    ///     // set MKL num threads appropriately for parallel BLAS
    ///     int omp_num_threads;
    ///     #pragma omp parallel
    ///     { omp_num_threads = omp_get_num_threads(); }
    ///     int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);
    ///     int64_t info_ref=0;
    ///
    ///     // allocate work space
    ///     std::vector<real_t> worklange(std::max(mlocA, nlocA));
    ///
    ///     //==================================================
    ///     // Run ScaLAPACK reference routine.
    ///     //==================================================
    ///     MPI_Barrier(MPI_COMM_WORLD);
    ///     time = libtest::get_wtime();
    ///     scalapack_pgbsv(n, nrhs, &A_ref[0], ione, ione, descA_ref, &ipiv_ref[0], ... B ..., &info_ref);
    ///     assert(0 == info_ref);
    ///     MPI_Barrier(MPI_COMM_WORLD);
    ///     double time_ref = libtest::get_wtime() - time;
    ///
    ///     // todo: The IPIV needs to be checked
    ///
    ///     // Norm of the reference result
    ///     real_t A_ref_norm = scalapack_plange(norm2str(norm), n, n, &A_ref[0], ione, ione, descA_ref, &worklange[0]);
    ///
    ///     // local operation: error = A_ref = A_ref - A_tst;   ipiv_ref = ipiv_ref - ipiv_tst
    ///     blas::axpy(A_ref.size(), -1.0, &A_tst[0], 1, &A_ref[0], 1);
    ///
    ///     // error = norm(error)
    ///     real_t error_norm = scalapack_plange(norm2str(norm), n, n, &A_ref[0], ione, ione, descA_ref, &worklange[0]);
    ///
    ///     // error = error / reference;
    ///     if (A_ref_norm != 0)
    ///         error_norm /= A_ref_norm;
    ///
    ///     params.ref_time() = time_ref;
    ///     params.ref_gflops() = gflop / time_ref;
    ///     params.error() = error_norm;
    ///
    ///     slate_set_num_blas_threads(saved_num_threads);
    ///
    ///     real_t eps = std::numeric_limits<real_t>::epsilon();
    ///     params.okay() = (params.error() <= 3*eps);
    /// }

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
