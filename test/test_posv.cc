#include "slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"

#include "slate/slate_mpi.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t> void test_posv_work(Params& params, bool run)
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
    int verbose = params.verbose();
    slate::Target target = char2target(params.target());

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

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplghe(&A_tst[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
    assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], n, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(n, nrhs, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    std::vector<scalar_t> B_ref;
    std::vector<scalar_t> B_orig;
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        assert(info == 0);

        B_ref = B_tst;
        scalapack_descinit(descB_ref, n, nrhs, nb, nb, izero, izero, ictxt, mlocB, &info);
        assert(info == 0);

        if (check && ref)
            B_orig = B_tst;
    }

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
            slate::potrf(A, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
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
        // potrf: Factor A = LL^H or A = U^H U.
        // potrs: Solve AX = B, after factoring A above.
        // posv:  Solve AX = B, including factoring A.
        //==================================================
        if (params.routine == "potrf") {
            slate::potrf(A, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }
        else if (params.routine == "potrs") {
            slate::potrs(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }
        else {
            slate::posv(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
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
            slate::potrs(A, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
        }

        // allocate work space
        size_t ldw = nb*ceil(ceil(mlocA / (double) nb) / (scalapack_ilcm(&nprow, &npcol) / nprow));
        std::vector<real_t> worklansyA(2*nlocA + mlocA + ldw);
        std::vector<real_t> worklangeB(std::max(mlocB, nlocB));

        // Norm of original matrix: || A ||_1
        real_t A_norm = scalapack_plansy("1", uplo2str(uplo), n, &A_ref[0], ione, ione, descA_ref, &worklansyA[0]);
        // Norm of updated rhs matrix: || X ||_1
        real_t X_norm = scalapack_plange("1", n, nrhs, &B_tst[0], ione, ione, descB_tst, &worklangeB[0]);

        // B_ref -= Aref*B_tst
        scalapack_psymm("left", uplo2str(uplo),
                        n, nrhs,
                        scalar_t(-1.0),
                        &A_ref[0], ione, ione, descA_ref,
                        &B_tst[0], ione, ione, descB_tst,
                        scalar_t(1.0),
                        &B_ref[0], ione, ione, descB_ref);

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
            assert(info == 0);
        }

        if (params.routine == "potrs") {
            // Factor matrix A.
            scalapack_ppotrf(uplo2str(uplo), n, &A_ref[0], ione, ione, descA_ref, &info);
            assert(info == 0);
        }

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        double time = libtest::get_wtime();
        if (params.routine == "potrf") {
            scalapack_ppotrf(uplo2str(uplo), n, &A_ref[0], ione, ione, descA_ref, &info);
        }
        else if (params.routine == "potrs") {
            scalapack_ppotrs(uplo2str(uplo), n, nrhs, &A_ref[0], ione, ione, descA_ref, &B_ref[0], ione, ione, descB_ref, &info);
        }
        else {
            scalapack_pposv(uplo2str(uplo), n, nrhs, &A_ref[0], ione, ione, descA_ref, &B_ref[0], ione, ione, descB_ref, &info);
        }
        assert(info == 0);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

        params.ref_time() = time_ref;
        params.ref_gflops() = gflop / time_ref;

        slate_set_num_blas_threads(saved_num_threads);
    }

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_posv(Params& params, bool run)
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_posv_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_posv_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_posv_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_posv_work<std::complex<double>> (params, run);
            break;
    }
}
