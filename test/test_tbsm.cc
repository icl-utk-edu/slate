#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"
#include "band_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#undef PIN_MATRICES

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_tbsm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::Norm;
    //using llong = long long;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.transA();
    // ref. code to check can't do transB; disable for now.
    //slate::Op transB = params.transB();
    slate::Diag diag = params.diag();
    scalar_t alpha = params.alpha();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t kd = params.kd();
    int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t lookahead = params.lookahead();
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    //params.gflops();
    params.ref_time();
    //params.ref_gflops();

    if (! run) {
        // Note is printed before table header.
        printf("%% Note this does NOT test pivoting in tbsm. See gbtrs for that.\n");
        return;
    }

    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported\n");
        return;
    }

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so trans(B) is m-by-n
    int64_t An  = (side == slate::Side::Left ? m : n);
    int64_t Am  = An;
    int64_t Bm  = m;  //(transB == slate::Op::NoTrans ? m : n);
    int64_t Bn  = n;  //(transB == slate::Op::NoTrans ? n : m);

    // local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descB_tst[9], descB_ref[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);
    zeroOutsideBand(&A_tst[0], Am, An, kd, kd, nb, nb, myrow, mycol, nprow, npcol, mlocA);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // if check is required, copy test data and create a descriptor for it
    std::vector< scalar_t > B_ref;
    if (check || ref) {
        scalapack_descinit(descB_ref, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
        slate_assert(info == 0);
        B_ref = B_tst;
    }

    // create SLATE matrices from the ScaLAPACK layouts
    auto Aband = BandFromScaLAPACK(
                     Am, An, kd, kd, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    auto A = slate::TriangularBandMatrix<scalar_t>(uplo, diag, Aband);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK
             (Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
    slate::Pivots pivots;

    // Make A diagonally dominant to be reasonably well conditioned.
    // tbsm seems to pass with unit diagonal, even without diagonal dominance.
    for (int i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, i)) {
            auto T = A(i, i);
            for (int ii = 0; ii < T.mb(); ++ii) {
                T.at(ii, ii) += Am;
            }
        }
    }

    if (transA == slate::Op::Trans)
        A = transpose(A);
    else if (transA == slate::Op::ConjTrans)
        A = conjTranspose(A);

    //if (transB == slate::Op::Trans)
    //    B = transpose(B);
    //else if (transB == slate::Op::ConjTrans)
    //    B = conjTranspose(B);

    if (verbose > 1) {
        // todo: print_matrix( A ) calls Matrix version;
        // need TriangularBandMatrix version.
        printf("alpha = %10.6f + %10.6fi;\n", real(alpha), imag(alpha));
        print_matrix("A_tst", mlocA, nlocA, &A_tst[0], lldA, p, q, MPI_COMM_WORLD);
        print_matrix("B_tst", mlocB, nlocB, &B_tst[0], lldB, p, q, MPI_COMM_WORLD);
        print_matrix("A", Aband);
        print_matrix("B", B);
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
    // Solve AX = alpha B (left) or XA = alpha B (right).
    //==================================================
    if (side == slate::Side::Left)
        slate::triangular_solve(alpha, A, B, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });
    else if (side == slate::Side::Right)
        slate::triangular_solve(alpha, B, A, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });
    else
        throw slate::Exception("unknown side");

    //---------------------
    // Using traditional BLAS/LAPACK name
    // slate::tbsm(side, alpha, A, pivots, B, {
    //     {slate::Option::Lookahead, lookahead},
    //     {slate::Option::Target, target}
    // });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    //double gflop = blas::Gflop<scalar_t>::gemm(m, n, k);
    params.time() = time_tst;
    //params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        print_matrix("B2", B);
        print_matrix("B2_tst", mlocB, nlocB, &B_tst[0], lldB, p, q, MPI_COMM_WORLD);
    }

    if (check || ref) {
        //printf("%% check & ref\n");
        // comparison with reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        std::vector<real_t> worklantr(std::max(mlocA, nlocA));
        std::vector<real_t> worklange(std::max(mlocB, nlocB));

        // get norms of the original data
        real_t A_norm = scalapack_plantr(norm2str(norm), uplo2str(uplo), diag2str(diag), Am, An, &A_tst[0], ione, ione, descA_tst, &worklantr[0]);
        real_t B_orig_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_tst[0], ione, ione, descB_tst, &worklange[0]);

        if (verbose > 1) {
            print_matrix("B_ref", mlocB, nlocB, &B_ref[0], lldB, p, q, MPI_COMM_WORLD);
        }

        //==================================================
        // Run ScaLAPACK reference routine.
        // Note this is on a FULL matrix, so ignore reference performance!
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();
        scalapack_ptrsm(side2str(side), uplo2str(uplo), op2str(transA), diag2str(diag),
                        m, n, alpha,
                        &A_tst[0], ione, ione, descA_tst,
                        &B_ref[0], ione, ione, descB_ref);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        if (verbose > 1) {
            print_matrix("B2_ref", mlocB, nlocB, &B_ref[0], lldB, p, q, MPI_COMM_WORLD);
        }
        // local operation: error = B_ref - B_tst
        blas::axpy(B_ref.size(), -1.0, &B_tst[0], 1, &B_ref[0], 1);

        // norm(B_ref - B_tst)
        real_t B_diff_norm = scalapack_plange(norm2str(norm), Bm, Bn, &B_ref[0], ione, ione, descB_ref, &worklange[0]);

        if (verbose > 1) {
            print_matrix("B_diff", mlocB, nlocB, &B_ref[0], lldB, p, q, MPI_COMM_WORLD);
        }
        real_t error = B_diff_norm
                     / (sqrt(real_t(Am) + 2) * std::abs(alpha) * A_norm * B_orig_norm);

        params.ref_time() = time_ref;
        //params.ref_gflops() = gflop / time_ref;
        params.error() = error;

        slate_set_num_blas_threads(saved_num_threads);

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }
    //printf("%% done\n");

    #ifdef PIN_MATRICES
    cuerror = cudaHostUnregister(&A_tst[0]);
    cuerror = cudaHostUnregister(&B_tst[0]);
    #endif

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_tbsm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tbsm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_tbsm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_tbsm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tbsm_work<std::complex<double>> (params, run);
            break;
    }
}
