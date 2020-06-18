#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#undef PIN_MATRICES

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_gemm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Norm;

    // get & mark input values
    slate::Op transA = params.transA();
    slate::Op transB = params.transB();
    scalar_t alpha = params.alpha();
    scalar_t beta = params.beta();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    slate::Norm norm = params.norm();
    bool check = params.check() == 'y' && ! ref_only;
    bool ref = params.ref() == 'y' || ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    std::string gemm_variant = params.gemm_variant();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // skip invalid or unimplemented options
    if (gemm_variant=="gemmA" && target!=slate::Target::HostTask) {
        printf("skipping: currently gemmA is only implemented for HostTask\n");
        return;
    }

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // sizes of A and B
    int64_t Am = (transA == slate::Op::NoTrans ? m : k);
    int64_t An = (transA == slate::Op::NoTrans ? k : m);
    int64_t Bm = (transB == slate::Op::NoTrans ? k : n);
    int64_t Bn = (transB == slate::Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;

    // local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descB_tst[9], descC_tst[9], descC_ref[9];
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

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // matrix C, figure out local size, allocate, create descriptor, initialize
    int64_t mlocC = scalapack_numroc(m, nb, myrow, izero, nprow);
    int64_t nlocC = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descC_tst, Cm, Cn, nb, nb, izero, izero, ictxt, mlocC, &info);
    slate_assert(info == 0);
    int64_t lldC = (int64_t)descC_tst[8];
    std::vector<scalar_t> C_tst(lldC*nlocC);
    scalapack_pplrnt(&C_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocC, iseed + 3);

    #ifdef PIN_MATRICES
    int cuerror;
    cuerror = cudaHostRegister(&A_tst[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    cuerror = cudaHostRegister(&B_tst[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    cuerror = cudaHostRegister(&C_tst[0], (size_t)size_A*sizeof(scalar_t), cudaHostRegisterDefault);
    #endif

    // if reference run is required, copy test data and create a descriptor for it
    std::vector<scalar_t> C_ref;
    slate::Matrix<scalar_t> C_ref_slate;
    if (check || ref) {
        C_ref = C_tst;
        scalapack_descinit(descC_ref, Cm, Cn, nb, nb, izero, izero, ictxt, mlocC, &info);
        slate_assert(info == 0);
        C_ref_slate = slate::Matrix<scalar_t>::fromScaLAPACK( m,  n, &C_ref[0], lldC, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    slate::Matrix<scalar_t> A, B, C;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(Am, An, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        B = slate::Matrix<scalar_t>(Bm, Bn, nb, nprow, npcol, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);
        copy(&B_tst[0], descB_tst, B);

        C = slate::Matrix<scalar_t>(Cm, Cn, nb, nprow, npcol, MPI_COMM_WORLD);
        C.insertLocalTiles(origin_target);
        copy(&C_tst[0], descC_tst, C);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(Am, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK(Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
        C = slate::Matrix<scalar_t>::fromScaLAPACK( m,  n, &C_tst[0], lldC, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    if (transA == slate::Op::Trans)
        A = transpose(A);
    else if (transA == slate::Op::ConjTrans)
        A = conjTranspose(A);

    if (transB == slate::Op::Trans)
        B = transpose(B);
    else if (transB == slate::Op::ConjTrans)
        B = conjTranspose(B);

    slate_assert(A.mt() == C.mt());
    slate_assert(B.nt() == C.nt());
    slate_assert(A.nt() == B.mt());

    // if reference run is required, record norms to be used in the check/ref
    real_t A_norm=0, B_norm=0, C_orig_norm=0;
    if (check || ref) {
        A_norm = slate::norm(norm, A);
        B_norm = slate::norm(norm,B);
        C_orig_norm = slate::norm(norm, C_ref_slate);
    }

    if (verbose >= 2) {
        print_matrix("A", A);
        print_matrix("B", B);
        print_matrix("C", C);
    }

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::gemm(m, n, k);

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time = testsweeper::get_wtime();

        //==================================================
        // Run SLATE test.
        // C = alpha A B + beta C.
        //==================================================
        if (gemm_variant == "gemmC")
            slate::multiply(
                alpha, A, B, beta, C, {
                    {slate::Option::Lookahead, lookahead},
                    {slate::Option::Target, target}
                });

            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::gemm(
            //     alpha, A, B, beta, C, {
            //         {slate::Option::Lookahead, lookahead},
            //         {slate::Option::Target, target}
            //     });
        else if (gemm_variant == "gemmA")
            slate::gemmA(
                alpha, A, B, beta, C, {
                    {slate::Option::Lookahead, lookahead},
                    {slate::Option::Target, target}
                });
        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }

        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        if (verbose >= 2) {
            C.tileGetAllForReading(C.hostNum(), slate::LayoutConvert::None);
            print_matrix("C2", C);
        }

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;
    }

    if (check || ref) {
        // comparison with reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        if (verbose >= 2)
            print_matrix("Cref", mlocC, nlocC, &C_ref[0], lldC, p, q, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double time = testsweeper::get_wtime();

        scalapack_pgemm(op2str(transA), op2str(transB), m, n, k, alpha,
                        &A_tst[0], ione, ione, descA_tst,
                        &B_tst[0], ione, ione, descB_tst, beta,
                        &C_ref[0], ione, ione, descC_ref);

        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        if (verbose >= 2)
            print_matrix("Cref2", mlocC, nlocC, &C_ref[0], lldC, p, q, MPI_COMM_WORLD);

        // Copy SLATE result back from GPU or CPU tiles.
        if (origin != slate::Origin::ScaLAPACK)
            copy(C, &C_tst[0], descC_tst);

        // get differences C_tst = C_tst - C_ref
        scalar_t one=1;
        slate::geadd(-one, C_ref_slate, one, C);

        if (verbose >= 2)
            print_matrix("Diff", C);

        // norm(C_tst - C_ref)
        real_t C_diff_norm = slate::norm(norm, C);

        real_t error = C_diff_norm
                     / (sqrt(real_t(k) + 2) * std::abs(alpha) * A_norm * B_norm
                        + 2 * std::abs(beta) * C_orig_norm);

        params.ref_time() = time_ref;
        params.ref_gflops() = gflop / time_ref;
        params.error() = error;

        slate_set_num_blas_threads(saved_num_threads);

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }

    #ifdef PIN_MATRICES
    cuerror = cudaHostUnregister(&A_tst[0]);
    cuerror = cudaHostUnregister(&B_tst[0]);
    cuerror = cudaHostUnregister(&C_tst[0]);
    #endif

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_gemm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gemm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gemm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemm_work<std::complex<double>> (params, run);
            break;
    }
}
