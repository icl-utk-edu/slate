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
void test_gbmm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::max;
    using blas::min;
    using slate::Norm;
    //using llong = long long;

    // get & mark input values
    slate::Op transA = params.transA();
    slate::Op transB = params.transB();
    scalar_t alpha = params.alpha();
    scalar_t beta = params.beta();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
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
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported\n");
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
    int izero = 0, ione = 1;

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
    // int mpirank = mycol*nprow + myrow;

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, lldA, iseed + 1);
    zeroOutsideBand(&A_tst[0], Am, An, kl, ku, nb, nb, myrow, mycol, nprow, npcol, lldA);

    if (verbose > 1) {
        print_matrix("A_tst", mlocA, nlocA, &A_tst[0], lldA, p, q, MPI_COMM_WORLD);
    }

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, lldB, iseed + 2);

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

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> C_ref;
    if (check || ref) {
        C_ref = C_tst;
        scalapack_descinit(descC_ref, Cm, Cn, nb, nb, izero, izero, ictxt, mlocC, &info);
        slate_assert(info == 0);
    }

    // create SLATE matrices from the ScaLAPACK layouts
    auto A = BandFromScaLAPACK(
                 Am, An, kl, ku, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
                 Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
    auto C = slate::Matrix<scalar_t>::fromScaLAPACK(
                 m, n, &C_tst[0], lldC, nb, nprow, npcol, MPI_COMM_WORLD);

    if (verbose > 1) {
        //printf("%% rank %d A2 kl %lld, ku %lld\n",
        //       A.mpiRank(), A.lowerBandwidth(), A.upperBandwidth());
        print_matrix("A", A);
        print_matrix("B", B);
        print_matrix("C", C);
        printf("alpha = %.4f + %.4fi;\nbeta  = %.4f + %.4fi;\n",
               real(alpha), imag(alpha),
               real(beta), imag(beta));
    }

    //printf("%% trans\n");
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
    slate::multiply(alpha, A, B, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    //---------------------
    // Using traditional BLAS/LAPACK name
    // slate::gbmm(alpha, A, B, beta, C, {
    // {slate::Option::Lookahead, lookahead},
    // {slate::Option::Target, target}
    // });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    double gflop = blas::Gflop<scalar_t>::gbmm(m, n, k, kl, ku);
    params.time() = time_tst;
    params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        print_matrix("C2", C);
        print_matrix("C_tst", mlocC, nlocC, &C_tst[0], lldC, p, q, MPI_COMM_WORLD);
    }

    if (check || ref) {
        //printf("%% check & ref\n");
        // comparison with reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        // allocate work space
        std::vector<real_t> worklange(std::max({mlocC, mlocB, mlocA, nlocC, nlocB, nlocA}));

        // get norms of the original data
        real_t A_norm      = scalapack_plange(norm2str(norm), Am, An, &A_tst[0], ione, ione, descA_tst, &worklange[0]);
        real_t B_norm      = scalapack_plange(norm2str(norm), Bm, Bn, &B_tst[0], ione, ione, descB_tst, &worklange[0]);
        real_t C_orig_norm = scalapack_plange(norm2str(norm), Cm, Cn, &C_ref[0], ione, ione, descC_ref, &worklange[0]);

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();
        scalapack_pgemm(op2str(transA), op2str(transB), m, n, k, alpha,
                        &A_tst[0], ione, ione, descA_tst,
                        &B_tst[0], ione, ione, descB_tst, beta,
                        &C_ref[0], ione, ione, descC_ref);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        if (verbose > 1) {
            print_matrix("C_ref", mlocC, nlocC, &C_ref[0], lldC, p, q, MPI_COMM_WORLD);
        }

        // perform a local operation to get differences C_ref = C_ref - C_tst
        blas::axpy(C_ref.size(), -1.0, &C_tst[0], 1, &C_ref[0], 1);

        if (verbose > 1) {
            print_matrix("C_diff", mlocC, nlocC, &C_ref[0], lldC, p, q, MPI_COMM_WORLD);
        }

        // norm(C_ref - C_tst)
        real_t C_diff_norm = scalapack_plange(
                                 norm2str(norm), Cm, Cn, &C_ref[0], ione, ione, descC_ref, &worklange[0]);

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
    //printf("%% done\n");

    #ifdef PIN_MATRICES
    cuerror = cudaHostUnregister(&A_tst[0]);
    cuerror = cudaHostUnregister(&B_tst[0]);
    cuerror = cudaHostUnregister(&C_tst[0]);
    #endif

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_gbmm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gbmm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gbmm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbmm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbmm_work<std::complex<double>> (params, run);
            break;
    }
}
