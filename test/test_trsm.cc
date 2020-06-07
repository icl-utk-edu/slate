#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_trsm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using slate::Op;
    using slate::Norm;
    using blas::real;
    // using llong = long long;

    // get & mark input values
    slate::Side side = params.side();
    slate::Uplo uplo = params.uplo();
    slate::Op transA = params.transA();
    slate::Diag diag = params.diag();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    scalar_t alpha = params.alpha();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
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

    // Error analysis applies in these norms.
    slate_assert(norm == Norm::One || norm == Norm::Inf || norm == Norm::Fro);

    // setup so B is m-by-n
    int64_t An  = (side == slate::Side::Left ? m : n);
    int64_t Am  = An;
    int64_t Bm  = m;
    int64_t Bn  = n;

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

    // pplghe generates a diagonally dominant matrix.
    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = descA_tst[8];
    std::vector< scalar_t > A_tst(lldA*nlocA);
    scalapack_pplghe(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc(Bm, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(Bn, nb, mycol, izero, npcol);
    scalapack_descinit(descB_tst, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = descB_tst[8];
    std::vector< scalar_t > B_tst(lldB*nlocB);
    scalapack_pplrnt(&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 1);

    // if check is required, copy test data and create a descriptor for it
    std::vector< scalar_t > B_ref;
    slate::Matrix<scalar_t> Bref;
    if (check || ref) {
        scalapack_descinit(descB_ref, Bm, Bn, nb, nb, izero, izero, ictxt, mlocB, &info);
        slate_assert(info == 0);
        B_ref = B_tst;
        Bref = slate::Matrix<scalar_t>::fromScaLAPACK
                   (Bm, Bn, &B_ref[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    if (verbose >= 2) {
        print_matrix( "A", mlocA, nlocA, &A_tst[0], lldA, p, q, MPI_COMM_WORLD);
        print_matrix( "B", mlocB, nlocB, &B_tst[0], lldB, p, q, MPI_COMM_WORLD);
    }

    // Cholesky factor of A to get a well conditioned triangular matrix.
    // Even when we replace the diagonal with unit diagonal,
    // it seems to still be well conditioned.
    auto AH = slate::HermitianMatrix<scalar_t>::fromScaLAPACK
        (uplo, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    slate::potrf(AH, {{slate::Option::Target, target}});

    slate::TriangularMatrix<scalar_t> A;
    slate::Matrix<scalar_t> B;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::TriangularMatrix<scalar_t>
                (uplo, diag, An, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        B = slate::Matrix<scalar_t>
                 (Bm, Bn, nb, nprow, npcol, MPI_COMM_WORLD);
        B.insertLocalTiles(origin_target);
        copy(&B_tst[0], descB_tst, B);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::TriangularMatrix<scalar_t>::fromScaLAPACK
                 (uplo, diag, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        B = slate::Matrix<scalar_t>::fromScaLAPACK
                 (Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
    }
    if (transA == Op::Trans)
        A = transpose(A);
    else if (transA == Op::ConjTrans)
        A = conjTranspose(A);

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
    // slate::trsm(side, alpha, A, B, {
    //     {slate::Option::Lookahead, lookahead},
    //     {slate::Option::Target, target}
    // });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop < scalar_t >::trsm(side, m, n);
    params.time() = time_tst;
    params.gflops() = gflop / time_tst;

    if (verbose >= 2) {
        print_matrix( "B_out", B, 24, 16 );
    }

    if (check) {
        //==================================================
        // Test results by checking the residual
        //
        //      || B - 1/alpha AX ||_1
        //     ------------------------ < epsilon
        //      || A ||_1 * N
        //
        //==================================================

        // get norms of the original data
        // todo: add TriangularMatrix norm
        auto AZ = static_cast< slate::TrapezoidMatrix<scalar_t> >( A );
        real_t A_norm = slate::norm(norm, AZ);

        scalar_t one = 1;
        slate::trmm(side, one/alpha, A, B);
        slate::geadd(-one, Bref, one, B);
        real_t error = slate::norm(norm, B);
        error = error / (Am * A_norm);
        params.error() = error;

        // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 3*eps);
    }

    if (ref) {
        // comparison with reference routine from ScaLAPACK for timing only

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();
        scalapack_ptrsm(side2str(side), uplo2str(uplo), op2str(transA), diag2str(diag),
                        m, n, alpha,
                        &A_tst[0], ione, ione, descA_tst,
                        &B_ref[0], ione, ione, descB_ref);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        if (verbose >= 2) {
            print_matrix( "B_ref", mlocB, nlocB, &B_ref[0], lldB, p, q, MPI_COMM_WORLD, 24, 16 );
        }

        params.ref_time() = time_ref;
        params.ref_gflops() = gflop / time_ref;

        slate_set_num_blas_threads(saved_num_threads);
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering.
}

// -----------------------------------------------------------------------------
void test_trsm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_trsm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trsm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trsm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trsm_work<std::complex<double>> (params, run);
            break;
    }
}
