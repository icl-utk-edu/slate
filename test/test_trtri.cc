#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_trtri_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Diag diag = params.diag();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose(); SLATE_UNUSED(verbose);
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9];
    int descC_chk[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // scalapack matrix A_tst, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    // scalapack_pplghe initializes a hermitian matrix in A_tst
    scalapack_pplghe(&A_tst[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::Matrix<scalar_t> A0(n, n, nb, p, q, MPI_COMM_WORLD);

    // Cholesky factor of A_tst to get a well conditioned triangular matrix
    // Even when we replace the diagonal with unit diagonal,
    // it seems to still be well conditioned.
    auto AH = slate::HermitianMatrix<scalar_t>::fromScaLAPACK
              (uplo, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    slate::potrf(AH);

    // Setup SLATE triangular matrix A based on data in A_tst (Cholesky factored)
    slate::TriangularMatrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::TriangularMatrix<scalar_t>
                (uplo, diag, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);
    }
    else {
        // Create SLATE matrix on CPU from the ScaLAPACK data in A_tst
        A = slate::TriangularMatrix<scalar_t>::fromScaLAPACK
            (uplo, diag, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    // if check (or ref) is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);
    }

    // If check is required, record the norm of the original triangular matrix
    real_t A_norm = 0.0;
    if (check) {
        // todo: add TriangularMatrix norm
        auto AZ = static_cast< slate::TrapezoidMatrix<scalar_t> >( A );
        slate::Norm norm = slate::Norm::One;
        A_norm = slate::norm(norm, AZ);
    }

    // if check is required, create matrix to hold the result of multiplying A and A_inv
    std::vector<scalar_t> C_chk;
    if (check) {
        // C_chk starts with the same size/dimensions as A_tst
        C_chk = A_tst;
        scalapack_descinit(descC_chk, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);
    }

    // Create SLATE matrix from the ScaLAPACK layouts
    slate::Matrix<scalar_t> C;
    C = slate::Matrix<scalar_t>::fromScaLAPACK(n, n, &C_chk[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);

    // trtri flop count
    double gflop = lapack::Gflop<scalar_t>::trtri(n);

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
        //==================================================
        // invert and measure time
        slate::trtri(A, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;
    }

    if (check) {
        //==================================================
        // Check  || I - inv(A)*A || / ( || A || * N ) <=  tol * eps

        if (origin != slate::Origin::ScaLAPACK) {
            // Copy data back from CPU/GPUs to ScaLAPACK layout
            copy(A, &A_tst[0], descA_tst);
        }

        // C_chk has been setup as an identity matrix
        // A_ref = inv(A) * A_ref
        scalar_t minus_one = -1.0, one = 1.0, zero = 0.0;
        scalapack_ptrmm("left", uplo2str(uplo), "notrans", diag2str(diag),
                        n, n, one,
                        &A_tst[0], ione, ione, descA_tst,
                        &A_ref[0], ione, ione, descA_ref);


        // Setup full nxn SLATE matrix in A0 on CPU pointing to ScaLAPACK data in A_tst
        A0 = slate::Matrix<scalar_t>::fromScaLAPACK
            (n, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        // Make C_chk into an identity matrix
        slate::set(zero, minus_one, C);
        // C = C - A ; note A0 is a general nxn SLATE matrix pointing to A_tst data
        slate::geadd(minus_one, A0, one, C);

        // Norm of C_chk ( = I - inv(A) * A )
        //// real_t C_norm = slate::norm(slate::norm::One, C);
        std::vector<real_t> worklange(n);
        real_t C_norm = scalapack_plange("One", n, n, &C_chk[0], ione, ione, descC_chk, &worklange[0]);

        double residual = C_norm / (A_norm * n);
        params.error() = residual;

        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref) {
        // todo: call to reference trtri from ScaLAPACK not implemented
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_trtri(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_trtri_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_trtri_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_trtri_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trtri_work<std::complex<double>> (params, run);
            break;
    }
}
