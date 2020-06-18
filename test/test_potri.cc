#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_potri_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
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

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplghe(&A_tst[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::HermitianMatrix<scalar_t> A0(uplo, n, nb, p, q, MPI_COMM_WORLD);

    // Setup SLATE matrix A based on Scalapack matrix and data in A_tst
    slate::HermitianMatrix<scalar_t> A;
    if (origin == slate::Origin::Devices) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    // if check (or ref) is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);
    }

    // If check is required: keep the norm(A original); create C_chk = identity matrix to hold A*inv(A)
    std::vector<scalar_t> C_chk;
    real_t A_norm = 0.0;
    if (check) {
        // If check is required, record the norm of the original matrix
        A_norm = slate::norm(slate::Norm::One, A);
        // if check is required, create identity matrix to check the result of multiplying A and A_inv
        // C_chk starts with the same size/dimensions as A_tst
        C_chk = A_tst;
        scalapack_descinit(descC_chk, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);
        // Make C_chk into an identity matrix
        scalar_t zero = 0.0; scalar_t one = 1.0;
        scalapack_plaset("All", n, n, zero, one, &C_chk[0], ione, ione, descC_chk);
    }

    double gflop = 0.0;
    // 1/3 n^3 + 1/2 n^2 flops for Cholesky factorization
    // 2/3 n^3 + 1/2 n^2 flops for Cholesky inversion
    gflop = lapack::Gflop<scalar_t>::potrf(n) + lapack::Gflop<scalar_t>::potri(n);

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
        // factor then invert; measure time for both
        slate::chol_factor(A, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });
        slate::chol_inverse_using_factor(A, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}
        });

        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::potrf(A, {
        //     {slate::Option::Lookahead, lookahead},
        //     {slate::Option::Target, target}
        // });
        // slate::potri(A, {
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
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;
    }

    if (check) {
        // Check  || I - inv(A)*A || / ( || A || * N ) <=  tol * eps

        if (origin != slate::Origin::ScaLAPACK) {
            // Copy SLATE result back from GPU or CPU tiles.
            copy(A, &A_tst[0], descA_tst);
        }

        // to make the diagonal of A_ref real
        slate::HermitianMatrix<scalar_t> Aref;
        Aref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                      uplo, n, &A_ref[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        // make diagonal real
        for (int64_t i = 0; i < Aref.nt(); ++i) {
            if (Aref.tileIsLocal(i, i)) {
                auto T = Aref(i, i);
                for (int ii = 0; ii < Aref.tileMb(i); ++ii) {
                    T.at(ii, ii) = std::real( T.at(ii, ii) );
                }
            }
        }

        // C_chk has been setup as an identity matrix; C_chk = C_chk - inv(A)*A
        // A should have real diagonal. potrf and potri ignore the img part on the diagonal
        scalar_t alpha = -1.0; scalar_t beta = 1.0;
        scalapack_phemm("Left", uplo2str(uplo), n, n, alpha,
                        &A_tst[0], ione, ione, descA_tst,
                        &A_ref[0], ione, ione, descA_ref, beta,
                        &C_chk[0], ione, ione, descC_chk);

        // Norm of C_chk ( = I - inv(A) * A )
        // allocate work space for lange and lanhe
        int lcm = scalapack_ilcm(&nprow, &npcol);
        int ldw = nb*slate::ceildiv(int(slate::ceildiv(nlocA, nb)), (lcm / nprow));
        int lwork = std::max(n, 2*mlocA + nlocA + ldw);
        std::vector<real_t> worknorm(lwork);
        real_t C_norm = scalapack_plange(
                            "One", n, n, &C_chk[0], ione, ione, descC_chk, &worknorm[0]);

        real_t A_inv_norm = scalapack_planhe(
                                "One", uplo2str(A.uplo()),
                                n, &A_tst[0], ione, ione, descA_tst, &worknorm[0]);

        double residual = C_norm / (A_norm * n * A_inv_norm);
        params.error() = residual;

        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref) {
        // todo: call to reference potri from ScaLAPACK not implemented
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_potri(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_potri_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_potri_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_potri_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_potri_work<std::complex<double>> (params, run);
            break;
    }
}
