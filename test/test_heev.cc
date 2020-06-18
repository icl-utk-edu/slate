#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_heev_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using llong = long long;

    // get & mark input values
    lapack::Job jobz = params.jobz();
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Norm norm = params.norm();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    params.matrix.mark();

    slate_assert(p == q);  // heev requires square process grid.

    params.time();
    params.ref_time();
    // params.gflops();
    // params.ref_gflops();

    if (! run)
        return;

    // Local values
    const int izero = 0, ione = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    bool wantz = (jobz == slate::Job::Vec);

    // figure out local size, allocate, create descriptor, initialize
    // matrix A (local input), m-by-n, symmetric matrix
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    int descA_tst[9];
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplghe(&A_tst[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix W (global output), W(n), gets eigenvalues in decending order
    std::vector<real_t> W_tst(n);

    // matrix Z (local output), Z(n,n), gets orthonormal eigenvectors corresponding to W of the reference scalapack
    int64_t mlocZ = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocZ = scalapack_numroc(n, nb, mycol, izero, npcol);
    int descZ_tst[9];
    scalapack_descinit(descZ_tst, n, n, nb, nb, izero, izero, ictxt, mlocZ, &info);
    slate_assert(info == 0);
    int64_t lldZ = (int64_t)descZ_tst[8];
    std::vector<scalar_t> Z_tst(lldZ * nlocZ, 0);

    // matrix Q (local output), Q(n,n), gets orthonormal eigenvectors corresponding to W of slate heev
    int64_t mlocQ = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocQ = scalapack_numroc(n, nb, mycol, izero, npcol);
    int descQ_tst[9];
    scalapack_descinit(descQ_tst, n, n, nb, nb, izero, izero, ictxt, mlocQ, &info);
    slate_assert(info == 0);
    int64_t lldQ = (int64_t)descQ_tst[8];
    std::vector<scalar_t> Q_tst(lldQ * nlocQ, 0);

    // Initialize SLATE data structures
    slate::HermitianMatrix<scalar_t> A;
    std::vector<real_t> W;
    slate::Matrix<scalar_t> Z;
    slate::Matrix<scalar_t> Q;

    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        W = W_tst;

        Z = slate::Matrix<scalar_t>(n, n, nb, nprow, npcol, MPI_COMM_WORLD);
        Z.insertLocalTiles(origin_target);
        if (wantz) {
            Q = slate::Matrix<scalar_t>(
                n, n, nb, nprow, npcol, MPI_COMM_WORLD);
            Q.insertLocalTiles(origin2target(origin));
        }
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
        W = W_tst;
        Z = slate::Matrix<scalar_t>::fromScaLAPACK(n, n, &Z_tst[0], lldZ, nb, nprow, npcol, MPI_COMM_WORLD);
        if (wantz) {
            Q_tst.resize(lldQ*nlocQ);
            Q = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Q_tst[0], lldQ, nb, nprow, npcol, MPI_COMM_WORLD);
        }
    }

    //lapack::TestMatrixType type = lapack::TestMatrixType::heev;
    //params.matrix.kind.set_default("heev");
    //params.matrix.cond.set_default(1e4);

    slate::generate_matrix( params.matrix, Z);
    A = slate::HermitianMatrix<scalar_t>(
               uplo, Z );
    copy(A, &A_tst[0], descA_tst);

    if (verbose >= 1) {
        printf( "%% A   %6lld-by-%6lld\n", llong(   A.m() ), llong(   A.n() ) );
        printf( "%% Z   %6lld-by-%6lld\n", llong(   Z.m() ), llong(   Z.n() ) );
    }

    if (verbose > 1) {
        print_matrix( "A",  A  );
    }

    std::vector<scalar_t> A_ref, Z_ref;
    std::vector<real_t> W_ref;
    if (check || ref) {
        A_ref = A_tst;
        W_ref = W_tst;
        Z_ref = Z_tst;
    }

    // SLATE test
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

        if (jobz == slate::Job::NoVec) {
            slate::eig_vals(A, W_tst, {
                    {slate::Option::Lookahead, lookahead},
                    {slate::Option::Target, target},
                    {slate::Option::MaxPanelThreads, panel_threads},
                    {slate::Option::InnerBlocking, ib}
                });
        }
        // else {
            // todo: slate::Job::Vec
        // }

        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::heev(jobz, A, W_tst, Q, {
        //         {slate::Option::Lookahead, lookahead},
        //         {slate::Option::Target, target},
        //         {slate::Option::MaxPanelThreads, panel_threads},
        //         {slate::Option::InnerBlocking, ib}
        //     });

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;

        if (verbose > 1) {
            print_matrix( "A",  A  );
            print_matrix( "Z",  Z  );
        }
    }

    if (check || ref) {
        // Run reference routine from ScaLAPACK

        // set num threads appropriately for parallel BLAS if possible
        int omp_num_threads = 1;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        // query for workspace size
        int64_t info_tst = 0;
        int64_t lwork = -1, lrwork = -1;
        std::vector<scalar_t> work(1);
        std::vector<real_t> rwork(1);
        scalapack_pheev(job2str(jobz), uplo2str(uplo), n,
                        &A_ref[0], ione, ione, descA_tst,
                        &W_ref[0],
                        &Z_ref[0], ione, ione, descZ_tst,
                        &work[0], -1, &rwork[0], -1, &info_tst);
        slate_assert(info_tst == 0);
        lwork = int64_t( real( work[0] ) );
        work.resize(lwork);
        // The lrwork, rwork parameters are only valid for complex
        if (slate::is_complex<scalar_t>::value) {
            lrwork = int64_t( real( rwork[0] ) );
            rwork.resize(lrwork);
        }
        // Run ScaLAPACK reference routine.
        MPI_Barrier(MPI_COMM_WORLD);
        double time = testsweeper::get_wtime();
        scalapack_pheev(job2str(jobz), uplo2str(uplo), n,
                        &A_ref[0], ione, ione, descA_tst,
                        &W_ref[0],
                        &Z_ref[0], ione, ione, descZ_tst,
                        &work[0], lwork, &rwork[0], lrwork, &info_tst);
        slate_assert(info_tst == 0);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        params.ref_time() = time_ref;

        // Reset omp thread number
        slate_set_num_blas_threads(saved_num_threads);

        // Reference Scalapack was run, check reference against test
        // Perform a local operation to get differences W_tst = W_tst - W_ref
        blas::axpy(W_ref.size(), -1.0, &W_ref[0], 1, &W_tst[0], 1);

        real_t reduced_error;
        real_t local_error;
        // Relative forward error: || W_ref - W_tst || / || W_ref ||
        local_error = lapack::lange(norm, W_tst.size(), 1, &W_tst[0], 1)
                       / lapack::lange(norm, W_ref.size(), 1, &W_ref[0], 1);

        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

        if (local_error > tol) {
            printf("\nOn MPI Rank = %d, the eigenvalues are suspicious, the error is  %e \n",
                A.mpiRank(), params.error());
            //for (int64_t i = 0; i < n; i++) {
            //    printf("\n %f", W_tst[i]);
            //}
        }

        slate_mpi_call(
            MPI_Allreduce( &local_error, &reduced_error,
                           1, slate::mpi_type<real_t>::value,
                           MPI_MAX, A.mpiComm()));

        params.error() = reduced_error;
        params.okay() = (params.error() <= tol);
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_heev(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_heev_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_heev_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_heev_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_heev_work<std::complex<double>> (params, run);
            break;
    }
}
