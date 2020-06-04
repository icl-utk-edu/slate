
#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "band_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_steqr2_work(
    Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::max;
    using llong = long long;
    // typedef long long llong;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    lapack::Job jobz = params.jobz();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho();

    bool wantz = (jobz == slate::Job::Vec);

    if (! run)
        return;

    int mpi_rank, mpi_size;
    slate_mpi_call(
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    // Local values
    scalar_t zero = 0.0; scalar_t one = 1.0;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descZ_tst[9];
    int iam = 0, nprocs = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix Z, figure out local size, allocate, create descriptor, initialize
    int64_t mlocZ = scalapack_numroc(n, nb, myrow, 0, nprow);
    int64_t nlocZ = scalapack_numroc(n, nb, mycol, 0, npcol);
    scalapack_descinit(descZ_tst, n, n, nb, nb, 0, 0, ictxt, mlocZ, &info);
    slate_assert(info == 0);
    int64_t lldZ = (int64_t)descZ_tst[8];
    std::vector<scalar_t> Z_tst(1);

    // skip invalid sizes
    if (n <= (nprow-1)*nb || n <= (npcol-1)*nb) {
        if (iam == 0) {
            printf("\nskipping: ScaLAPACK requires that all ranks have some rows & columns; "
                   "i.e., n > (p-1)*nb = %lld and n > (q-1)*nb = %lld\n",
                   llong( (nprow-1)*nb ), llong( (npcol-1)*nb ) );
        }
        return;
    }

    // Initialize the diagonal and subdiagonal
    std::vector<real_t> D(n), E(n - 1);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, 0, 0, 3 };
    //int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, D.size(), D.data());
    lapack::larnv(idist, iseed, E.size(), E.data());
    std::vector<real_t> Dref = D;
    std::vector<real_t> Eref = E;

    slate::Matrix<scalar_t> A; // To check the orth of the eigenvectors
    if (check) {
        A = slate::Matrix<scalar_t>(n, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles();
    }

    slate::Matrix<scalar_t> Z; // Matrix of the eigenvectors
    if (origin != slate::Origin::ScaLAPACK) {
        if (wantz) {
            Z = slate::Matrix<scalar_t>(
                n, n, nb, nprow, npcol, MPI_COMM_WORLD);
            Z.insertLocalTiles(origin2target(origin));
        }
    }
    else {
        if (wantz) {
            Z_tst.resize(lldZ*nlocZ);
            Z = slate::Matrix<scalar_t>::fromScaLAPACK(
                n, n, &Z_tst[0], lldZ, nb, nprow, npcol, MPI_COMM_WORLD);
        }
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
    //==================================================
    //slate::sterf(D, E);
    steqr2(jobz, D, E, Z);

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    params.time() = testsweeper::get_wtime() - time;

    if (trace)
        slate::trace::Trace::finish();

    if (check) {
        //==================================================
        // Test results
        //==================================================
        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

        //==================================================
        // Run LAPACK reference routine.
        //==================================================
        //MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();

        lapack::sterf(n, &Dref[0], &Eref[0]);

        //MPI_Barrier(MPI_COMM_WORLD);
        params.ref_time() = testsweeper::get_wtime() - time;

        slate_set_num_blas_threads(saved_num_threads);
        if (verbose) {
            // Print first 20 and last 20 rows.
            printf( "%9s  %9s\n", "D", "Dref" );
            for (int64_t i = 0; i < n; ++i) {
                if (i < 20 || i > n-20) {
                    bool okay = std::abs( D[i] - Dref[i] ) < tol;
                    printf( "%9.6f  %9.6f%s\n",
                            D[i], Dref[i], (okay ? "" : " !!") );
                }
            }
            printf( "\n" );
        }

        // Relative forward error: || D - Dref || / || Dref ||.
        real_t err = blas::nrm2(Dref.size(), &Dref[0], 1);
        blas::axpy(D.size(), -1.0, &D[0], 1, &Dref[0], 1);
        params.error() = blas::nrm2(Dref.size(), &Dref[0], 1) / err;

        //==================================================
        // Test results by checking the orthogonality of Q
        //
        //      || Q'Q - I ||_f
        //     ---------------- < tol * epsilon
        //           n
        //
        //==================================================
        const scalar_t minusone = -1;
        params.ortho() = 0.;
        if (wantz) {
            auto ZT = conjTranspose(Z);
            set(zero, one, A);
            slate::gemm(one, ZT, Z, minusone, A);
            params.ortho()  = slate::norm(slate::Norm::Fro, A) / n;
        }
        params.okay() = ((params.error() <= tol) && (params.ortho() <= tol));

    }
}

// -----------------------------------------------------------------------------
void test_steqr2(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_steqr2_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_steqr2_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_steqr2_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_steqr2_work<std::complex<double>> (params, run);
            break;
    }
}
