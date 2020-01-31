
#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "scalapack_support_routines.hh"
#include "internal/internal.hh"
#include "band_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_bdsqr_work(
    Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    // typedef long long llong;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    lapack::Job jobu = params.jobu();
    lapack::Job jobvt = params.jobvt();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.ortho_U();
    params.ortho_V();

    if (! run)
        return;

    int mpi_rank, mpi_size;
    slate_mpi_call(
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    int64_t min_mn = std::min(m, n);
    scalar_t zero = 0.0, one = 1.0, minusone = -1;
    // local values
    // const int izero = 0;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol;
    // int descA_tst[9];
    int iam = 0, nprocs = 1;


    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    std::vector<real_t> D(n), E(n - 1);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, 0, 0, 3 };
    lapack::larnv(idist, iseed, D.size(), D.data());
    lapack::larnv(idist, iseed, E.size(), E.data());
    std::vector<real_t> Dref = D;
    std::vector<real_t> Eref = E;

    slate::Matrix<scalar_t> U;
    slate::Matrix<scalar_t> VT;

    char jobu_  = job_compu2char( jobu );
    char jobvt_ = job_compu2char( jobvt );

    //if (jobu_ == 'V') {
        U = slate::Matrix<scalar_t>(m, min_mn, nb, nprow, npcol, MPI_COMM_WORLD);
        U.insertLocalTiles();
    //}

    //if (jobu_ == 'V') {
        VT = slate::Matrix<scalar_t>(min_mn, n, nb, nprow, npcol, MPI_COMM_WORLD);
        VT.insertLocalTiles();
    //}

    //---------
    // run test
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
    slate::bdsqr<scalar_t>(jobu, jobvt, D, E, U, VT);

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
        MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();

        scalar_t dummy[1];  // U, VT, C not needed for NoVec
        lapack::bdsqr(uplo, n, 0, 0, 0,
                      &Dref[0], &Eref[0], dummy, 1, dummy, 1, dummy, 1);

        MPI_Barrier(MPI_COMM_WORLD);
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
        blas::axpy(D.size(), -1.0, &Dref[0], 1, &D[0], 1);
        params.error() = blas::nrm2(D.size(), &D[0], 1)
                       / blas::nrm2(Dref.size(), &Dref[0], 1);

        //==================================================
        // Test results by checking the orthogonality of Q
        //
        //      || Q'Q - I ||_f
        //     ---------------- < tol * epsilon
        //           n      
        //
        //==================================================
        params.ortho_U() = 0.;
        params.ortho_V() = 0.;
        if (jobu_ == 'V') {
            slate::Matrix<scalar_t> Id;
            Id = slate::Matrix<scalar_t>(min_mn, min_mn, nb, nprow, npcol, MPI_COMM_WORLD);
            Id.insertLocalTiles();
            set( zero, one, Id);

            const scalar_t minusone = -1;
            auto UT = conj_transpose(U);
            slate::gemm(one, UT, U, minusone, Id);
            params.ortho_U()  = slate::norm(slate::Norm::Fro, Id) / std::sqrt(m);
        }
        // If we flip the fat matrix, then no need for Id_nn
        if (jobu_ == 'V') {
            slate::Matrix<scalar_t> Id_nn;
            Id_nn = slate::Matrix<scalar_t>(n, n, nb, nprow, npcol, MPI_COMM_WORLD);
            Id_nn.insertLocalTiles();
            set( zero, one, Id_nn);
            auto VTT = conj_transpose(VT);
            slate::gemm(one, VTT, VT, minusone, Id_nn);
            params.ortho_V()  = slate::norm(slate::Norm::Fro, Id_nn) / std::sqrt(n);
        }
        params.okay() = ( (params.error() <= tol) && (params.ortho_U() <= tol) 
                          && (params.ortho_V() <= tol));

    }
}

// -----------------------------------------------------------------------------
void test_bdsqr(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_bdsqr_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_bdsqr_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_bdsqr_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_bdsqr_work<std::complex<double>> (params, run);
            break;
    }
}
