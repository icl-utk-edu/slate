
#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "scalapack_support_routines.hh"
#include "band_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_sterf_work(
    Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    // typedef long long llong;

    // get & mark input values
    int64_t n = params.dim.n();
    //int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();

    // mark non-standard output values
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    int mpi_rank, mpi_size;
    slate_mpi_call(
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

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
    int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, D.size(), D.data());
    lapack::larnv(idist, iseed, E.size(), E.data());
    std::vector<real_t> Dref = D;
    std::vector<real_t> Eref = E;

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
    slate::sterf(D, E);

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
        if (mpi_rank == 0) {
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

            lapack::sterf(n, &Dref[0], &Eref[0]);

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
            params.okay() = (params.error() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_sterf(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_sterf_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_sterf_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_sterf_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_sterf_work<std::complex<double>> (params, run);
            break;
    }
}
