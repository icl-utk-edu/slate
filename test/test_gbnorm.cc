#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"
#include "band_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_gbnorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using blas::min;
    using blas::max;
    using slate::ceildiv;
    //using llong = long long;

    // get & mark input values
    slate::Norm norm = params.norm();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
    int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported\n");
        return;
    }
    if (target == slate::Target::Devices) {
        printf("skipping: currently target=devices is not supported\n");
        return;
    }

    // local values
    const int izero = 0, ione = 1;

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9];
    int iam = 0, nprocs = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(m, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    int64_t lldA  = std::max(int64_t(1), mlocA);
    scalapack_descinit(descA_tst, m, n, nb, nb, izero, izero, ictxt, lldA, &info);
    slate_assert(info == 0);
    std::vector<scalar_t> A_tst(lldA*nlocA);
    // todo: fix the generation
    //int iseed = 1;
    //scalapack_pplrnt(&A_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1);
    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    //lapack::larnv(2, iseeds, lldA*nlocA, &A_tst[0] );
    for (int64_t j = 0; j < nlocA; ++j)
        lapack::larnv(2, iseeds, mlocA, &A_tst[j*lldA]);
    zeroOutsideBand(&A_tst[0], m, n, kl, ku, nb, nb, myrow, mycol, nprow, npcol, lldA);

    // Create SLATE matrix from the ScaLAPACK layout.
    // TODO: data origin on GPU
    auto A = BandFromScaLAPACK(
                 m, n, kl, ku, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);

    if (verbose > 2) {
        print_matrix("A_tst", mlocA, nlocA, &A_tst[0], lldA, p, q, MPI_COMM_WORLD);
    }
    if (verbose > 1) {
        print_matrix("A", A);
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
    // Compute || A ||_norm.
    //==================================================
    real_t A_norm = slate::norm(norm, A, {
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

    if (check || ref) {
        // comparison with reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        // allocate work space
        std::vector<real_t> worklange(std::max(mlocA, nlocA));

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();
        real_t A_norm_ref = scalapack_plange(
                                norm2str(norm),
                                m, n, &A_tst[0], ione, ione, descA_tst, &worklange[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        // difference between norms
        real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
        if (norm == slate::Norm::One) {
            error /= sqrt(m);
        }
        else if (norm == slate::Norm::Inf) {
            error /= sqrt(n);
        }
        else if (norm == slate::Norm::Fro) {
            error /= sqrt(m*n);
        }

        if (verbose && mpi_rank == 0) {
            printf("norm %15.8e, ref %15.8e, ref - norm %5.2f, error %9.2e\n",
                   A_norm, A_norm_ref, A_norm_ref - A_norm, error);
        }

        // Allow for difference, except max norm in real should be exact.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        real_t tol;
        if (norm == slate::Norm::Max && ! slate::is_complex<scalar_t>::value)
            tol = 0;
        else
            tol = 3*eps;

        params.ref_time() = time_ref;
        params.error() = error;

        slate_set_num_blas_threads(saved_num_threads);

        // Allow for difference
        params.okay() = (params.error() <= tol);
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_gbnorm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gbnorm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gbnorm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbnorm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbnorm_work<std::complex<double>> (params, run);
            break;
    }
}
