#include "slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"

#include "slate_mpi.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads(int nt);
inline int slate_set_num_blas_threads(const int nt) { return MKL_Set_Num_Threads(nt); }
#else
inline int slate_set_num_blas_threads(const int nt) { return -1; }
#endif

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_trnorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    lapack::Uplo uplo = params.uplo.value();
    lapack::Diag diag = params.diag.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    bool check = params.check.value()=='y';
    bool ref = params.ref.value()=='y';
    bool trace = params.trace.value()=='y';
    int verbose = params.verbose.value();
    slate::Target target = char2target(params.target.value());

    // mark non-standard output values
    params.time.value();
    params.ref_time.value();

    if (! run)
        return;

    // Sizes of data
    int64_t Am = m;
    int64_t An = n;

    // local values
    static int i0=0, i1=1;

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9];
    int iam=0, nprocs=1;
    int iseed = 1;

//printf( "%d Cblacs\n", mpi_rank ); fflush( stdout );
    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, i0, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, i0, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info);
    assert(info==0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA * nlocA);
    scalapack_pplrnt(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1);

    int i = rand() % lldA;
    int j = rand() % nlocA;
    A_tst[i + j*lldA] = -12.3456;

//printf( "%d TrapezoidMatrix\n", mpi_rank ); fflush( stdout );
    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::TrapezoidMatrix<scalar_t> A0(uplo, Am, An, nb, p, q, MPI_COMM_WORLD);

    slate::TrapezoidMatrix<scalar_t> A;
    std::vector<scalar_t*> Aarray(A.num_devices());
    if (target == slate::Target::Devices) {
        // Distribute local ScaLAPACK data in 1D-cyclic fashion to GPU devices.
        for (int device = 0; device < A.num_devices(); ++device) {
            int64_t ndevA = scalapack_numroc(nlocA, nb, device, i0, A.num_devices());
            size_t len = blas::max((int64_t)sizeof(double) * lldA * ndevA, 1);
            cudaMalloc(&Aarray[device], len);
            assert(Aarray[device] != nullptr);
            int64_t jj_dev = 0;
            for (int64_t jj_local = device*nb; jj_local < nlocA; jj_local += nb) {
                int64_t jb = std::min(nb, nlocA - jj_local);
                cublasSetMatrix(mlocA, jb, sizeof(scalar_t),
                                &A_tst[ jj_local * lldA ], lldA,
                                &Aarray[device][ jj_dev * lldA ], lldA);
                jj_dev += nb;
            }
        }
        // Create SLATE matrix from the device layout.
        A = slate::TrapezoidMatrix<scalar_t>::fromDevices(
            uplo, Am, An, Aarray.data(), Aarray.size(), lldA, nb,
            nprow, npcol, MPI_COMM_WORLD);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layout.
        A = slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(
            uplo, Am, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // call the test routine
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = libtest::get_wtime();

//printf( "%d slate::trnorm\n", mpi_rank ); fflush( stdout );
    real_t A_norm = slate::trnorm(norm, diag, A, {
        {slate::Option::Target, target}
    });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = libtest::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time.value() = time_tst;

    if (check || ref) {
        // comparison with reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        // allocate work space
        std::vector<real_t> worklantr(std::max(mlocA, nlocA));

        // run the reference routine
//printf( "%d scalapack_plantr\n", mpi_rank ); fflush( stdout );
        MPI_Barrier(MPI_COMM_WORLD);
        time = libtest::get_wtime();
        real_t A_norm_ref = scalapack_plantr(
            norm2str(norm), uplo2str(A.uplo()), diag2str(diag),
            Am, An, &A_tst[0], i1, i1, descA_tst, &worklantr[0]);
          //Am-4, An-4, &A_tst[0], 5, 5, descA_tst, &worklantr[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

        real_t A_norm_la = lapack::lantr(norm, uplo, diag, Am, An, &A_tst[0], lldA);
        real_t error_la = std::abs(A_norm - A_norm_la) / A_norm_la;

        // difference between norms
        real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
        if (norm == lapack::Norm::One) {
            error /= sqrt( Am );
            error_la /= sqrt( Am );
        }
        else if (norm == lapack::Norm::Inf) {
            error /= sqrt( An );
            error_la /= sqrt( An );
        }
        else if (norm == lapack::Norm::Fro) {
            error /= sqrt( Am*An );
            error_la /= sqrt( Am*An );
        }

        if (verbose) {
            printf( "rank %d, norm %.8e, ref %.8e, la %.8e, error %.2e, error_la %.2e ",
                    mpi_rank, A_norm, A_norm_ref, A_norm_la, error, error_la );
        }

        // Allow for difference, except max norm in real should be exact.
        real_t eps = std::numeric_limits<real_t>::epsilon();
        real_t tol;
        if (norm == lapack::Norm::Max && ! slate::is_complex<scalar_t>::value)
            tol = 0;
        else
            tol = 3*eps;

        params.ref_time.value() = time_ref;
        params.error.value() = error;

        slate_set_num_blas_threads(saved_num_threads);

        // Allow for difference
        params.okay.value() = (params.error.value() <= tol);
    }

    if (target == slate::Target::Devices) {
        for (int device = 0; device < A.num_devices(); ++device) {
            cudaFree(Aarray[device]);
            Aarray[device] = nullptr;
        }
    }

    //Cblacs_exit(1) is commented out because it does not handle re-entering ... some unknown problem
    //Cblacs_exit(1); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_trnorm(Params& params, bool run)
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_trnorm_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_trnorm_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_trnorm_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_trnorm_work<std::complex<double>> (params, run);
            break;
    }
}
