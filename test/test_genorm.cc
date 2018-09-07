#include "slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"

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
void test_genorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;
    using lld = long long;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    bool check = params.check.value()=='y';
    bool ref = params.ref.value()=='y';
    bool trace = params.trace.value()=='y';
    int verbose = params.verbose.value();
    int extended = params.extended.value();
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

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, i0, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, i0, npcol);
    int64_t lldA  = std::max( int64_t(1), mlocA );
    scalapack_descinit(descA_tst, Am, An, nb, nb, i0, i0, ictxt, lldA, &info);
    assert(info==0);
    std::vector<scalar_t> A_tst(lldA * nlocA);
    // todo: fix the generation
    //int iseed = 1;
    //scalapack_pplrnt(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1);
    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    //lapack::larnv(2, iseeds, lldA * nlocA, &A_tst[0] );
    for (int64_t j = 0; j < nlocA; ++j)
        lapack::larnv(2, iseeds, mlocA, &A_tst[j*lldA]);

    //if (verbose > 1) {
    //    print_matrix(mlocA, nlocA, &A_tst[0], lldA, p, q, MPI_COMM_WORLD);
    //}

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::Matrix<scalar_t> A0(Am, An, nb, p, q, MPI_COMM_WORLD);

    slate::Matrix<scalar_t> A;
    std::vector<scalar_t*> Aarray(A.num_devices());
    if (target == slate::Target::Devices) {
        // Distribute local ScaLAPACK data in 1D-cyclic fashion to GPU devices.
        for (int device = 0; device < A.num_devices(); ++device) {
            int64_t ndevA = scalapack_numroc(nlocA, nb, device, i0, A.num_devices());
            size_t len = blas::max((int64_t)sizeof(scalar_t) * lldA * ndevA, 1);
            slate_cuda_call(
                cudaSetDevice(device));
            slate_cuda_call(
                cudaMalloc((void**)&Aarray[device], len));
            assert(Aarray[device] != nullptr);
            int64_t jj_dev = 0;
            for (int64_t jj_local = nb*device; jj_local < nlocA;
                 jj_local += nb*A.num_devices())
            {
                int64_t jb = std::min(nb, nlocA - jj_local);
                slate_cublas_call(
                    cublasSetMatrix(
                        mlocA, jb, sizeof(scalar_t),
                        &A_tst[ jj_local * lldA ], lldA,
                        &Aarray[device][ jj_dev * lldA ], lldA));
                jj_dev += nb;
            }
        }
        // Create SLATE matrix from the device layout.
        A = slate::Matrix<scalar_t>::fromDevices(
            Am, An, Aarray.data(), Aarray.size(), lldA, nb,
            nprow, npcol, MPI_COMM_WORLD);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layout.
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
            Am, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    if (verbose > 1) {
        print_matrix("A", A);
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // call the test routine
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = libtest::get_wtime();

    real_t A_norm = slate::norm(norm, A, {
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
        std::vector<real_t> worklange(std::max(mlocA, nlocA));

        // run the reference routine
        MPI_Barrier(MPI_COMM_WORLD);
        time = libtest::get_wtime();
        real_t A_norm_ref = scalapack_plange(
            norm2str(norm),
            Am, An, &A_tst[0], i1, i1, descA_tst, &worklange[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

        //A_norm_ref = lapack::lange(
        //    norm,
        //    Am, An, &A_tst[0], lldA);

        // difference between norms
        real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
        if (norm == lapack::Norm::One) {
            error /= sqrt( Am );
        }
        else if (norm == lapack::Norm::Inf) {
            error /= sqrt( An );
        }
        else if (norm == lapack::Norm::Fro) {
            error /= sqrt( Am*An );
        }

        if (verbose && mpi_rank == 0) {
            printf( "norm %15.8e, ref %15.8e, ref - norm %5.2f, error %9.2e\n",
                    A_norm, A_norm_ref, A_norm_ref - A_norm, error );
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

    //---------- extended tests
    if (extended) {
        // allocate work space
        std::vector<real_t> worklange(std::max(mlocA, nlocA));

        // seed all MPI processes the same
        srand( 1234 );

        // Test tiles in 2x2 in all 4 corners, and 4 random rows and cols,
        // up to 64 tiles total.
        // Indices may be out-of-bounds if mt or nt is small, so check in loops.
        int64_t mt = A.mt();
        int64_t nt = A.nt();
        std::set<int64_t> i_indices = { 0, 1, mt-2, mt-1 };
        std::set<int64_t> j_indices = { 0, 1, nt-2, nt-1 };
        for (size_t k = 0; k < 4; ++k) {
            i_indices.insert( rand() % mt );
            j_indices.insert( rand() % nt );
        }
        for (auto j: j_indices) {
            if (j < 0 || j >= nt)
                continue;
            int64_t jb = std::min( n - j*nb, nb );
            assert( jb == A.tileNb( j ) );

            for (auto i: i_indices) {
                if (i < 0 || i >= mt)
                    continue;
                int64_t ib = std::min( m - i*nb, nb );
                assert( ib == A.tileMb( i ) );

                // Test entries in 2x2 in all 4 corners, and 1 other random row and col,
                // up to 25 entries per tile.
                // Indices may be out-of-bounds if ib or jb is small, so check in loops.
                std::set<int64_t> ii_indices = { 0, 1, ib-2, ib-1, rand() % ib };
                std::set<int64_t> jj_indices = { 0, 1, jb-2, jb-1, rand() % jb };

                // todo: complex peak
                scalar_t peak = rand() / double(RAND_MAX) * 1e6 + 1e6;
                if (rand() < RAND_MAX/2)
                    peak *= -1;
                if (rand() < RAND_MAX/20)
                    peak = nan("");
                scalar_t save = 0;

                for (auto jj: jj_indices) {
                    if (jj < 0 || jj >= jb)
                        continue;

                    for (auto ii: ii_indices) {
                        if (ii < 0 || ii >= ib) {
                            continue;
                        }

                        int64_t ilocal = int(i / p)*nb + ii;
                        int64_t jlocal = int(j / q)*nb + jj;
                        if (A.tileIsLocal(i, j)) {
                            A.tileMoveToHost(i, j, A.tileDevice(i, j));
                            auto T = A(i, j);
                            save = T(ii, jj);
                            T.at(ii, jj) = peak;
                            A_tst[ ilocal + jlocal*lldA ] = peak;
                            // todo: this move shouldn't be required -- the trnorm should copy data itself.
                            A.tileMoveToDevice(i, j, A.tileDevice(i, j));
                        }

                        real_t A_norm = slate::norm(norm, A, {
                            {slate::Option::Target, target}
                        });

                        real_t A_norm_ref = scalapack_plange(
                            norm2str(norm),
                            Am, An, &A_tst[0], i1, i1, descA_tst, &worklange[0]);

                        // difference between norms
                        real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
                        if (norm == lapack::Norm::One) {
                            error /= sqrt( Am );
                        }
                        else if (norm == lapack::Norm::Inf) {
                            error /= sqrt( An );
                        }
                        else if (norm == lapack::Norm::Fro) {
                            error /= sqrt( Am*An );
                        }

                        // Allow for difference, except max norm in real should be exact.
                        real_t eps = std::numeric_limits<real_t>::epsilon();
                        real_t tol;
                        if (norm == lapack::Norm::Max && ! slate::is_complex<scalar_t>::value)
                            tol = 0;
                        else
                            tol = 3*eps;

                        if (mpi_rank == 0) {
                            // if peak is nan, expect A_norm to be nan.
                            bool okay = (std::isnan(real(peak))
                                            ? std::isnan(A_norm)
                                            : error <= tol);
                            params.okay.value() = params.okay.value() && okay;
                            if (verbose || ! okay) {
                                printf( "i %5lld, j %5lld, ii %3lld, jj %3lld, peak %15.8e, norm %15.8e, ref %15.8e, error %9.2e, %s\n",
                                        (lld) i, (lld) j, (lld) ii, (lld) jj,
                                        real(peak), A_norm, A_norm_ref, error,
                                        (okay ? "pass" : "failed") );
                            }
                        }

                        if (A.tileIsLocal(i, j)) {
                            A.tileMoveToHost(i, j, A.tileDevice(i, j));
                            auto T = A(i, j);
                            T.at(ii, jj) = save;
                            A_tst[ ilocal + jlocal*lldA ] = save;
                            // todo: this move shouldn't be required -- the trnorm should copy data itself.
                            A.tileMoveToDevice(i, j, A.tileDevice(i, j));
                        }
                    }
                }
            }
        }
    }

    //---------- cleanup
    if (target == slate::Target::Devices) {
        for (int device = 0; device < A.num_devices(); ++device) {
            slate_cuda_call(
                cudaFree(Aarray[device]));
            Aarray[device] = nullptr;
        }
    }

    //Cblacs_exit(1) is commented out because it does not handle re-entering ... some unknown problem
    //Cblacs_exit(1); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_genorm(Params& params, bool run)
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_genorm_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_genorm_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_genorm_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_genorm_work<std::complex<double>> (params, run);
            break;
    }
}
