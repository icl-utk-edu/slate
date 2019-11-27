#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template<typename scalar_t>
void test_synorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;
    using llong = long long;

    // get & mark input values
    slate::Norm norm = params.norm();
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    bool check = params.check() == 'y';
    bool ref = params.ref() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    int extended = params.extended();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

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
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    int64_t lldA  = std::max(int64_t(1), mlocA);
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, lldA, &info);
    slate_assert(info == 0);
    std::vector<scalar_t> A_tst(lldA*nlocA);
    // todo: fix the generation
    // int iseed = 1;
    // scalapack_pplrnt(&A_tst[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1);
    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    //lapack::larnv(2, iseeds, lldA*nlocA, &A_tst[0] );
    for (int64_t j = 0; j < nlocA; ++j)
        lapack::larnv(2, iseeds, mlocA, &A_tst[j*lldA]);

    //if (verbose > 1) {
    //    print_matrix("A_tst", mlocA, nlocA, &A_tst[0], lldA, p, q, MPI_COMM_WORLD);
    //}

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::SymmetricMatrix<scalar_t> A0(uplo, n, nb, p, q, MPI_COMM_WORLD);

    slate::SymmetricMatrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::SymmetricMatrix<scalar_t>(uplo, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layout.
        A = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK(
                uplo, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
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
        int lcm = scalapack_ilcm(&nprow, &npcol);
        int ldw = nb*slate::ceildiv(int(slate::ceildiv(nlocA, nb)), (lcm / nprow));
        int lwork = 2*mlocA + nlocA + ldw;
        std::vector<real_t> worklansy(lwork);

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();
        real_t A_norm_ref = scalapack_plansy(
                                norm2str(norm), uplo2str(A.uplo()),
                                n, &A_tst[0], ione, ione, descA_tst, &worklansy[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        //A_norm_ref = lapack::lansy(
        //    norm, A.uplo(),
        //    n, &A_tst[0], lldA);

        // difference between norms
        real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
        if (norm == slate::Norm::One || norm == slate::Norm::Inf) {
            error /= sqrt(n);
        }
        else if (norm == slate::Norm::Fro) {
            error /= n;  // = sqrt( n*n );
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

    //---------- extended tests
    if (extended) {
        // allocate work space
        int lcm = scalapack_ilcm(&nprow, &npcol);
        int ldw = nb*slate::ceildiv(int(slate::ceildiv(nlocA, nb)), (lcm / nprow));
        int lwork = 2*mlocA + nlocA + ldw;
        std::vector<real_t> worklansy(lwork);

        // seed all MPI processes the same
        srand(1234);

        // Test tiles in 2x2 in all 4 corners, and 4 random rows and cols,
        // up to 64 tiles total.
        // Indices may be out-of-bounds if nt is small, so check in loops.
        int64_t nt = A.nt();
        std::set<int64_t> j_indices = { 0, 1, nt - 2, nt - 1 };
        for (size_t k = 0; k < 4; ++k) {
            j_indices.insert(rand() % nt);
        }
        for (auto j : j_indices) {
            if (j < 0 || j >= nt)
                continue;
            int64_t jb = std::min(n - j*nb, nb);
            slate_assert(jb == A.tileNb(j));

            for (auto i : j_indices) {
                // lower requires i >= j
                // upper requires i <= j
                if (i < 0 || i >= nt || (uplo == slate::Uplo::Lower ? i < j : i > j))
                    continue;
                int64_t ib = std::min(n - i*nb, nb);
                slate_assert(ib == A.tileMb(i));

                // Test entries in 2x2 in all 4 corners, and 1 other random row and col,
                // up to 25 entries per tile.
                // Indices may be out-of-bounds if ib or jb is small, so check in loops.
                std::set<int64_t> ii_indices = { 0, 1, ib - 2, ib - 1, rand() % ib };
                std::set<int64_t> jj_indices = { 0, 1, jb - 2, jb - 1, rand() % jb };

                // todo: complex peak
                scalar_t peak = rand() / double(RAND_MAX)*1e6 + 1e6;
                if (rand() < RAND_MAX / 2)
                    peak *= -1;
                if (rand() < RAND_MAX / 20)
                    peak = nan("");
                scalar_t save = 0;

                for (auto jj : jj_indices) {
                    if (jj < 0 || jj >= jb)
                        continue;

                    for (auto ii : ii_indices) {
                        if (ii < 0 || ii >= ib ||
                            (i == j && (uplo == slate::Uplo::Lower ? ii < jj : ii > jj))) {
                            continue;
                        }

                        int64_t ilocal = int(i / p)*nb + ii;
                        int64_t jlocal = int(j / q)*nb + jj;
                        if (A.tileIsLocal(i, j)) {
                            A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
                            auto T = A(i, j);
                            save = T(ii, jj);
                            T.at(ii, jj) = peak;
                            A_tst[ ilocal + jlocal*lldA ] = peak;
                            // todo: this move shouldn't be required -- the trnorm should copy data itself.
                            A.tileGetForWriting(i, j, A.tileDevice(i, j), slate::LayoutConvert::ColMajor);
                        }

                        real_t A_norm = slate::norm(norm, A, {
                            {slate::Option::Target, target}
                        });

                        real_t A_norm_ref = scalapack_plansy(
                                                norm2str(norm), uplo2str(A.uplo()),
                                                n, &A_tst[0], ione, ione, descA_tst, &worklansy[0]);

                        // difference between norms
                        real_t error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
                        if (norm == slate::Norm::One || norm == slate::Norm::Inf) {
                            error /= sqrt(n);
                        }
                        else if (norm == slate::Norm::Fro) {
                            error /= sqrt(n*n);
                        }

                        // Allow for difference, except max norm in real should be exact.
                        real_t eps = std::numeric_limits<real_t>::epsilon();
                        real_t tol;
                        if (norm == slate::Norm::Max && ! slate::is_complex<scalar_t>::value)
                            tol = 0;
                        else
                            tol = 3*eps;

                        if (mpi_rank == 0) {
                            // if peak is nan, expect A_norm to be nan.
                            bool okay = (std::isnan(real(peak))
                                         ? std::isnan(A_norm)
                                         : error <= tol);
                            params.okay() = params.okay() && okay;
                            if (verbose || ! okay) {
                                printf("i %5lld, j %5lld, ii %3lld, jj %3lld, peak %15.8e, norm %15.8e, ref %15.8e, error %9.2e, %s\n",
                                       llong( i ), llong( j ), llong( ii ), llong( jj ),
                                       real(peak), A_norm, A_norm_ref, error,
                                       (okay ? "pass" : "failed"));
                            }
                        }

                        if (A.tileIsLocal(i, j)) {
                            A.tileGetForWriting(i, j, slate::LayoutConvert::ColMajor);
                            auto T = A(i, j);
                            T.at(ii, jj) = save;
                            A_tst[ ilocal + jlocal*lldA ] = save;
                            // todo: this move shouldn't be required -- the trnorm should copy data itself.
                            A.tileGetForWriting(i, j, A.tileDevice(i, j), slate::LayoutConvert::ColMajor);
                        }
                    }
                }
            }
        }
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_synorm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_synorm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_synorm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_synorm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_synorm_work<std::complex<double>> (params, run);
            break;
    }
}
