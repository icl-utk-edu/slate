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
void test_genorm_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;
    using slate::ceildiv;
    using llong = long long;

    // get & mark input values
    slate::Norm norm = params.norm();
    slate::NormScope scope = params.scope();
    slate::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t p = params.p();
    int64_t q = params.q();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
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

    //if (verbose > 1) {
    //    print_matrix("A_tst", mlocA, nlocA, &A_tst[0], lldA, p, q, MPI_COMM_WORLD);
    //}

    // todo: work-around to initialize BaseMatrix::num_devices_
    slate::Matrix<scalar_t> A0(m, n, nb, p, q, MPI_COMM_WORLD);

    slate::Matrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);
    }
    else {
        // Create SLATE matrix from the ScaLAPACK layout.
        A = slate::Matrix<scalar_t>::fromScaLAPACK(
                m, n, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    std::vector<real_t> values;
    if (scope == slate::NormScope::Columns) {
        values.resize(A.n());
    }
    else if (scope == slate::NormScope::Rows) {
        values.resize(A.m());
    }

    if (trans == slate::Op::Trans)
        A = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        A = conjTranspose(A);

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

    real_t A_norm = 0;
    if (! ref_only) {

        //==================================================
        // Run SLATE test.
        // Compute || A ||_norm.
        //==================================================

        if (scope == slate::NormScope::Matrix) {
            A_norm = slate::norm(norm, A, {
                {slate::Option::Target, target}
            });
        }
        else if (scope == slate::NormScope::Columns) {
            slate::colNorms(norm, A, values.data(), {
                {slate::Option::Target, target}
            });
        }
        else if (scope == slate::NormScope::Rows) {
            slate_error("Not implemented yet");
            // slate::rowNorms(norm, A, values.data(), {
            //     {slate::Option::Target, target}
            // });
        }

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
    }

    if (check || ref) {
        // comparison with reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        // allocate work space
        std::vector<real_t> worklange(std::max(mlocA, nlocA));

        // (Sca)LAPACK norms don't support trans; map One <=> Inf norm.
        slate::Norm op_norm = norm;
        if (trans == slate::Op::Trans || trans == slate::Op::ConjTrans) {
            if (norm == slate::Norm::One)
                op_norm = slate::Norm::Inf;
            else if (norm == slate::Norm::Inf)
                op_norm = slate::Norm::One;
        }

        // difference between norms
        real_t error = 0.;
        real_t A_norm_ref = 0;

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        time = testsweeper::get_wtime();
        if (scope == slate::NormScope::Matrix) {
            A_norm_ref = scalapack_plange(
                norm2str(op_norm),
                m, n, &A_tst[0], ione, ione, descA_tst, &worklange[0]);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else if (scope == slate::NormScope::Columns) {
            for (int64_t c = 0; c < n; ++c) {
                int64_t c_1 = c+1;
                A_norm_ref = scalapack_plange(
                    norm2str(norm),
                    m, 1, &A_tst[0], ione, c_1, descA_tst, &worklange[0]);
                MPI_Barrier(MPI_COMM_WORLD);
                error += std::abs(values[c] - A_norm_ref) / A_norm_ref;
            }
        }
        else if (scope == slate::NormScope::Rows) {
            // todo
        }
        double time_ref = testsweeper::get_wtime() - time;

        //A_norm_ref = lapack::lange(
        //    op_norm,
        //    m, n, &A_tst[0], lldA);

        if (scope == slate::NormScope::Matrix) {
            // difference between norms
            error = std::abs(A_norm - A_norm_ref) / A_norm_ref;
            if (op_norm == slate::Norm::One) {
                error /= sqrt(m);
            }
            else if (op_norm == slate::Norm::Inf) {
                error /= sqrt(n);
            }
            else if (op_norm == slate::Norm::Fro) {
                error /= sqrt(m*n);
            }

            if (verbose && mpi_rank == 0) {
                printf("norm %15.8e, ref %15.8e, ref - norm %5.2f, error %9.2e\n",
                       A_norm, A_norm_ref, A_norm_ref - A_norm, error);
            }
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
    if (extended && scope == slate::NormScope::Matrix) {
        // allocate work space
        std::vector<real_t> worklange(std::max(mlocA, nlocA));

        // seed all MPI processes the same
        srand(1234);

        // Test tiles in 2x2 in all 4 corners, and 4 random rows and cols,
        // up to 64 tiles total.
        // Indices may be out-of-bounds if mt or nt is small, so check in loops.
        int64_t mt = A.mt();
        int64_t nt = A.nt();
        std::set<int64_t> i_indices = { 0, 1, mt - 2, mt - 1 };
        std::set<int64_t> j_indices = { 0, 1, nt - 2, nt - 1 };
        for (size_t k = 0; k < 4; ++k) {
            i_indices.insert(rand() % mt);
            j_indices.insert(rand() % nt);
        }
        for (auto j : j_indices) {
            if (j < 0 || j >= nt)
                continue;
            int64_t jb = std::min(n - j*nb, nb);
            slate_assert(jb == A.tileNb(j));

            for (auto i : i_indices) {
                if (i < 0 || i >= mt)
                    continue;
                int64_t ib = std::min(m - i*nb, nb);
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
                        if (ii < 0 || ii >= ib) {
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

                        real_t A_norm_ref = scalapack_plange(
                                                norm2str(norm),
                                                m, n, &A_tst[0], ione, ione, descA_tst, &worklange[0]);

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
void test_genorm(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_genorm_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_genorm_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_genorm_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_genorm_work<std::complex<double>> (params, run);
            break;
    }
}
