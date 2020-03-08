#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"
#include "matrix_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmtr_he2hb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    slate::Side side = params.side();
    slate::Op trans = params.trans();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();
    //params.gflops();

    if (! run)
        return;

    slate_assert(p == q); // Requires a square processing grid.

    //==================================================
    // quick returns:
    //==================================================
    // todo: implement none-ScaLAPACK layout.
    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported.\n");
        return;
    }
    // todo:  he2hb currently doesn't support uplo == upper, needs to figure out
    //        a different solution.
    if (uplo == slate::Uplo::Upper) {
        printf("skipping: currently slate::Uplo::Upper isn't supported.\n");
        return;
    }

    int mpi_rank;
    slate_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    int mpi_size;
    slate_mpi_call(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert( p*q <= mpi_size);

    const int64_t myrow = whoismyrow(mpi_rank, p);
    const int64_t mycol = whoismycol(mpi_rank, p);

    // Matrix A
    // Figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(n, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lldA   = mlocal;
    std::vector<scalar_t> A_data(lldA*nlocal);
    int64_t idist = 3; // normal
    int64_t iseed[4] = {0, myrow, mycol, 3};
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());
    // Create SLATE matrices from the ScaLAPACK layouts.
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Matrix A_ref
    slate::HermitianMatrix<scalar_t> A_ref;
    if (check) {
        if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
            (side == slate::Side::Right && trans != slate::Op::NoTrans)) {
            A_ref = slate::HermitianMatrix<scalar_t>(
                uplo, n, nb, p, q, MPI_COMM_WORLD);

            A_ref.insertLocalTiles();
            slate::copy(A, A_ref);

            if (verbose > 3) {
                print_matrix("A_ref", A_ref);
            }
        }
    }

    // Matrix A_sym
    slate::Matrix<scalar_t> A_sym;
    if ((side == slate::Side::Left  && trans != slate::Op::NoTrans) ||
        (side == slate::Side::Right && trans == slate::Op::NoTrans)) {
        A_sym = slate::Matrix<scalar_t>(n, n, nb, p, q, MPI_COMM_WORLD);

        A_sym.insertLocalTiles();
        he2ge(A, A_sym);

        if (verbose > 1) {
            print_matrix("A_sym", A_sym);
        }
    }

    // Triangular Factors T
    slate::TriangularFactors<scalar_t> T;
    slate::he2hb(A, T, {{slate::Option::Target, target}});

    if (verbose > 2) {
        print_matrix("A_factored", A);
        print_matrix("T_local",    T[0]);
        print_matrix("T_reduce",   T[1]);
    }

    // Matrix B
    slate::Matrix< scalar_t > B(n, n, nb, p, q, MPI_COMM_WORLD);

    B.insertLocalTiles();
    he2gb(A, B);

    if (verbose > 1) {
        print_matrix("B", B);
    }

    // todo
    //double gflop = lapack::Gflop<scalar_t>::unmtr_he2hb(n, n);

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
    if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
        (side == slate::Side::Right && trans != slate::Op::NoTrans)) {
        slate::unmtr_he2hb(side, trans, A, T, B, {
            {slate::Option::Target, target}
        });
    }
    else if ((side == slate::Side::Left  && trans != slate::Op::NoTrans) ||
             (side == slate::Side::Right && trans == slate::Op::NoTrans)) {
        slate::unmtr_he2hb(side, trans, A, T, A_sym, {
           {slate::Option::Target, target}
        });
    }

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time_tst;
    //params.gflops() = gflop / time_tst;

    if (check) {
        const scalar_t negative_one = -1;

        if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
            (side == slate::Side::Right && trans != slate::Op::NoTrans)) {
            //==================================================
            // Test results by checking backwards error
            //
            //      || A - QBQ^H ||_1
            //     ------------------- < tol * epsilon
            //      || A ||_1 * n
            //
            //==================================================

            if (trans == slate::Op::NoTrans) {
                // QB is already computed, we need (QB)Q^H
                // (QB)Q^H
                slate::unmtr_he2hb(slate::Side::Right,
                                   slate::Op::ConjTrans, A, T, B,
                                   {{slate::Option::Target, target}});
            }
            else {
                // BQ^H is already computed, we need QB
                // (QB)Q^H
                slate::unmtr_he2hb(slate::Side::Left,
                                   slate::Op::NoTrans, A, T, B,
                                   {{slate::Option::Target, target}});
            }

            // Norm of original matrix: || A ||_1, where A is in A_ref
            real_t A_ref_norm = slate::norm(slate::Norm::One, A_ref);

            // Form A - QBQ^H, where A is in A_ref.
            for (int64_t j = 0; j < A_ref.nt(); ++j) {
                for (int64_t i = j; i < A_ref.nt(); ++i) {
                    if (A_ref.tileIsLocal(i, j)) {
                        auto A_refij = A_ref(i, j);
                        auto Bij = B(i, j);
                        // if i == j, Aij was Lower; set it to General for axpy.
                        A_refij.uplo(slate::Uplo::General);
                        axpy(negative_one, Bij, A_refij);
                    }
                }
            }

            if (verbose > 1) {
                print_matrix("A - QBQ^H", A_ref);
            }

            // Norm of backwards error: || A - QBQ^H ||_1
            params.error()  = slate::norm(slate::Norm::One, A_ref)
                            / (n * A_ref_norm);
        }
        else if ((side == slate::Side::Left  && trans != slate::Op::NoTrans) ||
                 (side == slate::Side::Right && trans == slate::Op::NoTrans)) {
            //==================================================
            // Test results by checking forward error
            //
            //      || Q^HAQ - B ||_1
            //     ------------------- < tol * epsilon
            //      || B ||_1 * n
            //
            //==================================================

            if (trans == slate::Op::NoTrans) {
                // AQ is already computed, we need Q^HA
                // (Q^HA)Q
                slate::unmtr_he2hb(slate::Side::Left,
                                   slate::Op::ConjTrans, A, T, A_sym,
                                   {{slate::Option::Target, target}});
            }
            else {
                // Q^HA is already computed, we need (Q^HA)Q
                // (Q^HA)Q
                slate::unmtr_he2hb(slate::Side::Right,
                                   slate::Op::NoTrans, A, T, A_sym,
                                   {{slate::Option::Target, target}});
            }

            // Norm of B matrix: || B ||_1
            real_t B_norm = slate::norm(slate::Norm::One, B);

            // Form Q^HAQ - B
            for (int64_t i = 0; i < A_sym.nt(); ++i) {
                for (int64_t j = 0; j < A_sym.mt(); ++j) {
                    if (A_sym.tileIsLocal(i, j)) {
                        axpy(negative_one, B(i, j), A_sym(i, j));
                    }
                }
            }

            if (verbose > 1) {
                print_matrix("Q^HAQ - B", A_sym);
            }

            // Norm of backwards error: || Q^HAQ - B ||_1
            params.error() = slate::norm(slate::Norm::One, A_sym)
                           / (n * B_norm);
        }

        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon() / 2;
        params.okay() = (params.error() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_unmtr_he2hb(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmtr_he2hb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_unmtr_he2hb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmtr_he2hb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmtr_he2hb_work<std::complex<double>> (params, run);
            break;
    }
}
