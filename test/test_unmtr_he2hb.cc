#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
// Similar to ScaLAPACK numroc (number of rows or columns).
int64_t localRowsCols(int64_t n, int64_t nb, int iproc, int mpi_size);

//------------------------------------------------------------------------------
template < typename scalar_t >
void he2hbInitMatrixBFromMatrixA(
    slate::HermitianMatrix< scalar_t > A, slate::Matrix< scalar_t >& B);

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_unmtr_he2hb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    // using blas::real;
    // using blas::conj;

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

    //==================================================
    // quick returns:
    //==================================================
    // todo: implement none-ScaLAPACK layout.
    if (origin != slate::Origin::ScaLAPACK) {
        printf("skipping: currently only origin=scalapack is supported.\n");
        return;
    }
    // no formula currently
    if (trans == slate::Op::Trans) {
        printf("skipping: use slate::Op::NoTrans or slate::Op::ConjTrans.\n");
        return;
    }
    // todo:  he2hb currently doesn't support uplo == upper, needs to figure out
    //        a different solution.
    if (uplo == slate::Uplo::Upper) {
        printf("skipping: slate::Uplo::Lower is currently not supported.\n");
        return;
    }

    // Requires a square processing grid.
    assert(p == q);

    // Local values
    const scalar_t one = 1;

    // MPI variables
    int mpi_rank, mpi_size;
    slate_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert(p*q <= mpi_size);

    int myrow = mpi_rank % p;
    int mycol = mpi_rank / p;

    // Matrix A
    // Figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(n, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lldA   = mlocal;
    std::vector<scalar_t> A_data(lldA*nlocal);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());
    // Create SLATE matrices from the ScaLAPACK layouts.
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
                    uplo, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);
    // Copy test data for check.
    slate::HermitianMatrix<scalar_t> A_ref(uplo, n, nb, p, q, MPI_COMM_WORLD);
    A_ref.insertLocalTiles();
    slate::copy(A, A_ref);
    // Output A
    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Triangular Factors T, empty
    slate::TriangularFactors<scalar_t> T;

    slate::he2hb(A, T, {{slate::Option::Target, target}});

    // Output A
    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Output A, and T after he2hb
    if (verbose > 1) {
        print_matrix("A_factored", A);
        print_matrix("T_local",    T[ 0 ]);
        print_matrix("T_reduce",   T[ 1 ]);
    }

    slate::Matrix< scalar_t > B(n, n, nb, p, q, MPI_COMM_WORLD);
    B.insertLocalTiles();
    he2hbInitMatrixBFromMatrixA<scalar_t>(A, B);
    if (verbose > 1) {
        print_matrix("B", B);
    }

    if (check && side == slate::Side::Right) {
        if (trans == slate::Op::ConjTrans) {
            // Compute QB
            slate::unmtr_he2hb(slate::Side::Left, uplo,
                               slate::Op::NoTrans, A, T, B,
                               {{slate::Option::Target, target}});
        }
        else if (trans == slate::Op::NoTrans) {
            printf(
              "skipping: no backward error check for the combination of slate::Side::Right and slate::Op::NoTrans.\n");
            return;
        }
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
    slate::unmtr_he2hb(side, uplo, trans, A, T, B, {
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
    // params.gflops() = gflop / time_tst;

    if (check) {
        if ((side == slate::Side::Left  && trans == slate::Op::NoTrans) ||
            (side == slate::Side::Right && trans == slate::Op::ConjTrans)) {
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
                slate::unmtr_he2hb(slate::Side::Right, uplo,
                                   slate::Op::ConjTrans, A, T, B,
                                   {{slate::Option::Target, target}});
            }

            // Norm of original matrix: || A ||_1
            real_t A_norm = slate::norm(slate::Norm::One, A_ref);

            // Form A - QBQ^H, where A is in A_ref.
            // todo: slate::tradd(one, TriangularMatrix(B),
            //                   -one, TriangularMatrix(A_ref));
            for (int64_t j = 0; j < A.nt(); ++j) {
                for (int64_t i = j; i < A.nt(); ++i) {
                    if (A_ref.tileIsLocal(i, j)) {
                        auto Aij = A_ref(i, j);
                        auto Bij = B(i, j);
                        // if i == j, Aij was Lower; set it to General for axpy.
                        Aij.uplo(slate::Uplo::General);
                        axpy(-one, Bij, Aij);
                    }
                }
            }

            if (verbose > 1) {
                print_matrix("A - QBQ^H", A_ref);
            }

            // Norm of backwards error: || A - QBQ^H ||_1
            params.error() = slate::norm(slate::Norm::One, A_ref) / (n * A_norm);
            real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
            params.okay() = (params.error() <= tol);

        }
        else if ((side == slate::Side::Left  && trans == slate::Op::ConjTrans) ||
                 (side == slate::Side::Right && trans == slate::Op::NoTrans)) {
            //==================================================
            // Test results by checking backwards error
            //
            //      || Q^HAQ - B ||_1
            //     ------------------- < tol * epsilon
            //      || B ||_1 * n
            //
            //==================================================

            if (trans == slate::Op::ConjTrans) {
                printf(
                    "skipping: no backward error check for the combination of slate::Side::Left and slate::Op::ConjTrans.\n");
                return;
            }
        }

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
