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
// The function implementation is in test_ge2tb.cc file.
int64_t localRowsCols(int64_t n, int64_t nb, int iproc, int mpi_size);

//------------------------------------------------------------------------------
// Zero out B, then copy band matrix B from A.
// B is stored as a non-symmetric matrix, so we can apply Q from left
// and right separately.
template < typename scalar_t >
void he2gb(slate::HermitianMatrix< scalar_t > A, slate::Matrix< scalar_t > B)
{
    // It must be defined here to avoid having numerical error with complex
    // numbers when calling conj();
    using blas::conj;
    const int64_t nt = A.nt();
    const scalar_t zero = 0;
    set(zero, B);
    for (int64_t i = 0; i < nt; ++i) {
        if (B.tileIsLocal(i, i)) {
            // diagonal tile
            auto Aii = A(i, i);
            auto Bii = B(i, i);
            Aii.uplo(slate::Uplo::Lower);
            Bii.uplo(slate::Uplo::Lower);
            tzcopy(Aii, Bii);
            // Symmetrize the tile.
            for (int64_t jj = 0; jj < Bii.nb(); ++jj)
                for (int64_t ii = jj; ii < Bii.mb(); ++ii)
                    Bii.at(jj, ii) = conj(Bii(ii, jj));
        }
        if (i+1 < nt && B.tileIsLocal(i+1, i)) {
            // sub-diagonal tile
            auto Ai1i = A(i+1, i);
            auto Bi1i = B(i+1, i);
            Ai1i.uplo(slate::Uplo::Upper);
            Bi1i.uplo(slate::Uplo::Upper);
            tzcopy(Ai1i, Bi1i);
            if (! B.tileIsLocal(i, i+1))
                B.tileSend(i+1, i, B.tileRank(i, i+1));
        }
        if (i+1 < nt && B.tileIsLocal(i, i+1)) {
            if (! B.tileIsLocal(i+1, i)) {
                // Remote copy-transpose B(i+1, i) => B(i, i+1);
                // assumes square tiles!
                B.tileRecv(i, i+1, B.tileRank(i+1, i), slate::Layout::ColMajor);
                deepConjTranspose(B(i, i+1));
            }
            else {
                // Local copy-transpose B(i+1, i) => B(i, i+1).
                deepConjTranspose(B(i+1, i), B(i, i+1));
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_he2hb_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    // using blas::real;
    // using blas::conj;
    // using llong = long long;

    // get & mark input values
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    origin = slate::Origin::ScaLAPACK;  // todo: for now

    // mark non-standard output values
    params.time();
    //params.gflops();

    if (! run)
        return;

    // Requires a square processing grid.
    assert(p == q);
    assert(uplo == slate::Uplo::Lower);  // only lower for now.

    // Local values
    const scalar_t one = 1;

    // MPI variables
    int mpi_rank, mpi_size;
    slate_mpi_call(
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert(p*q <= mpi_size);

    int myrow = mpi_rank % p;
    int mycol = mpi_rank / p;

    // matrix A, figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(n, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lldA   = mlocal;
    std::vector<scalar_t> A_data(lldA*nlocal);
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());

    slate::HermitianMatrix<scalar_t> A;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, p, q, MPI_COMM_WORLD);
        A.insertLocalTiles(origin2target(origin));
        // todo: need ScaLAPACK descriptor for copy. hmpf!
        //copy(A_data.data(), descA_tst, A);
        assert(false);
    }
    else {
        // Create SLATE matrices from the ScaLAPACK layouts.
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
            uplo, n, A_data.data(), lldA, nb, p, q, MPI_COMM_WORLD);
    }
    slate::TriangularFactors<scalar_t> T;

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Copy test data for check.
    slate::HermitianMatrix<scalar_t> A_ref(uplo, n, nb, p, q, MPI_COMM_WORLD);
    A_ref.insertLocalTiles();
    slate::copy(A, A_ref);

    // todo
    //double gflop = lapack::Gflop<scalar_t>::he2hb(n, n);

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
    slate::he2hb(A, T, {
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, ib}
    });

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = testsweeper::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    params.time() = time_tst;
    //params.gflops() = gflop / time_tst;

    if (verbose > 1) {
        print_matrix("A_factored", A);
        print_matrix("Tlocal",  T[0]);
        print_matrix("Treduce", T[1]);
    }

    if (check) {
        //==================================================
        // Test results by checking backwards error
        //
        //      || QBQ^H - A ||_1
        //     ------------------- < tol * epsilon
        //      || A ||_1 * n
        //
        //==================================================

        // Norm of original matrix: || A ||_1
        real_t A_norm = slate::norm(slate::Norm::One, A_ref);

        slate::Matrix< scalar_t > B(n, n, nb, p, q, MPI_COMM_WORLD);
        B.insertLocalTiles();
        he2gb(A, B);
        if (verbose > 1) {
            print_matrix("B", B);
        }

        // Form QB, where Q's representation is in lower part of A and T.
        // auto Asub = slate::Matrix<scalar_t>( A, 1, nt-1, 0, nt-1 );
        // auto Bsub = B.sub(1, nt-1, 0, nt-1);
        // slate::TriangularFactors<scalar_t> Tsub = {
        //     T[0].sub(1, nt-1, 0, nt-1),
        //     T[1].sub(1, nt-1, 0, nt-1)
        // };
        // slate::unmqr(slate::Side::Left, slate::Op::NoTrans, Asub, Tsub, Bsub,
        //              {{slate::Option::Target, target}});

        slate::unmtr_he2hb(slate::Side::Left,
                           slate::Op::NoTrans, A, T, B,
                           {{slate::Option::Target, target}});
        if (verbose > 1) {
            print_matrix("Q^H B", B);
        }

        // Form (QB)Q^H
        // Bsub = B.sub(0, nt-1, 1, nt-1);
        // slate::unmqr(slate::Side::Right, slate::Op::ConjTrans, Asub, Tsub, Bsub,
        //              {{slate::Option::Target, target}});

        slate::unmtr_he2hb(slate::Side::Right,
                           slate::Op::ConjTrans, A, T, B,
                           {{slate::Option::Target, target}});
        if (verbose > 1) {
            print_matrix("Q^H B Q", B);
        }

        // Form QBQ^H - A, where A is in A_ref.
        // todo: slate::tradd(-one, TriangularMatrix(A_ref),
        //                     one, TriangularMatrix(B));
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.nt(); ++i) {
                if (A_ref.tileIsLocal(i, j)) {
                    auto Aij = A_ref(i, j);
                    auto Bij = B(i, j);
                    // if i == j, Aij was Lower; set it to General for axpy.
                    Aij.uplo(slate::Uplo::General);
                    axpy(-one, Aij, Bij);
                }
            }
        }
        slate::HermitianMatrix<scalar_t> B_he(uplo, B);
        if (verbose > 1) {
            print_matrix("QBQ^H - A", B_he);
        }

        // Norm of backwards error: || QBQ^H - A ||_1
        params.error() = slate::norm(slate::Norm::One, B_he) / (n * A_norm);
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
        params.okay() = (params.error() <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_he2hb(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_he2hb_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_he2hb_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_he2hb_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_he2hb_work<std::complex<double>> (params, run);
            break;
    }
}
