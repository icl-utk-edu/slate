
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
    auto lower = uplo == slate::Uplo::Lower;
    slate::Diag diag = params.diag();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    // int64_t ku = params.ku();  // upper band
    int64_t nb = params.nb();
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
    params.error2();

    if (! run)
        return;

    int mpi_rank, mpi_size;
    slate_mpi_call(
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    int64_t lda = m, Am = m, An = n;
    // int64_t seed[] = {0, 1, 2, 3};
    int64_t kl = lower ? 1 : 0;
    int64_t ku = lower ? 0 : 1;

    // local values
    const int izero = 0;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9];
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc(Am, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(An, nb, mycol, izero, npcol);
    scalapack_descinit(descA_tst, Am, An, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);
    zeroOutsideBand(&A_tst[0], Am, An, kl, ku, nb, nb, myrow, mycol, nprow, npcol, mlocA);

    // create SLATE matrices from the ScaLAPACK layouts
    auto Aband = BandFromScaLAPACK(
                     Am, An, kl, ku, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);
    auto A = slate::TriangularBandMatrix<scalar_t>(uplo, diag, Aband);

    if (verbose)
        print_matrix( "A", A, 5, 4 );
    // Make A diagonally dominant to be reasonably well conditioned.
    for (int i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, i)) {
            auto T = A(i, i);
            for (int ii = 0; ii < T.mb(); ++ii) {
                T.at(ii, ii) += scalar_t(1.0);
            }
        }
    }
    if (verbose)
        print_matrix( "A", A, 5, 4 );

    std::vector< blas::real_type<scalar_t> > S;
    std::vector< blas::real_type<scalar_t> > E;
    slate::internal::copytb2bd(A, S, E);

    //---------
    // run test
    if (trace)
        slate::trace::Trace::on();
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = libtest::get_wtime();

    //==================================================
    // Run SLATE test.
    //==================================================
    slate::bdsqr<scalar_t>(S, E);

    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    params.time() = libtest::get_wtime() - time;

    if (trace)
        slate::trace::Trace::finish();

    if (check) {
        //==================================================
        // Test results
        // Gather the whole matrix onto rank 0.
        //==================================================
        std::vector<scalar_t> A1( lda*n, 0. );
        A.gather(&A1[0], lda);

        if (mpi_rank == 0) {
            if (verbose)
                print_matrix( "A1", m, m, &A1[0], lda, 5, 4 );

            // Check that updated A is bidiagonal by finding max value outside bidiagonal.
            real_t max_value = 0;
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    if ( (! lower && (j > i+1 || j < i))
                        || (lower && (j > i   || j < i-1)))
                        max_value = std::max( std::abs( A1[i + j*lda] ), max_value );
                }
            }
            params.error2() = max_value;

            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
            std::vector<real_t> D(m);
            std::vector<real_t> E(m-1);
            scalar_t dummy[1];  // U, VT, C not needed for NoVec

            // Copy diagonal & super-diagonal.
            int64_t D_index = 0;
            int64_t E_index = 0;
            D[D_index++] = real( A1[0] );
            for (int64_t i = 1; i < Am; ++i) {
                // Copy super-diagonal to E.
                E[E_index] = real( lower ? A1[i + (i-1)*lda] : A1[i-1 + i*lda] );
                ++E_index;

                // Copy main diagonal to S2.
                D[D_index] = real( A1[i + i*lda] );
                ++D_index;
            }
            if (verbose) {
                print_matrix("D", m, 1, &D[0], 1);
                print_matrix("E", m-1, 1, &E[0], 1);
            }

            lapack::bdsqr(uplo, m, 0, 0, 0,
                          &D[0], &E[0], dummy, 1, dummy, 1, dummy, 1);
            slate_set_num_blas_threads(saved_num_threads);
            if (verbose) {
                printf( "%9s  %9s\n", "D", "S" );
                for (int64_t i = 0; i < m; ++i) {
                    if (i < 20 || i > m-20) {
                        bool okay = std::abs( D[i] - S[i] ) < tol;
                        printf( "%9.6f  %9.6f%s\n",
                                D[i], S[i], (okay ? "" : " !!") );
                    }
                }
                printf( "\n" );
            }


            blas::axpy(S.size(), -1.0, &D[0], 1, &S[0], 1);
            params.error() = blas::nrm2(S.size(), &S[0], 1) / D[0];
            params.okay() = (params.error() <= tol && params.error2() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_bdsqr(Params& params, bool run)
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_bdsqr_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_bdsqr_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_bdsqr_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_bdsqr_work<std::complex<double>> (params, run);
            break;
    }
}
