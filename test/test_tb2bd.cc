
#include "slate/slate.hh"
#include "blas.hh"
#include "test.hh"
#include "print_matrix.hh"
#include "scalapack_support_routines.hh"
#include "internal/internal.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_tb2bd_work(
    Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using blas::imag;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t ku = nb;  // upper band; for now use ku == nb.
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
    params.error.name("S - Sref\nerror");
    params.error2.name("off-diag\nerror");

    if (! run)
        return;

    int mpi_rank, mpi_size;
    slate_mpi_call(
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    int64_t lda = m;
    int64_t seed[] = {0, 1, 2, 3};
    int64_t min_mn = std::min(m, n);

    std::vector<scalar_t> A1( lda*n );
    lapack::larnv(1, seed, A1.size(), &A1[0]);
    std::vector<scalar_t> A3( lda*n );

    // zero outside the upper band
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            if (j > i+ku || j < i)
                A1[i + j*lda] = 0;
        }
        // Diagonal from ge2tb is real.
        A1[j + j*lda] = real( A1[j + j*lda] );
    }

    if (verbose && mpi_rank == 0)
        print_matrix( "A1", m, n, &A1[0], lda );

    std::vector<real_t> S1(min_mn);
    if (check) {
        //==================================================
        // For checking results, compute SVD of original matrix A.
        //==================================================
        if (mpi_rank == 0) {
            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            std::vector<scalar_t> A2 = A1;
            std::vector<scalar_t> U ( 1 );  // ( lda*n );  // U, VT not needed for NoVec
            std::vector<scalar_t> VT( 1 );  // ( lda*n );
            lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec,
                          m, n, &A2[0], lda, &S1[0], &U[0], lda, &VT[0], lda);

            slate_set_num_blas_threads(saved_num_threads);
        }
    }

    auto Afull = slate::Matrix<scalar_t>::fromLAPACK(
        m, n, &A1[0], lda, nb, p, q, MPI_COMM_WORLD);

    auto Afullrm = slate::Matrix<scalar_t>::fromLAPACK(
        m, n, &A3[0], lda, nb, 1, 1, MPI_COMM_WORLD);
    auto Abandrm = slate::BandMatrix<scalar_t>(ku, ku, Afullrm);
    //auto Aband = slate::BandMatrix<scalar_t>(ku, ku, Afull);
    auto A     = slate::TriangularBandMatrix<scalar_t>( lapack::Uplo::Upper,
                                                        lapack::Diag::NonUnit,
                                                        Abandrm);
    //auto A = slate::TriangularBandMatrix<scalar_t>(
    //                    slate::Uplo::Upper, slate::Diag::NonUnit,
    //                    m, ku, nb, p, q, MPI_COMM_WORLD);
    //slate::internal::copyge2tb(Afull, A);
    //print_matrix( "Aband", Aband);

    A.ge2tbGather(Afull);

    // int64_t index = 0; // index in Ad storage
    int64_t jj = 0; // col index
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii = 0; // row index
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j) &&
                ((ii == jj) ||
                 ( ii < jj && (jj - (ii + A.tileMb(i) - 1)) <= (A.bandwidth()+1) ) ) )
            {

                if (i == j) {
                    //lapack::laset(lapack::MatrixType::Lower, A(i, j).mb(), A(i, j).nb(),
                    //      0, 0, A(i, j).data(), A(i, j).stride());
                    if (i > 0) {
                        auto T_ptr = A.tileInsert( i, j-1 );
                        lapack::laset(lapack::MatrixType::General, T_ptr->mb(), T_ptr->nb(),
                              0, 0, T_ptr->data(), T_ptr->stride());
                    }
                }

                if ((j < A.nt()-1) && (i == (j - 1))) {
                    //lapack::laset(lapack::MatrixType::Upper, A(i, j).mb(), A(i, j).nb(),
                    //      0, 0, A(i, j).data(), A(i, j).stride());
                    auto T_ptr = A.tileInsert( i, j+1 );
                    lapack::laset(lapack::MatrixType::General, T_ptr->mb(), T_ptr->nb(),
                          0, 0, T_ptr->data(), T_ptr->stride());
                }
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    //print_matrix("A", A);

    //---------
    // run test
    if (trace)
        slate::trace::Trace::on();
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = testsweeper::get_wtime();

    //==================================================
    // Run SLATE test.
    //==================================================
    if (mpi_rank == 0) {
        slate::tb2bd(A);
    }

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
        // Gather the whole matrix onto rank 0.
        //==================================================
        A.gather(&A1[0], lda);

        if (mpi_rank == 0) {
            if (verbose)
                print_matrix( "A1_out", m, n, &A1[0], lda );

            // Check that updated A is real bidiagonal by finding max value
            // outside bidiagonal, and imaginary parts of bidiagonal.
            // Unclear why this increases modestly with n.
            real_t max_value = 0;
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    auto val = A1[i + j*lda];
                    if (j > i+1 || j < i)
                        max_value = std::max( std::abs(val), max_value );
                    else
                        max_value = std::max( std::abs(imag(val)), max_value );
                }
            }
            params.error2() = max_value / sqrt(n);

            // set MKL num threads appropriately for parallel BLAS
            int omp_num_threads;
            #pragma omp parallel
            { omp_num_threads = omp_get_num_threads(); }
            int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

            // Check that the singular values of updated A
            // match the singular values of the original A.
            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
            std::vector<real_t> S2(min_mn);
            std::vector<real_t> E(min_mn - 1);  // super-diagonal
            scalar_t dummy[1];  // U, VT, C not needed for NoVec

            // Copy diagonal & super-diagonal.
            int64_t D_index = 0;
            int64_t E_index = 0;
            for (int64_t i = 0; i < std::min(A.mt(), A.nt()); ++i) {
                // Copy 1 element from super-diagonal tile to E.
                if (i > 0) {
                    auto T = A(i-1, i);
                    E[E_index] = real( T(T.mb()-1, 0) );
                    E_index += 1;
                }

                // Copy main diagonal to S2.
                auto T = A(i, i);
                auto len = std::min(T.mb(), T.nb());
                for (int64_t j = 0; j < len; ++j) {
                    S2[D_index + j] = real( T(j, j) );
                }
                D_index += len;

                // Copy super-diagonal to E.
                for (int64_t j = 0; j < len-1; ++j) {
                    E[E_index + j] = real( T(j, j+1) );
                }
                E_index += len-1;
            }
            if (verbose) {
                print_matrix("D", min_mn, 1, &S2[0], min_mn);
                print_matrix("E", min_mn-1, 1, &E[0], min_mn-1);
            }

            lapack::bdsqr(lapack::Uplo::Upper, min_mn, 0, 0, 0,
                          &S2[0], &E[0], dummy, 1, dummy, 1, dummy, 1);
            slate_set_num_blas_threads(saved_num_threads);

            if (verbose) {
                printf( "%9s  %9s\n", "S1", "S2" );
                for (int64_t i = 0; i < std::min(m, n); ++i) {
                    if (i < 20 || i > std::min(m, n)-20) {
                        bool okay = std::abs( S1[i] - S2[i] ) < tol;
                        printf( "%9.6f  %9.6f%s\n",
                                S1[i], S2[i], (okay ? "" : " !!") );
                    }
                }
                printf( "\n" );
            }

            // Relative forward error: || S - Sref || / || Sref ||.
            blas::axpy(S2.size(), -1.0, &S1[0], 1, &S2[0], 1);
            params.error() = blas::nrm2(S2.size(), &S2[0], 1)
                           / blas::nrm2(S1.size(), &S1[0], 1);
            params.okay() = (params.error() <= tol && params.error2() <= tol);
        }
    }
}

// -----------------------------------------------------------------------------
void test_tb2bd(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tb2bd_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_tb2bd_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_tb2bd_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tb2bd_work<std::complex<double>> (params, run);
            break;
    }
}
