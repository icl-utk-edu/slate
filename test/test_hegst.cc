#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"
#include "aux/Debug.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
// Similar to ScaLAPACK numroc (number of rows or columns).
// The function implementation is in test_ge2tb.cc file.
int64_t localRowsCols(int64_t n, int64_t nb, int iproc, int mpi_size);

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hegst_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    int64_t itype = params.itype();
    slate::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int verbose = params.verbose();
    slate::Target target = params.target();


    if (! run)
        return;

    // MPI variables
    int mpi_rank, mpi_size;
    slate_mpi_call(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    slate_mpi_call(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    slate_assert(p*q <= mpi_size);

    int myrow = mpi_rank % p;
    int mycol = mpi_rank / p;

    // Figure out local size, allocate, initialize
    int64_t mlocal = localRowsCols(n, nb, myrow, p);
    int64_t nlocal = localRowsCols(n, nb, mycol, q);
    int64_t lld   = mlocal;
    int64_t idist = 3; // normal
    int64_t iseed[4] = { 0, myrow, mycol, 3 };

    // Matrix A
    std::vector<scalar_t> A_data(lld*nlocal);
    lapack::larnv(idist, iseed, A_data.size(), A_data.data());
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, n, A_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    if (verbose > 1) {
        print_matrix("A", A);
    }

    // Matrix A_ref
    std::vector<scalar_t> A_ref_data(lld*nlocal);
    auto A_ref = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, n, A_ref_data.data(), lld, nb, p, q, MPI_COMM_WORLD);
    // Copy test data for check -- keep matrix A for refernce
    slate::copy(A, A_ref);

    if (verbose > 1) {
        print_matrix("A_ref", A_ref);
    }

    // Matrix B
    std::vector<scalar_t> B_data(lld*nlocal);
    lapack::larnv(idist, iseed, B_data.size(), B_data.data());
    auto B = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(
        uplo, n, B_data.data(), lld, nb, p, q, MPI_COMM_WORLD);

    for (int64_t j = 0; j < B.nt(); ++j) {
        for (int64_t i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, j)) {
                auto Bii = B(i, i);
                if (i == j) {
                    for (int64_t jj = 0; jj < Bii.nb(); ++jj) {
                        for (int64_t ii = jj; ii < Bii.mb(); ++ii) {
                            if (ii == jj) {
                                Bii.at(jj, ii) = std::abs(Bii.at(jj, ii)) + n;
                            }
                        }
                    }
                }
            }
        }
    }

    if (verbose > 1) {
        print_matrix("B", B);
    }

    slate::potrf(B, {{slate::Option::Target, target}});

    if (verbose > 1) {
        print_matrix("B_factored", B);
    }

    slate::hegst(itype, A, B, {{slate::Option::Target, target}});

    if (verbose > 1) {
        print_matrix("A_hegst", A);
    }


    {
        real_t A_norm = slate::norm(slate::Norm::One, A_ref);

        int ictxt;
        Cblacs_get(-1, 0, &ictxt);
        Cblacs_gridinit(&ictxt, "Col", p, q);

        const int izero = 0;
        int descA[9], descB[9], info;
        scalapack_descinit(
            descA, n, n, nb, nb, izero, izero, ictxt, mlocal, &info);
        slate_assert(info == 0);
        scalapack_descinit(
            descB, n, n, nb, nb, izero, izero, ictxt, mlocal, &info);
        slate_assert(info == 0);
        const int64_t ione = 1;
        double scale;
        scalapack_phegst(itype, uplo2str(uplo), n,
            &A_ref_data[0], ione, ione, descA,
            &B_data[0],     ione, ione, descB,
            &scale, &info);
        slate_assert(info == 0);

        if (verbose > 1) {
            print_matrix("A_ref_hegst", A_ref);
        }

        // Local operation: error = A_ref - A
        blas::axpy(
            A_ref_data.size(), scalar_t(-1.0),
            A_data.data(), 1,
            A_ref_data.data(), 1);

        params.error() = slate::norm(slate::Norm::One, A_ref) / (n * A_norm);
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon()/2;
        params.okay() = (params.error() <= tol);
    }

}

// -----------------------------------------------------------------------------
void test_hegst(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hegst_work<float>(params, run);
            break;

        case testsweeper::DataType::Double:
            test_hegst_work<double>(params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hegst_work<std::complex<float>>(params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hegst_work<std::complex<double>>(params, run);
            break;
    }
}
