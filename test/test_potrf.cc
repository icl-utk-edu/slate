#include "slate.hh"
#include "slate_Debug.hh"
#include "test.hh"
#include "error.hh"
#include "lapack_flops.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SLATE_WITH_MPI
#include <mpi.h>
#else
#include "slate_NoMpi.hh"
#endif

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"

// ------------------------------------------------------------------------------
template < typename scalar_t > void test_potrf_work(Params & params, bool run)
{
    using real_t = blas::real_type < scalar_t >;

    // get & mark input values
    slate::Uplo uplo = params.uplo.value();
    // int64_t align = params.align.value();
    int64_t lookahead = params.lookahead.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t s = params.nrhs.value();
    int64_t nb = params.nb.value();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();
    params.error2.value();

    if(!run)
        return;

    // Get ScaLAPACK compatible versions of some of the parameters
    int s_ = s, nb_ = nb, n_ = n;
    const char *uplo_str = blas::uplo2str(uplo);

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info, mloc, nloc, sloc;
    int descA_tst[9], descA_ref[9];
    int iam = 0;
    int nprocs = 1;
    static int i0 = 0, i1 = 1;
    static scalar_t m1 = -1e0, p1 = 1e0;

    Cblacs_pinfo(&iam, &nprocs);
    assert(p * q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Row", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    mloc = scalapack_numroc(&n_, &nb_, &myrow, &i0, &nprow);
    nloc = scalapack_numroc(&n_, &nb_, &mycol, &i0, &npcol);
    sloc = scalapack_numroc(&s_, &nb_, &mycol, &i0, &npcol);

    // typedef long long lld;
    size_t size_A = (size_t) (mloc * nloc);
    std::vector < scalar_t > A_tst(size_A);
    std::vector < scalar_t > A_ref(size_A);

    // Initialize the matrix
    int iseed = iam;
    scalapack_descinit(descA_tst, &n_, &n_, &nb_, &nb_, &i0, &i0, &ictxt, &mloc, &info);
    assert(0 == info);
    scalapack_pplghe(&A_tst[0], n_, n_, nb_, nb_, myrow, mycol, nprow, npcol, mloc, iseed);

    // Create SLATE matrix from the ScaLAPACK layouts
    int64_t local_lda = descA_tst[8];
    auto A = slate::HermitianMatrix < scalar_t >::fromScaLAPACK(uplo, n, &A_tst[0], local_lda, nb, nprow, npcol, MPI_COMM_WORLD);

    // If check is required, save A in A_ref and create a descriptor for it
    if(params.check.value() == 'y') {
        scalapack_descinit(descA_ref, &n_, &n_, &nb_, &nb_, &i0, &i0, &ictxt, &mloc, &info);
        assert(0 == info);
        A_ref = A_tst;
    }
    // Call the routine using ScaLAPACK layout
    MPI_Barrier(MPI_COMM_WORLD);
    double time = libtest::get_wtime();
    slate::potrf < slate::Target::HostTask > (A, lookahead);
    // scalapack_ppotrf( uplo_str, &n, &A_tst[0], &i1, &i1, descA_tst, &info ); assert( 0 == info );
    MPI_Barrier(MPI_COMM_WORLD);
    double time_tst = libtest::get_wtime() - time;

    // Compute and save timing/performance
    double gflop = lapack::Gflop < scalar_t >::potrf(n);
    params.time.value() = time_tst;
    params.gflops.value() = gflop / time_tst;

    real_t eps = std::numeric_limits < real_t >::epsilon();
    real_t tol = params.tol.value();

    if(params.check.value() == 'y') {
        // A numerical check of the value
        // This should not alter A_ref

        // create B and copy it to X
        int descB_ref[9];
        scalapack_descinit(descB_ref, &n_, &s_, &nb_, &nb_, &i0, &i0, &ictxt, &mloc, &info);
        assert(0 == info);
        size_t size_B = (size_t) (mloc * sloc);
        std::vector < scalar_t > B_ref(size_B);
        std::vector < scalar_t > X_ref(size_B);

        // Initialize B
        scalapack_pplrnt(&B_ref[0], n, s, nb, nb, myrow, mycol, nprow, npcol, mloc, iseed + 1);
        // Copy B to X (use local copies instead of globally via ScaLAPACK pdlacpy)
        X_ref = B_ref;

        // Get norms of A and B
        size_t ldw = nb * ceil(ceil(mloc / (double) nb) / (ilcm_(&nprow, &npcol) / nprow));
        std::vector < scalar_t > worklansy(2 * nloc + mloc + ldw);
        real_t Anorm = scalapack_plansy("I", uplo_str, &n_, &A_ref[0], &i1, &i1, descA_ref, &worklansy[0]);
        real_t Bnorm = scalapack_plange("I", &n_, &s_, &B_ref[0], &i1, &i1, descB_ref, &worklansy[0]);

        // Solve for X using the A_tst factorization
        scalapack_ppotrs(uplo_str, &n_, &s_, &A_tst[0], &i1, &i1, descA_ref, &X_ref[0], &i1, &i1, descB_ref, &info);
        assert(0 == info);

        // Compute B(diff) = B - A_ref * X
        scalapack_psymm(uplo_str, uplo_str, &n_, &s_, &m1, &A_ref[0], &i1, &i1, descA_tst, &X_ref[0], &i1, &i1, descB_ref, &p1, &B_ref[0], &i1, &i1, descB_ref);

        // Norms of X and B(diff)
        std::vector < scalar_t > worklange(mloc);
        real_t Xnorm = scalapack_plange("I", &n_, &s_, &X_ref[0], &i1, &i1, descB_ref, &worklange[0]);
        real_t Rnorm = scalapack_plange("I", &n_, &s_, &B_ref[0], &i1, &i1, descB_ref, &worklange[0]);
        real_t resid = Rnorm / ((Bnorm + Anorm * Xnorm) * fmax(n, n));
        params.error.value() = resid;
    }

    if(params.ref.value() == 'y') {
        // A comparison with a reference routine from ScaLAPACK
        // This expects to get the original (clean) A_ref and will alter A_ref

        // Run the reference routine on A_ref
        MPI_Barrier(MPI_COMM_WORLD);
        double time = libtest::get_wtime();
        scalapack_ppotrf(uplo_str, &n_, &A_ref[0], &i1, &i1, descA_ref, &info);
        assert(0 == info);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

        // norm(A_ref)
        size_t ldw = nb * ceil(ceil(mloc / (double) nb) / (scalapack_ilcm(&nprow, &npcol) / nprow));
        std::vector < scalar_t > worklansy(2 * nloc + mloc + ldw);
        real_t A_ref_norm = scalapack_plansy("I", uplo_str, &n_, &A_ref[0], &i1, &i1, descA_ref, &worklansy[0]);

        // Local operation: error = A_ref = A_ref - A_tst
        for(size_t i = 0; i < A_ref.size(); i++)
            A_ref[i] = A_ref[i] - A_tst[i];

        // error = norm(error)
        real_t error_norm = scalapack_plansy("I", uplo_str, &n_, &A_ref[0], &i1, &i1, descA_ref, &worklansy[0]);

        // error = error / norm;
        if(error_norm != 0)
            error_norm /= A_ref_norm;

        params.ref_time.value() = time_ref;
        params.ref_gflops.value() = gflop / time_ref;
        params.error2.value() = error_norm;
    }

    params.okay.value() = ((params.error.value() <= tol) && (params.error2.value() <= 3 * eps));

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_potrf(Params & params, bool run)
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_potrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_potrf_work < double >(params, run);
            break;

        case libtest::DataType::SingleComplex:
            throw std::exception();     // test_potrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            throw std::exception();     // test_potrf_work< std::complex<double> >( params, run );
            break;
    }
}
