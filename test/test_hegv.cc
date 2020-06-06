#include "slate/slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_hegv_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using llong = long long;

    // get & mark input values
    slate::Job jobz = params.jobz();
    slate::Uplo uplo = params.uplo();
    int64_t itype = params.itype();
    int64_t n = params.dim.n();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t panel_threads = params.panel_threads();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    blas_int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // todo:  relax these assumptions
    //        required by he2hb
    slate_assert(p == q); // Requires a square processing grid.
    slate_assert(uplo == slate::Uplo::Lower);  // only lower for now (he2hb).
    // todo: vector
    slate_assert(jobz == lapack::Job::NoVec);  // only NoVec for now.

    params.time();
    params.ref_time();
    params.error2();

    if (! run)
        return;

    // Local values
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const blas_int izero = 0;

    // initialize BLACS
    blas_int iam = 0, nprocs = 1, ictxt;
    blas_int nprow, npcol, myrow, mycol;
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // variables for scalapack wrapped routines
    int64_t iseed = 1;
    int info;

    // figure out local size, allocate, create descriptor, initialize
    // matrix A (local input/local output), n-by-n, Hermitian
    int64_t mlocA = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    blas_int descA_tst[9];
    scalapack_descinit(descA_tst, n, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst_vec(lldA*nlocA);
    scalapack_pplghe(&A_tst_vec[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix B (local input/local output), n-by-n, Hermitian
    int64_t mlocB = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocB = scalapack_numroc(n, nb, mycol, izero, npcol);
    blas_int descB_tst[9];
    scalapack_descinit(descB_tst, n, n, nb, nb, izero, izero, ictxt, mlocB, &info);
    slate_assert(info == 0);
    int64_t lldB = (int64_t)descB_tst[8];
    std::vector<scalar_t> B_tst_vec(lldB*nlocB);
    scalapack_pplghe(&B_tst_vec[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed + 2);

    // matrix W (global output), W(n), gets eigenvalues in decending order
    std::vector<real_t> W_tst_vec(n);

    // matrix Z (local output), n-by-n , gets orthonormal eigenvectors corresponding to W
    int64_t mlocZ = scalapack_numroc(n, nb, myrow, izero, nprow);
    int64_t nlocZ = scalapack_numroc(n, nb, mycol, izero, npcol);
    blas_int descZ_tst[9];
    scalapack_descinit(descZ_tst, n, n, nb, nb, izero, izero, ictxt, mlocZ, &info);
    slate_assert(info == 0);
    int64_t lldZ = (int64_t)descZ_tst[8];
    std::vector<scalar_t> Z_tst_vec(lldZ * nlocZ);
    scalapack_pplrnt(&Z_tst_vec[0], n, n, nb, nb, myrow, mycol, nprow, npcol, mlocZ, iseed + 3);

    // Initialize SLATE data structures
    slate::HermitianMatrix<scalar_t> A;
    slate::HermitianMatrix<scalar_t> B;
    std::vector<real_t> W_vec;
    slate::Matrix<scalar_t> Z;

    // Copy data from ScaLAPACK as needed
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);

        A = slate::HermitianMatrix<scalar_t>(uplo, n, nb, nprow, npcol, mpi_comm);
        A.insertLocalTiles(origin_target);
        copy(&A_tst_vec[0], descA_tst, A);

        B = slate::HermitianMatrix<scalar_t>(uplo, n, nb, nprow, npcol, mpi_comm);
        B.insertLocalTiles(origin_target);
        copy(&B_tst_vec[0], descB_tst, B);

        W_vec = W_tst_vec;

        Z = slate::Matrix<scalar_t>(n, n, nb, nprow, npcol, mpi_comm);
        Z.insertLocalTiles(origin_target);
        copy(&Z_tst_vec[0], descZ_tst, Z); // Z is output, so this copy is not needed
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &A_tst_vec[0], lldA, nb, nprow, npcol, mpi_comm);
        B = slate::HermitianMatrix<scalar_t>::fromScaLAPACK(uplo, n, &B_tst_vec[0], lldB, nb, nprow, npcol, mpi_comm);
        W_vec = W_tst_vec;
        Z = slate::Matrix<scalar_t>::fromScaLAPACK(n, n, &Z_tst_vec[0], lldZ, nb, nprow, npcol, mpi_comm);
    }

    if (verbose >= 1) {
        printf("%% A   %6lld-by-%6lld\n", llong(A.m()), llong(A.n()));
        printf("%% B   %6lld-by-%6lld\n", llong(B.m()), llong(B.n()));
        printf("%% Z   %6lld-by-%6lld\n", llong(Z.m()), llong(Z.n()));
    }

    if (verbose >= 2) {
        print_matrix("A", A);
        print_matrix("B", B);
        print_matrix("Z", Z);
    }

    std::vector<scalar_t> A_ref_vec, B_ref_vec, Z_ref_vec;
    std::vector<real_t> W_ref_vec;
    if (ref || check) {
        A_ref_vec = A_tst_vec;
        B_ref_vec = B_tst_vec;
        W_ref_vec = W_tst_vec;
        Z_ref_vec = Z_tst_vec;
    }

    slate::HermitianMatrix<scalar_t> A_orig;
    slate::HermitianMatrix<scalar_t> B_orig;
    if (check) {
        A_orig = A.emptyLike();
        A_orig.insertLocalTiles();
        copy(A, A_orig);
        B_orig = B.emptyLike();
        B_orig.insertLocalTiles();
        copy(B, B_orig);
    }

    slate::Options const opts = {
        {slate::Option::Lookahead,       lookahead},
        {slate::Option::Target,          target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking,   ib}
    };

    if (! ref_only) {

        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        { slate::trace::Block trace_block("MPI_Barrier");  MPI_Barrier(mpi_comm); }
        double time = testsweeper::get_wtime();

        //==================================================
        // Run SLATE test.
        //==================================================
        if (jobz == slate::Job::NoVec) {
            slate::eig_vals(itype, A, B, W_vec, opts);
        }
        // else {
            // todo: slate::Job::Vec
        // }

        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::hegv(itype, jobz, A, B, W_vec, Z, opts);

        { slate::trace::Block trace_block("MPI_Barrier");  MPI_Barrier(mpi_comm); }
        double time_tst = testsweeper::get_wtime() - time;
        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
    }

    if (verbose >= 2) {
        print_matrix("A", A);
        print_matrix("B", B);
        print_matrix("Z", Z);
    }

    if (check && jobz == slate::Job::Vec) {
        // do error checks for the operations
        // from ScaLAPACK testing (pzgsepchk.f)
        // where A is a symmetric matrix,
        // B is symmetric positive definite,
        // Z is orthogonal containing eigenvectors
        // and D is diagonal containing eigenvalues
        // One of the following test ratios is computed:
        // itype = 1: R_norm = | A Z - B Z D | / ( |A| |Z| n ulp )
        // itype = 2: R_norm = | A B Z - Z D | / ( |A| |Z| n ulp )
        // itype = 3: R_norm = | B A Z - Z D | / ( |A| |Z| n ulp )

        // create C as a empty allocated matrix
        slate::Matrix<scalar_t> C = Z.emptyLike();
        C.insertLocalTiles();

        // calculate some norms
        real_t A_norm = slate::norm(slate::Norm::One, A_orig);
        real_t Z_norm = slate::norm(slate::Norm::One, Z);
        real_t R_norm = 0;
        scalar_t zero = 0.0, one = 1.0;

        if (itype == 1) {
            // C = AZ + 0*C = AZ
            slate::hemm(slate::Side::Left, one, A_orig, Z, zero, C, opts);
            // Z = ZD
            // todo: Does the Z matrix need to be forced back to the CPU if it is not there?
            int64_t joff = 0;
            for (int64_t j = 0; j < Z.nt(); ++j) {
                int64_t ioff = 0;
                for (int64_t i = 0; i < Z.mt(); ++i) {
                    if (Z.tileIsLocal(i, j)) {
                        auto T = Z.at(i, j);
                        for (int jj = 0; jj < T.nb(); ++jj)
                            for (int ii = 0; ii < T.mb(); ++ii)
                                T.at(ii, jj) *= W_vec[ jj + joff ];
                    }
                    ioff += Z.tileMb(i);
                }
                joff += Z.tileNb(j);
            }
            // C = C - BZ  (i.e. AZ - BZD)
            slate::hemm(slate::Side::Left, one, B_orig, Z, -one, C, opts);
            // R_norm = | A Z - B Z D | / ( |A| |Z| n )
            R_norm = slate::norm(slate::Norm::One, C) / A_norm / Z_norm / n;
        }
        else if (itype == 2) {
            // C = Bz + 0*C = AZ
            slate::hemm(slate::Side::Left, one, B_orig, Z, zero, C, opts);
            // Z = ZD
            int64_t joff = 0;
            for (int64_t j = 0; j < Z.nt(); ++j) {
                int64_t ioff = 0;
                for (int64_t i = 0; i < Z.mt(); ++i) {
                    if (Z.tileIsLocal(i, j)) {
                        auto T = Z.at(i, j);
                        for (int jj = 0; jj < T.nb(); ++jj)
                            for (int ii = 0; ii < T.mb(); ++ii)
                                T.at(ii, jj) *= W_vec[ jj + joff ];
                    }
                    ioff += Z.tileMb(i);
                }
                joff += Z.tileNb(j);
            }
            // Z = AC - Z
            slate::hemm(slate::Side::Left, one, A_orig, C, -one, Z, opts);
            // R_norm = | A B Z - Z D | / ( |A| |Z| n )
            R_norm = slate::norm(slate::Norm::One, Z) / A_norm / Z_norm / n;
        }
        else if (itype == 3) {
            // C = AZ + 0*C = AZ
            slate::hemm(slate::Side::Left, one, A_orig, Z, zero, C, opts);
            // Z = ZD
            int64_t joff = 0;
            for (int64_t j = 0; j < Z.nt(); ++j) {
                int64_t ioff = 0;
                for (int64_t i = 0; i < Z.mt(); ++i) {
                    if (Z.tileIsLocal(i, j)) {
                        auto T = Z.at(i, j);
                        for (int jj = 0; jj < T.nb(); ++jj)
                            for (int ii = 0; ii < T.mb(); ++ii)
                                T.at(ii, jj) *= W_vec[ jj + joff ];
                    }
                    ioff += Z.tileMb(i);
                }
                joff += Z.tileNb(j);
            }
            // Z = BC - Z   = ( BAZ - ZD )
            slate::hemm(slate::Side::Left, one, B_orig, C, -one, Z, opts);
            // R_norm = | B A Z - Z D | / ( |A| |Z| n )
            R_norm = slate::norm(slate::Norm::One, Z) / A_norm / Z_norm / n;

        }
        params.error() = R_norm;
        real_t tol = params.tol() * std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= tol);
    }

    if (ref || check) {
        // Run reference routine from ScaLAPACK

        // set num threads appropriately for parallel BLAS if possible
        int omp_num_threads = 1;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        const char* range = "A";
        int64_t ia=1, ja=1, ib=1, jb=1, iz=1, jz=1;
        int64_t vl=0, vu=0, il=0, iu=0;
        real_t abstol=0;
        int64_t m=0, nz=0;
        real_t orfac=0;

        // query for workspace size
        int64_t info_tst = 0;
        int64_t lwork = -1, lrwork = -1, liwork=-1;
        std::vector<scalar_t> work(1);
        std::vector<real_t> rwork(1);
        std::vector<int> iwork(1);
        std::vector<int> ifail(n);
        std::vector<int> iclustr(2*p*q);
        std::vector<real_t> gap(p*q);
        scalapack_phegvx(itype, job2str(jobz), range, uplo2str(uplo), n,
                         &A_ref_vec[0], ia, ja, descA_tst,
                         &B_ref_vec[0], ib, jb, descB_tst,
                         vl, vu, il, iu, abstol, &m, &nz, &W_ref_vec[0], orfac,
                         &Z_ref_vec[0], iz, jz, descZ_tst,
                         &work[0], lwork, &rwork[0], lrwork, &iwork[0], liwork,
                         &ifail[0], &iclustr[0], &gap[0], &info_tst);

        // resize workspace based on query for workspace sizes
        slate_assert(info_tst == 0);
        lwork = int64_t(real(work[0]));
        work.resize(lwork);
        // The lrwork, rwork parameters are only valid for complex
        if (slate::is_complex<scalar_t>::value) {
            lrwork = int64_t(real(rwork[0]));
            rwork.resize(lrwork);
        }
        liwork = int64_t(iwork[0]);
        iwork.resize(liwork);

        // Run ScaLAPACK reference routine.
        MPI_Barrier(mpi_comm);
        double time = testsweeper::get_wtime();

        scalapack_phegvx(itype, job2str(jobz), range, uplo2str(uplo), n,
                         &A_ref_vec[0], ia, ja, descA_tst,
                         &B_ref_vec[0], ib, jb, descB_tst,
                         vl, vu, il, iu, abstol, &m, &nz, &W_ref_vec[0], orfac,
                         &Z_ref_vec[0], iz, jz, descZ_tst,
                         &work[0], lwork, &rwork[0], lrwork, &iwork[0], liwork,
                         &ifail[0], &iclustr[0], &gap[0], &info_tst);

        slate_assert(info_tst == 0);
        MPI_Barrier(mpi_comm);
        double time_ref = testsweeper::get_wtime() - time;

        params.ref_time() = time_ref;

        // Reset omp thread number
        slate_set_num_blas_threads(saved_num_threads);

        if (! ref_only) {
            // Reference Scalapack was run, check reference eigenvalues
            // Perform a local operation to get differences W_vec = W_vec - W_ref
            blas::axpy(W_vec.size(), -1.0, &W_ref_vec[0], 1, &W_vec[0], 1);
            // Relative forward error: || W_ref - W_tst || / || W_ref ||
            params.error2() = lapack::lange(slate::Norm::One, W_vec.size(), 1, &W_vec[0], 1)
                / lapack::lange(slate::Norm::One, W_ref_vec.size(), 1, &W_ref_vec[0], 1);
            real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();
            params.okay() = (params.error2() <= tol);
        }
    }

    Cblacs_gridexit(ictxt);
}

// -----------------------------------------------------------------------------
void test_hegv(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hegv_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_hegv_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_hegv_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hegv_work<std::complex<double>> (params, run);
            break;
    }
}
