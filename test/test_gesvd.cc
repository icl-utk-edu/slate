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
#include <utility>

//------------------------------------------------------------------------------
template <typename scalar_t> void test_gesvd_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using llong = long long;

    // get & mark input values
    lapack::Job jobu = params.jobu();
    lapack::Job jobvt = params.jobvt();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();

    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t lookahead = params.lookahead();
    bool ref_only = params.ref() == 'o';
    bool runtst = (! ref_only);
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Norm norm = params.norm();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    params.time();
    params.ref_time();
    // params.gflops();
    // params.ref_gflops();

    if (! run)
        return;

    // Local values
    int64_t minmn = std::min(m, n);
    const int izero = 0, ione = 1;
    // const scalar_t zero = 0, one = 1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int iam = 0, nprocs = 1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    slate_assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // skip invalid sizes
    if (m <= (p-1)*nb || n <= (q-1)*nb) {
        if (iam == 0) {
            printf("\nskipping: ScaLAPACK requires that all ranks have some rows & columns; "
                   "i.e., m > (p-1)*nb = %lld and n > (q-1)*nb = %lld\n",
                   llong( (p-1)*nb ), llong( (q-1)*nb ) );
        }
        return;
    }

    // skip unsupported 
    if (jobu != lapack::Job::NoVec) {
        if (iam == 0) 
            printf("\nskipping: Only singular values supported (vectors not yet supported)\n");
        return;
    }

    // figure out local size, allocate, create descriptor, initialize
    // matrix A (local input), m-by-n
    int64_t mlocA = scalapack_numroc(m, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    int descA_tst[9];
    scalapack_descinit(descA_tst, m, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix U (local output), U(m, minmn), singular values of A
    int64_t mlocU = scalapack_numroc(m, nb, myrow, izero, nprow);
    int64_t nlocU = scalapack_numroc(minmn, nb, mycol, izero, npcol);
    int descU_tst[9];
    scalapack_descinit(descU_tst, m, minmn, nb, nb, izero, izero, ictxt, mlocU, &info);
    slate_assert(info == 0);
    int64_t lldU = (int64_t)descU_tst[8];
    std::vector<scalar_t> U_tst(lldU * nlocU, 0);

    // matrix VT (local output), VT(minmn, n)
    int64_t mlocVT = scalapack_numroc(minmn, nb, myrow, izero, nprow);
    int64_t nlocVT = scalapack_numroc(n, nb, mycol, izero, npcol);
    int descVT_tst[9];
    scalapack_descinit(descVT_tst, minmn, n, nb, nb, izero, izero, ictxt, mlocVT, &info);
    slate_assert(info == 0);
    int64_t lldVT = (int64_t)descVT_tst[8];
    std::vector<scalar_t> VT_tst(lldVT * nlocVT, 0);

    int64_t lwork = 1 + 6*std::max(m, n) + 1*std::max(m, n);
    std::vector<scalar_t> work(lwork, 0);

    // matrix S (global output), S(size), singular values of A
    std::vector<real_t> S_tst(minmn);

    slate::Matrix<scalar_t> A; // (m, n);
    slate::Matrix<scalar_t> U; // (m, minmn);
    slate::Matrix<scalar_t> VT; // (minmn, n);
    std::vector<real_t> S;

    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        S = S_tst;

        U = slate::Matrix<scalar_t>(m, minmn, nb, nprow, npcol, MPI_COMM_WORLD);
        U.insertLocalTiles(origin_target);
        copy(&U_tst[0], descU_tst, U); // U is output, so not really needed

        VT = slate::Matrix<scalar_t>(minmn, n, nb, nprow, npcol, MPI_COMM_WORLD);
        VT.insertLocalTiles(origin_target);
        copy(&VT_tst[0], descVT_tst, VT); // VT is output, so not really needed
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A = slate::Matrix<scalar_t>::fromScaLAPACK(m, n, &A_tst[0],  lldA,  nb, nprow, npcol, MPI_COMM_WORLD);
        S = S_tst;
        U = slate::Matrix<scalar_t>::fromScaLAPACK(m, minmn, &U_tst[0], lldU, nb, nprow, npcol, MPI_COMM_WORLD);
        VT = slate::Matrix<scalar_t>::fromScaLAPACK(minmn, n, &VT_tst[0], lldVT, nb, nprow, npcol, MPI_COMM_WORLD);
    }

    if (verbose >= 1) {
        printf( "%% A   %6lld-by-%6lld\n", llong(   A.m() ), llong(   A.n() ) );
        printf( "%% U   %6lld-by-%6lld\n", llong(   U.m() ), llong(   U.n() ) );
        printf( "%% VT  %6lld-by-%6lld\n", llong(  VT.m() ), llong(  VT.n() ) );
    }

    if (verbose > 1) {
        print_matrix( "A",  A  );
        print_matrix( "U",  U  );
        print_matrix( "VT", VT );
    }

    std::vector<scalar_t> A_ref, U_ref, VT_ref;
    std::vector<real_t> S_ref;
    if (ref) {
        A_ref = A_tst;
        S_ref = S_tst;
        U_ref = U_tst;
        VT_ref = VT_tst;
    }

    if (runtst) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time = libtest::get_wtime();

        //==================================================
        // Run SLATE test.
        //==================================================
        ////////////////////////////////////////////////////////////  
        // todo: Wrong call below here
        if (iam == 0) printf("TODO: REAL GESVD CALL NEEDED... Edit test_gesvd.cc and update\n");
        if (0==1) { 
            gesvd(A, S, {
                    {slate::Option::Lookahead, lookahead},
                    {slate::Option::Target, target}
                });
        }
        // Run using ScaLAPACK
        slate_set_num_blas_threads(omp_get_max_threads());
        // query for workspace size        
        int64_t info_tst = 0;
        scalar_t dummywork;
        scalapack_pgesvd(job2str(jobu), job2str(jobvt), m, n, 
                         &A_tst[0],  ione, ione, descA_tst, 
                         &S_tst[0],
                         &U_tst[0], ione, ione, descU_tst,
                         &VT_tst[0], ione, ione, descVT_tst,
                         &dummywork, -1, &info_tst);
        slate_assert(info_tst == 0);
        lwork = int64_t( real( dummywork ) );
        work.resize(lwork);
        // Run ScaLAPACK reference routine.
        scalapack_pgesvd(job2str(jobu), job2str(jobvt), m, n, 
                         &A_tst[0],  ione, ione, descA_tst, 
                         &S_tst[0],
                         &U_tst[0], ione, ione, descU_tst,
                         &VT_tst[0], ione, ione, descVT_tst,
                         &work[0], lwork, &info_tst);
        slate_assert(info_tst == 0);
        slate_set_num_blas_threads(1);
        // todo: Wrong call above here
        ////////////////////////////////////////////////////////////  

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = libtest::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;

        if (verbose > 1) {
            print_matrix( "A",  A  );
            print_matrix( "U",  U  );
            print_matrix( "VT", VT );
        }
    }

    if (ref) {
        // Run reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads = 1;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);

        // query for workspace size        
        int64_t info_ref = 0;
        scalar_t dummywork;
        scalapack_pgesvd(job2str(jobu), job2str(jobvt), m, n, 
                         &A_ref[0],  ione, ione, descA_tst, &S_ref[0],
                         &U_ref[0], ione, ione, descU_tst,
                         &VT_ref[0], ione, ione, descVT_tst,
                         &dummywork, -1, &info_ref);
        slate_assert(info_ref == 0);
        lwork = int64_t( real( dummywork ) );
        work.resize(lwork);

        // Run ScaLAPACK reference routine.
        MPI_Barrier(MPI_COMM_WORLD);
        double time = libtest::get_wtime();
        scalapack_pgesvd(job2str(jobu), job2str(jobvt), m, n, 
                         &A_ref[0],  ione, ione, descA_tst, &S_ref[0],
                         &U_ref[0], ione, ione, descU_tst,
                         &VT_ref[0], ione, ione, descVT_tst,
                         &work[0], lwork, &info_ref);
        slate_assert(info_ref == 0);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

        params.ref_time() = time_ref;

        slate_set_num_blas_threads(saved_num_threads);
    }

    if (ref && check) {
        // Reference Scalapack was run, check reference against test
        // perform a local operation to get differences S_ref = S_ref - S_tst
        blas::axpy(S_ref.size(), -1.0, &S_tst[0], 1, &S_ref[0], 1);
        // norm(S_ref - S_tst)
        real_t S_diff_norm = lapack::lange(norm, S_ref.size(), 1, &S_ref[0], 1);
        // todo: Is the scaling meaningful
        real_t error = S_diff_norm / std::max(m, n); 

        params.error() = error;
        // todo: Any justification for this tolerance
        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay() = (params.error() <= 1*eps);
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_gesvd(Params& params, bool run)
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gesvd_work<float> (params, run);
            break;

        case libtest::DataType::Double:
            test_gesvd_work<double> (params, run);
            break;

        case libtest::DataType::SingleComplex:
            test_gesvd_work<std::complex<float>> (params, run);
            break;

        case libtest::DataType::DoubleComplex:
            test_gesvd_work<std::complex<double>> (params, run);
            break;
    }
}
