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
template <typename scalar_t>
void test_gels_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;
    using llong = long long;

    // get & mark input values
    slate::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t p = params.p();
    int64_t q = params.q();
    int64_t nb = params.nb();
    int64_t ib = params.ib();
    int64_t lookahead = params.lookahead();
    int64_t panel_threads = params.panel_threads();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    bool consistent = true;

    // mark non-standard output values
    params.error.name("leastsqr\n" "error");
    params.error2();
    params.error2.name("min. norm\n" "error");
    params.error3();
    params.error3.name("residual\n" "error");
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // A is m-by-n, BX is max(m, n)-by-nrhs.
    // If op == NoTrans, op(A) is m-by-n, X is n-by-nrhs, B is m-by-nrhs
    // otherwise,        op(A) is n-by-m, X is m-by-nrhs, B is n-by-nrhs.
    int64_t opAm = (trans == slate::Op::NoTrans ? m : n);
    int64_t opAn = (trans == slate::Op::NoTrans ? n : m);
    int64_t maxmn = std::max(m, n);

    // Local values
    const int izero = 0, ione = 1;
    const scalar_t zero = 0, one = 1;

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

    // figure out local size, allocate, create descriptor, initialize
    // matrix A, m-by-n
    int64_t mlocA = scalapack_numroc(m, nb, myrow, izero, nprow);
    int64_t nlocA = scalapack_numroc(n, nb, mycol, izero, npcol);
    int descA_tst[9];
    scalapack_descinit(descA_tst, m, n, nb, nb, izero, izero, ictxt, mlocA, &info);
    slate_assert(info == 0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst(lldA*nlocA);
    scalapack_pplrnt(&A_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed + 1);

    // matrix X0, opAn-by-nrhs
    // used if making a consistent equation, B = A*X0
    int64_t mlocX0 = scalapack_numroc(opAn, nb, myrow, izero, nprow);
    int64_t nlocX0 = scalapack_numroc(nrhs, nb, mycol, izero, npcol);
    int descX0_tst[9];
    scalapack_descinit(descX0_tst, opAn, nrhs, nb, nb, izero, izero, ictxt, mlocX0, &info);
    slate_assert(info == 0);
    int64_t lldX0 = (int64_t)descX0_tst[8];
    std::vector<scalar_t> X0_tst(lldX0*nlocX0);
    scalapack_pplrnt(&X0_tst[0], opAn, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocX0, iseed + 1);

    // matrix BX, which stores B (input) and X (output), max(m, n)-by-nrhs
    int64_t mlocBX = scalapack_numroc(maxmn, nb, myrow, izero, nprow);
    int64_t nlocBX = scalapack_numroc(nrhs,  nb, mycol, izero, npcol);
    int descBX_tst[9];
    scalapack_descinit(descBX_tst, maxmn, nrhs, nb, nb, izero, izero, ictxt, mlocBX, &info);
    slate_assert(info == 0);
    int64_t lldBX = (int64_t)descBX_tst[8];
    std::vector<scalar_t> BX_tst(lldBX*nlocBX);
    scalapack_pplrnt(&BX_tst[0], maxmn, nrhs, nb, nb, myrow, mycol, nprow, npcol, mlocBX, iseed + 1);

    // workspace for ScaLAPACK
    int64_t lwork;
    std::vector<scalar_t> work;

    slate::Matrix<scalar_t> A, X0, BX;
    if (origin != slate::Origin::ScaLAPACK) {
        // Copy local ScaLAPACK data to GPU or CPU tiles.
        slate::Target origin_target = origin2target(origin);
        A = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
        A.insertLocalTiles(origin_target);
        copy(&A_tst[0], descA_tst, A);

        X0 = slate::Matrix<scalar_t>(opAn, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
        X0.insertLocalTiles(origin_target);
        copy(&X0_tst[0], descX0_tst, X0);

        BX = slate::Matrix<scalar_t>(maxmn, nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
        BX.insertLocalTiles(origin_target);
        copy(&BX_tst[0], descBX_tst, BX);
    }
    else {
        // create SLATE matrices from the ScaLAPACK layouts
        A  = slate::Matrix<scalar_t>::fromScaLAPACK(m,     n,    &A_tst[0],  lldA,  nb, nprow, npcol, MPI_COMM_WORLD);
        X0 = slate::Matrix<scalar_t>::fromScaLAPACK(opAn,  nrhs, &X0_tst[0], lldX0, nb, nprow, npcol, MPI_COMM_WORLD);
        BX = slate::Matrix<scalar_t>::fromScaLAPACK(maxmn, nrhs, &BX_tst[0], lldBX, nb, nprow, npcol, MPI_COMM_WORLD);
    }
    // Create SLATE matrix from the ScaLAPACK layouts
    // slate::TriangularFactors<scalar_t> T;

    // In square case, B = X = BX. In rectangular cases, B or X is sub-matrix.
    auto B = BX;
    auto X = BX;
    if (opAm > opAn) {
        // over-determined
        X = BX.slice(0, opAn-1, 0, nrhs-1);
    }
    else if (opAm < opAn) {
        // under-determined
        B = BX.slice(0, opAm-1, 0, nrhs-1);
    }

    // Apply trans
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose(A);
    else if (trans == slate::Op::ConjTrans)
        opA = conjTranspose(A);

    if (verbose >= 1) {
        printf( "%% A   %6lld-by-%6lld\n", llong(   A.m() ), llong(   A.n() ) );
        printf( "%% opA %6lld-by-%6lld\n", llong( opA.m() ), llong( opA.n() ) );
        printf( "%% X0  %6lld-by-%6lld\n", llong(  X0.m() ), llong(  X0.n() ) );
        printf( "%% B   %6lld-by-%6lld\n", llong(   B.m() ), llong(   B.n() ) );
        printf( "%% X   %6lld-by-%6lld\n", llong(   X.m() ), llong(   X.n() ) );
        printf( "%% BX  %6lld-by-%6lld\n", llong(  BX.m() ), llong(  BX.n() ) );
    }

    // Form consistent RHS, B = A * X0.
    if (consistent) {
        slate::multiply(one, opA, X0, zero, B);
        // Using traditional BLAS/LAPACK name
        // slate::gemm(one, opA, X0, zero, B);
        if (origin != slate::Origin::ScaLAPACK) {
            // refresh ScaLAPACK data; B is sub-matrix of BX
            copy(BX, &BX_tst[0], descBX_tst);
        }
    }

    if (verbose > 1) {
        print_matrix( "A",  A  );
        print_matrix( "X0", X0 );
        print_matrix( "B",  B  );
        print_matrix( "X",  X  );
        print_matrix( "BX", BX );
    }

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref, BX_ref;
    slate::Matrix<scalar_t> Aref, opAref, BXref, Bref;
    int descA_ref[9], descBX_ref[9];
    if (check || ref) {
        A_ref = A_tst;
        scalapack_descinit(descA_ref, m, n, nb, nb, izero, izero, ictxt, mlocA, &info);
        slate_assert(info == 0);
        Aref = slate::Matrix<scalar_t>::fromScaLAPACK(
            m, n, &A_ref[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);

        BX_ref = BX_tst;
        scalapack_descinit(descBX_ref, maxmn, nrhs, nb, nb, izero, izero, ictxt, mlocBX, &info);
        slate_assert(info == 0);
        BXref = slate::Matrix<scalar_t>::fromScaLAPACK(
            maxmn, nrhs, &BX_ref[0], lldBX, nb, nprow, npcol, MPI_COMM_WORLD);

        if (opAm >= opAn) {
            Bref = BXref;
        }
        else if (opAm < opAn) {
            Bref = BXref.slice(0, opAm-1, 0, nrhs-1);
        }

        // Apply trans
        opAref = Aref;
        if (trans == slate::Op::Trans)
            opAref = transpose(Aref);
        else if (trans == slate::Op::ConjTrans)
            opAref = conjTranspose(Aref);
    }

    double gflop = lapack::Gflop<scalar_t>::gels(m, n, nrhs);

    if (! ref_only) {
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
        slate::least_squares_solve(opA, BX, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target},
            {slate::Option::MaxPanelThreads, panel_threads},
            {slate::Option::InnerBlocking, ib}
        });

        //---------------------
        // Using traditional BLAS/LAPACK name
        // slate::gels(opA, T, BX, {
        //     {slate::Option::Lookahead, lookahead},
        //     {slate::Option::Target, target},
        //     {slate::Option::MaxPanelThreads, panel_threads},
        //     {slate::Option::InnerBlocking, ib}
        // });

        {
            slate::trace::Block trace_block("MPI_Barrier");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double time_tst = testsweeper::get_wtime() - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time_tst;
        params.gflops() = gflop / time_tst;

        if (verbose > 1) {
            print_matrix( "A2", A );
            print_matrix( "BX2", BX );
        }
    }

    if (check) {
        //==================================================
        // Test results.
        //==================================================
        real_t tol = params.tol() * 0.5 * std::numeric_limits<real_t>::epsilon();

        real_t opA_norm = slate::norm(slate::Norm::One, opAref);
        real_t   B_norm = slate::norm(slate::Norm::One, Bref);

        // todo: need to store Bref for reference ScaLAPACK run?
        // residual = B - op(A) X, stored in Bref
        slate::multiply(-one, opAref, X, one, Bref);
        // Using traditional BLAS/LAPACK name
        // slate::gemm(-one, opAref, X, one, Bref);

        if (opAm >= opAn) {
            //--------------------------------------------------
            // Over-determined case, least squares solution.
            // Check that the residual Res = AX - B is orthogonal to op(A):
            //
            //      || Res^H op(A) ||_1
            //     ------------------------------------------- < tol * epsilon
            //      max(m, n, nrhs) || op(A) ||_1 * || B ||_1

            // todo: scale residual to unit max, and scale error below
            // see LAPACK [sdcz]qrt17.f
            //real_t R_max = slate::norm(slate::Norm::Max, B);
            //slate::scale(1, R_max, B);

            // RA = R^H op(A)
            slate::Matrix<scalar_t> RA(nrhs, opAn, nb, nprow, npcol, MPI_COMM_WORLD);
            RA.insertLocalTiles();
            auto RT = conjTranspose(Bref);
            slate::multiply(one, RT, opA, zero, RA);
            // Using traditional BLAS/LAPACK name
            // slate::gemm(one, RT, opA, zero, RA);

            real_t error = slate::norm(slate::Norm::One, RA);
            if (opA_norm != 0)
                error /= opA_norm;
            // todo: error *= R_max
            if (B_norm != 0)
                error /= B_norm;
            error /= blas::max(m, n, nrhs);
            params.error() = error;
            params.okay() = (params.error() <= tol);
        }
        else {
            //--------------------------------------------------
            // opAm < opAn
            // Under-determined case, minimum norm solution.
            // Check that X is in the row-span of op(A),
            // i.e., it has no component in the null space of op(A),
            // by doing QR factorization of D = [ op(A)^H, X ] and examining
            // E = R( opAm : opAm+nrhs-1, opAm : opAm+nrhs-1 ).
            //
            //      || E ||_max / max(m, n, nrhs) < tol * epsilon

            // op(A)^H is opAn-by-opAm, X is opAn-by-nrhs
            // D = [ op(A)^H, X ] is opAn-by-(opAm + pad + nrhs)
            // Xstart = (opAm + pad) / nb
            // padding with zero columns so X starts on nice boundary to need only local copy.
            int64_t Xstart = slate::ceildiv( slate::ceildiv( opAm, nb ), q ) * q;
            slate::Matrix<scalar_t> D(opAn, Xstart*nb + nrhs, nb, nprow, npcol, MPI_COMM_WORLD);
            D.insertLocalTiles();

            // zero D.
            // todo: only need to zero the padding tiles in D.
            set(zero, D);

            // copy op(A)^H -> D
            // todo: support op(A)^H = A^H. Requires distributed copy of A^H to D.
            slate_assert(trans != slate::Op::NoTrans);
            auto DS = D.slice(0, Aref.m()-1, 0, Aref.n()-1);
            copy(Aref, DS);
            auto DX = D.sub(0, D.mt()-1, Xstart, D.nt()-1);
            copy(X, DX);

            if (verbose >= 1)
                printf( "%% D %lld-by-%lld\n", llong( D.mt() ), llong( D.nt() ) );
            if (verbose > 1)
                print_matrix("D", D);

            slate::TriangularFactors<scalar_t> TD;
            slate::qr_factor(D, TD);
            //---------------------
            // Using traditional BLAS/LAPACK name
            // slate::geqrf(D, TD);

            if (verbose > 1) {
                auto DR = slate::TrapezoidMatrix<scalar_t>(
                    slate::Uplo::Upper, slate::Diag::NonUnit, D );
                print_matrix("DR", DR);
            }

            // error = || R_{opAm : opAn-1, opAm : opAm+nrhs-1} ||_max
            // todo: if we can slice R at arbitrary row, just do norm(Max, R_{...}).
            // todo: istart/ioffset assumes fixed nb
            int64_t istart  = opAm / nb; // row m's tile
            int64_t ioffset = opAm % nb; // row m's offset in tile
            real_t local_error = 0;
            for (int64_t i = istart; i < D.mt(); ++i) {
                for (int64_t j = std::max(i, Xstart); j < D.nt(); ++j) { // upper
                    if (D.tileIsLocal(i, j)) {
                        auto T = D(i, j);
                        if (i == j) {
                            for (int64_t jj = 0; jj < T.nb(); ++jj)
                                for (int64_t ii = ioffset; ii < jj && ii < T.mb(); ++ii) // upper
                                    local_error = std::max( local_error, std::abs( T(ii, jj) ) );
                        }
                        else {
                            for (int64_t jj = 0; jj < T.nb(); ++jj)
                                for (int64_t ii = ioffset; ii < T.mb(); ++ii)
                                    local_error = std::max( local_error, std::abs( T(ii, jj) ) );
                        }
                    }
                }
                ioffset = 0; // no offset for subsequent block rows
            }
            real_t error2;
            MPI_Allreduce(&local_error, &error2, 1, slate::mpi_type<real_t>::value, MPI_MAX, BX.mpiComm());
            error2 /= blas::max(m, n, nrhs);
            params.error2() = error2;
            params.okay() = (params.error2() <= tol);
        }

        //--------------------------------------------------
        // If op(A) X = B is consistent, because either B = op(A) X0
        // or opAm <= opAn, check the residual:
        //
        //      || Res ||_1
        //     ----------------------------------- < tol * epsilon
        //      max(m, n) || op(A) ||_1 || X ||_1
        if (consistent || opAm <= opAn) {
            real_t X_norm = slate::norm(slate::Norm::One, X);
            real_t error3 = slate::norm(slate::Norm::One, Bref);
            if (opA_norm != 0)
                error3 /= opA_norm;
            if (X_norm != 0)
                error3 /= X_norm;
            error3 /= std::max(m, n);
            params.error3() = error3;
            params.okay() = (params.okay() && params.error3() <= tol);
        }
    }

    if (ref) {
        // A comparison with a reference routine from ScaLAPACK for timing only

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);
        int64_t info_ref = 0;

        // query for workspace size
        scalar_t dummy;
        scalapack_pgels(op2str(trans), m, n, nrhs,
                        &A_ref[0],  ione, ione, descA_ref,
                        &BX_ref[0], ione, ione, descBX_ref,
                        &dummy, -1, &info_ref);
        slate_assert(info_ref == 0);
        lwork = int64_t( real( dummy ) );
        work.resize(lwork);

        //==================================================
        // Run ScaLAPACK reference routine.
        //==================================================
        MPI_Barrier(MPI_COMM_WORLD);
        double time = testsweeper::get_wtime();
        scalapack_pgels(op2str(trans), m, n, nrhs,
                        &A_ref[0],  ione, ione, descA_ref,
                        &BX_ref[0], ione, ione, descBX_ref,
                        work.data(), lwork, &info_ref);
        slate_assert(info_ref == 0);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = testsweeper::get_wtime() - time;

        params.ref_time() = time_ref;
        params.ref_gflops() = gflop / time_ref;

        slate_set_num_blas_threads(saved_num_threads);
    }

    Cblacs_gridexit(ictxt);
    //Cblacs_exit(1) does not handle re-entering
}

// -----------------------------------------------------------------------------
void test_gels(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gels_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_gels_work<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_gels_work<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gels_work<std::complex<double>> (params, run);
            break;
    }
}
