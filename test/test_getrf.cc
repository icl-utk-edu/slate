#include "slate.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"

#include "slate_mpi.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads (int nt);
inline int slate_set_num_blas_threads (const int nt) { return MKL_Set_Num_Threads (nt); }
#else
inline int slate_set_num_blas_threads (const int nt) { return -1; }
#endif

//------------------------------------------------------------------------------
template <typename scalar_t> void test_getrf_work (Params &params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t nb = params.nb.value();
    int64_t lookahead = params.lookahead.value();
    int64_t panel_threads = params.panel_threads.value();
    lapack::Norm norm = params.norm.value();
    bool check = params.check.value()=='y';
    bool ref = params.ref.value()=='y';
    bool trace = params.trace.value()=='y';
    slate::Target target = char2target (params.target.value());

    // mark non-standard output values
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    int64_t Am = m;
    int64_t An = n;

    // Local values
    static int i0=0, i1=1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9];
    int iam=0, nprocs=1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo (&iam, &nprocs);
    assert (p*q <= nprocs);
    Cblacs_get (-1, 0, &ictxt);
    Cblacs_gridinit (&ictxt, "Col", p, q);
    Cblacs_gridinfo (ictxt, &nprow, &npcol, &myrow, &mycol);

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc (Am, nb, myrow, i0, nprow);
    int64_t nlocA = scalapack_numroc (An, nb, mycol, i0, npcol);
    scalapack_descinit (descA_tst, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info);
    assert (info==0);
    int64_t lldA = (int64_t)descA_tst[8];
    std::vector<scalar_t> A_tst (lldA * nlocA);
    scalapack_pplrnt (&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1);

    // allocate ipiv locally
    size_t ipiv_size = (size_t) (lldA + nb);
    std::vector<int> ipiv_tst(ipiv_size);

    // Create SLATE matrix from the ScaLAPACK layouts
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK (Am, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    std::vector<int> ipiv_ref;
    if (check || ref) {
        A_ref.resize (A_tst.size());
        A_ref = A_tst;
        scalapack_descinit (descA_ref, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info);
        assert (info==0);
        ipiv_ref.resize(ipiv_tst.size());
    }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // run test
    {
        slate::trace::Block trace_block ("MPI_Barrier");
        MPI_Barrier (MPI_COMM_WORLD);
    }
    
    slate::Pivots pivots;

    double time = libtest::get_wtime();

    // todo: the example/test call to pgetrf can be removed later
    // int64_t info_tst=0;
    // scalapack_pgetrf (m, n, &A_tst[0], i1, i1, descA_tst, &ipiv_tst[0], &info_tst);
    slate::getrf (A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads}
    });

    MPI_Barrier (MPI_COMM_WORLD);
    {
        slate::trace::Block trace_block ("MPI_Barrier");
        MPI_Barrier (MPI_COMM_WORLD);
    }
    double time_tst = libtest::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    double gflop = lapack::Gflop<scalar_t>::getrf (m, n);
    params.time.value() = time_tst;
    params.gflops.value() = gflop / time_tst;

    if (check || ref) {
        // A comparison with a reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);
        int64_t info_ref=0;

        // todo: The IPIV needs to be checked

        //when matrix is square
        if(Am == An){
            //check residual for accuracy by performing a solve

            //================================================================
            // Test results by checking the residual
            //
            //                      || B - AX ||_I
            //                --------------------------- < epsilon
            //                 || A ||_I * || X ||_I * N
            //
            //================================================================
            int64_t Bm = n;
            int64_t Bn = n;
            int descB_tst[9], descB_ref[9];
            std::vector<scalar_t> B_ref;

            // matrix B, figure out local size, allocate, create descriptor, initialize
            int64_t mlocB = scalapack_numroc (Bm, nb, myrow, i0, nprow);
            int64_t nlocB = scalapack_numroc (Bn, nb, mycol, i0, npcol);
            scalapack_descinit (descB_tst, Bm, Bn, nb, nb, i0, i0, ictxt, mlocB, &info);
            assert (info==0);
            int64_t lldB = (int64_t)descB_tst[8];
            std::vector<scalar_t> B_tst (lldB * nlocB);
            scalapack_pplrnt (&B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed+2);

            B_ref.resize (B_tst.size());
            B_ref = B_tst;
            scalapack_descinit (descB_ref, Bm, Bn, nb, nb, i0, i0, ictxt, mlocB, &info);
            assert (info==0);

            //*
            auto B = slate::Matrix<scalar_t>::fromScaLAPACK (Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD);
            
            slate::getrs (A, pivots, B, {
                {slate::Option::Lookahead, lookahead},
                {slate::Option::Target, target}
            });
            /*/
            // solving with Scalapack suffers accuracy decline !!
            // collect the pivots
            for (int64_t k = 0, ps = pivots.size(), ip_ind = 0; k < ps; ++k) {
                for (int64_t t = 0, pks = pivots.at(k).size(); t < pks; ++t) {
                    //TODO this is not good when tile size is not uniform
                    ipiv_tst[ip_ind++] = pivots.at(k).at(t).tileIndex() * nb + pivots.at(k).at(t).elementOffset() + 1;
                }
            }

            scalapack_pgetrs ("N", Bm, Bn, &A_tst[0], i1, i1, descA_tst, &ipiv_tst[0], &B_tst[0], i1, i1, descB_tst, &info_ref);
            assert (0 == info_ref);
            //*/

            // allocate work space
            std::vector<real_t> worklangeA (std::max ({mlocA, nlocA}));
            std::vector<real_t> worklangeB (std::max ({mlocB, nlocB}));

            // Norm of the orig matrix: || A ||_I
            real_t A_norm = scalapack_plange (norm2str (norm), Am, An, &A_ref[0], i1, i1, descA_ref, &worklangeA[0]);
            // norm of updated rhs matrix: || X ||_I
            real_t X_norm = scalapack_plange (norm2str (norm), Bm, Bn, &B_tst[0], i1, i1, descB_tst, &worklangeB[0]);

            // B_ref -= Aref*B_tst
            scalapack_pgemm ("notrans", "notrans", 
                             Bm, Bn, An, 
                             scalar_t(-1.0),
                             &A_ref[0], i1, i1, descA_ref,
                             &B_tst[0], i1, i1, descB_tst, 
                             scalar_t(1.0),
                             &B_ref[0], i1, i1, descB_ref);

            // || B - AX ||_I
            real_t R_norm = scalapack_plange (norm2str (norm), Bm, Bn, &B_ref[0], i1, i1, descB_ref, &worklangeB[0]);

            double residual = R_norm / (n * A_norm * X_norm);
            params.error.value() = residual;
        }

        // allocate work space
        std::vector<real_t> worklange (std::max ({mlocA, nlocA}));

        // Run the reference routine
        MPI_Barrier (MPI_COMM_WORLD);
        time = libtest::get_wtime();
        scalapack_pgetrf (m, n, &A_ref[0], i1, i1, descA_ref, &ipiv_ref[0], &info_ref);
        assert (0 == info_ref);
        MPI_Barrier (MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

        params.ref_time.value() = time_ref;
        params.ref_gflops.value() = gflop / time_ref;

        //when matrix is rectangular, we cannot solve
        if(Am != An){
            //compare to reference implementation for accuracy check

            // Norm of the reference result
            real_t A_ref_norm = scalapack_plange (norm2str (norm), Am, An, &A_ref[0], i1, i1, descA_ref, &worklange[0]);

            // local operation: error = A_ref = A_ref - A_tst;   ipiv_ref = ipiv_ref - ipiv_tst
            blas::axpy (A_ref.size(), -1.0, &A_tst[0], 1, &A_ref[0], 1);

            // error = norm(error)
            real_t error_norm = scalapack_plange (norm2str (norm), Am, An, &A_ref[0], i1, i1, descA_ref, &worklange[0]);

            // error = error / reference;
            if (A_ref_norm != 0)
                error_norm /= A_ref_norm;
            
            params.error.value() = error_norm;
        }


        slate_set_num_blas_threads(saved_num_threads);

        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay.value() = (params.error.value() <= 3*eps);
    }

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_getrf (Params &params, bool run)
{
    switch (params.datatype.value()) {
    case libtest::DataType::Integer:
        throw std::exception();
        break;

    case libtest::DataType::Single:
        test_getrf_work<float> (params, run);
        break;

    case libtest::DataType::Double:
        test_getrf_work<double> (params, run);
        break;

    case libtest::DataType::SingleComplex:
        test_getrf_work<std::complex<float>> (params, run);
        break;

    case libtest::DataType::DoubleComplex:
        test_getrf_work<std::complex<double>> (params, run);
        break;
    }
}
