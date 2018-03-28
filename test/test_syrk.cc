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

#include "slate.hh"
#include "slate_Debug.hh"
#include "test.hh"
#include "error.hh"
#include "blas_flops.hh"

#include "scalapack_wrappers.hh"

#ifdef SLATE_WITH_MKL
//#include "mkl.h"
extern "C" int MKL_Set_Num_Threads( int nt );
#else
inline int MKL_Set_Num_Threads( int nt ) { return 1; }
#endif

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_syrk_work( Params& params, bool run )
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    scalar_t alpha  = params.alpha.value();
    scalar_t beta  = params.beta.value();
    // int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.n();
    int64_t nb = params.nb.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t lookahead = params.lookahead.value();
    bool check = params.check.value()=='y';
    bool ref = params.ref.value()=='y';
    bool trace = params.trace.value()=='y';

    // mark non-standard output values
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // Local values
    static int i0=0, i1=1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info, mloc, nloc;
    int descA_tst[9], descC_tst[9], descC_ref[9];
    int iam=0, nprocs=1;
    int n_ = n; 
    int nb_ = nb;

    // Initialize BLACS and ScaLAPACK
    Cblacs_pinfo( &iam, &nprocs );  assert(p*q <= nprocs);
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Row", p, q );
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    mloc = scalapack_numroc( &n_, &nb_, &myrow, &i0, &nprow );
    nloc = scalapack_numroc( &n_, &nb_, &mycol, &i0, &npcol );

    // typedef long long lld;

    // Allocate space
    size_t size_A = (size_t)( mloc*nloc );
    std::vector< scalar_t > A_tst( size_A );
    size_t size_C = (size_t)( mloc*nloc );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref;

    // Initialize the matrix
    int iseed = 0;
    scalapack_pdplrnt( &A_tst[0], n_, n_, nb_, nb_, myrow, mycol, nprow, npcol, mloc, iseed+1 );
    scalapack_pdplghe( &C_tst[0], n_, n_, nb_, nb_, myrow, mycol, nprow, npcol, mloc, iseed+2 );

    // Create ScaLAPACK descriptors
    scalapack_descinit( descA_tst, &n_, &n_, &nb_, &nb_, &i0, &i0, &ictxt, &mloc, &info ); assert(info==0);
    scalapack_descinit( descC_tst, &n_, &n_, &nb_, &nb_, &i0, &i0, &ictxt, &mloc, &info ); assert(info==0);

    // If check is required, save data and create a descriptor for it
    if ( check || ref ) {
        C_ref.resize( size_C );
        C_ref = C_tst;
        scalapack_descinit( descC_ref, &n_, &n_, &nb_, &nb_, &i0, &i0, &ictxt, &mloc, &info ); assert(info==0);
    }

    // Create SLATE matrices from the ScaLAPACK layouts
    int64_t llda = (int64_t)descA_tst[8];
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK( n_, n_, &A_tst[0], llda, nb_, nprow, npcol, MPI_COMM_WORLD );
    int64_t lldc = (int64_t)descC_tst[8];
    auto C = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK( uplo, n_, &C_tst[0], lldc, nb_, nprow, npcol, MPI_COMM_WORLD );

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // Call the routine using ScaLAPACK layout
    MPI_Barrier(MPI_COMM_WORLD);
    double time = libtest::get_wtime();
    if ( params.target.value() == 't' )
        slate::syrk<slate::Target::HostTask>(
            alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
    else if ( params.target.value() == 'n' )
        slate::syrk<slate::Target::HostNest>(
            alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
    else if ( params.target.value() == 'b' )
        slate::syrk<slate::Target::HostBatch>(
            alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
    else if ( params.target.value() == 'd' )
        slate::syrk<slate::Target::Devices>(
            alpha, A, beta, C, {{slate::Option::Lookahead, lookahead}});
    // scalapack_psyrk( side2str(side), uplo2str(uplo), op2str(trans), diag2str(diag),
    //                  &n_, &n_, &alpha,
    //                  &A_tst[0], &i1, &i1, descA_tst,
    //                  &B_tst[0], &i1, &i1, descB_tst);
    MPI_Barrier(MPI_COMM_WORLD);
    double time_tst = libtest::get_wtime() - time;
    
    if (trace) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop < scalar_t >::syrk( n, k );
    params.time.value() = time_tst;
    params.gflops.value() = gflop / time_tst;

    real_t tol = params.tol.value();

    if ( check || ref ) {
        // Comparison with reference routine from ScaLAPACK

        // Set MKL num threads appropriately for parallel BLAS
        int omp_num_threads = 1;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_mkl_num_threads = MKL_Set_Num_Threads(omp_num_threads);

        // Run the reference routine
        MPI_Barrier(MPI_COMM_WORLD);        
        double time = libtest::get_wtime();
        // scalapack_psyrk( side2str(side), uplo2str(uplo), op2str(trans), diag2str(diag),  
        //                  &n_, &n_, &alpha,
        //                  &A_tst[0], &i1, &i1, descA_tst,
        //                  &B_ref[0], &i1, &i1, descB_ref);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_ref = libtest::get_wtime() - time;

        // Allocate work space
        std::vector< scalar_t > worklange( mloc );

        // Local operation: error = _ref - _tst
        blas::axpy(size_C, -1.0, &C_tst[0], 1, &C_ref[0], 1);

        // norm(B_tst)
        real_t B_tst_norm = scalapack_plange( "I", &n_, &n_, &C_tst[0], &i1, &i1, descC_tst, &worklange[0]);

        // norm(B_ref - B_tst)
        real_t error_norm = scalapack_plange( "I", &n_, &n_, &C_ref[0], &i1, &i1, descC_tst, &worklange[0] );
        if ( B_tst_norm != 0 )
            error_norm /=  B_tst_norm;

        params.ref_time.value() = time_ref;
        params.ref_gflops.value() = gflop / time_ref;
        params.error.value() = error_norm;

        MKL_Set_Num_Threads(saved_mkl_num_threads);
    }

    params.okay.value() = (params.error.value() <= tol);

    //Cblacs_exit(1) is commented out because it does not handle re-entering ... some unknown problem
    //Cblacs_exit(1); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_syrk( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            throw std::exception();// test_syrk_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_syrk_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            throw std::exception();// test_syrk_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            throw std::exception();// test_syrk_work< std::complex<double> >( params, run );
            break;
    }
}
