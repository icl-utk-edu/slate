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

#include "myscalapack_fortran.h"
#include "myscalapack_common.h"

#ifdef SLATE_WITH_MPI
#include <mpi.h>
#else
#include "slate_NoMpi.hh"
#endif

extern "C" void trace_on();
extern "C" void trace_off();
extern "C" void trace_finish();

static int i0=0, i1=1;
static double m1=-1e0, p1=1e0;

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_pdpotrf_work( Params& params, bool run )
{
    typedef typename blas::traits< scalar_t >::real_t real_t;
    // typedef long long lld;
    
    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    // int64_t align = params.align.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    // int64_t lookahead = params.lookahead.value();
    int64_t s_ = params.nrhs.value();
    int s = s_;
   
    int64_t nb_ = params.nb.value();
    int nb = nb_;
    // int64_t nt = params.n.value();
    int64_t n_ = params.dim.n();
    int n = n_;
    int m = n_;
    const char *uplo_str = blas::uplo2str(uplo);

    // mark non-standard output values
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();
    params.error2.value();

    if (! run)
        return;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info, mloc, nloc, sloc;
    int descA_tst[9];
    int descA_ref[9];
    int iam = 0;
    int nprocs = 1;

    Cblacs_pinfo( &iam, &nprocs );
    assert(p*q <= nprocs);
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Row", p, q );
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    mloc = numroc_( &m, &nb, &myrow, &i0, &nprow );
    nloc = numroc_( &n, &nb, &mycol, &i0, &npcol );
    sloc = numroc_( &s, &nb, &mycol, &i0, &npcol );

    size_t size_A = (size_t) ( mloc*nloc);
    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );

    // Initialize the matrix
    int iseed = iam;
    descinit_( descA_tst, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
    assert( 0 == info );
    scalapack_pdplghe( &A_tst[0], m, n, nb, nb, myrow, mycol, nprow, npcol, mloc, iseed );

    // If check is required, save A in A_ref and create a descriptor for it
    if ( params.check.value() == 'y' ) {
        descinit_( descA_ref, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
        assert( 0 == info );
        A_ref = A_tst;
    }

    // Call the routine using ScaLAPACK layout
    double time = libtest::get_wtime();
    // slate::potrf< slate::Target::HostTask >(uplo, A, lookahead);
    pdpotrf_( uplo_str, &n, &A_tst[0], &i1, &i1, descA_tst, &info );
    assert( 0 == info );
    double time_tst = libtest::get_wtime() - time;

    // Get the maximum time 
    if( 0 != iam ) {
        MPI_Reduce( &time_tst, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    } else {
        MPI_Reduce( MPI_IN_PLACE, &time_tst, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    }

    // Compute and save timing/performance 
    double gflop = lapack::Gflop< scalar_t >::potrf( n );
    params.time.value() = time_tst;
    params.gflops.value() = gflop / time_tst;

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol.value();

    if ( params.check.value() == 'y' ) {
        // This should not alter A_ref
        // A numerical check of the value

        // Variables required
        real_t Anorm, Bnorm, Xnorm, Rnorm=-1;

        // create B and copy it to X
        int descBref[9];
        descinit_( descBref, &n, &s, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
        assert( 0 == info );
        size_t size_B = (size_t) ( mloc*sloc);
        std::vector< scalar_t > B_ref( size_B );
        std::vector< scalar_t > X_ref( size_B );

        // Initialize B 
        scalapack_pdplrnt( &B_ref[0], n, s, nb, nb, myrow, mycol,  nprow, npcol, mloc, iseed + 1 );
        // Copy B to X (use local copies instead of globally via ScaLAPACK pdlacpy)
        X_ref = B_ref;

        // Get norms of A and B
        size_t ldw = nb*ceil(ceil(mloc/(double)nb)/(ilcm_(&nprow, &npcol)/nprow));
        std::vector< scalar_t > worklansy( 2*nloc + mloc + ldw );
        Anorm = pdlansy_( "I", uplo_str, &n, &A_ref[0], &i1, &i1, descA_ref, &worklansy[0] );
        Bnorm = pdlange_( "I", &n, &s, &B_ref[0], &i1, &i1, descBref, &worklansy[0] );

        // Solve for X using the A_tst factorization
        pdpotrs_( uplo_str, &n, &s, &A_tst[0], &i1, &i1, descA_ref, &X_ref[0], &i1, &i1, descBref, &info );
        assert( 0 == info );

        // Compute B(diff) = B - A_ref * X 
        pdsymm_( uplo_str, uplo_str, &n, &s, &m1, &A_ref[0], &i1, &i1, descA_tst, &X_ref[0], &i1, &i1, descBref, &p1, &B_ref[0], &i1, &i1, descBref);

        // Norms of X and B(diff)
        std::vector< scalar_t > worklange( mloc );
        Xnorm = pdlange_( "I", &n, &s, &X_ref[0], &i1, &i1, descBref, &worklange[0] );
        Rnorm = pdlange_( "I", &n, &s, &B_ref[0], &i1, &i1, descBref, &worklange[0] );

        double resid = Rnorm / ( (Bnorm + Anorm * Xnorm) * fmax(m, n) );
        params.error.value() = resid;
    }

    if ( params.ref.value() == 'y' ) {
        // This expects to get the original (clean) A_ref and will alter A_ref
        // A comparison with a reference routine from ScaLAPACK
        
        // Run the reference routine on A_ref
        double time = libtest::get_wtime();
        pdpotrf_( uplo_str, &n, &A_ref[0], &i1, &i1, descA_ref, &info );
        assert( 0 == info );
        double time_ref = libtest::get_wtime() - time;

        // norm(A_ref)        
        size_t ldw = nb*ceil(ceil(mloc/(double)nb)/(ilcm_(&nprow, &npcol)/nprow));
        std::vector< scalar_t > worklansy( 2*nloc + mloc + ldw );
        real_t A_ref_norm = pdlansy_( "I", uplo_str, &n, &A_ref[0], &i1, &i1, descA_ref, &worklansy[0] );

        // Local operation: error = A_ref = A_ref - A_tst
        for(size_t i = 0; i < A_ref.size(); i++)
            A_ref[i] = A_ref[i] - A_tst[i];

        // error = norm(error)
        real_t error_norm = pdlansy_( "I", uplo_str, &n, &A_ref[0], &i1, &i1, descA_ref, &worklansy[0] );

        // error = error / norm;
        if (error_norm != 0)
            error_norm /= A_ref_norm;

        // Get the max of time for the reference run
        if( 0 != iam ) {
            MPI_Reduce( &time_ref, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        } else {
            MPI_Reduce( MPI_IN_PLACE, &time_ref, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        }

        params.ref_time.value() = time_ref;
        params.ref_gflops.value() = gflop / time_ref;
        
        params.error2.value() = error_norm;
    }

    params.okay.value() = ( (params.error.value() <= tol) && (params.error2.value() <= 3*eps) );

    //Cblacs_exit is commented out because it does not all re-entering ... some unknown problem
    //Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_pdpotrf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            throw std::exception();// test_pdpotrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pdpotrf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            throw std::exception();// test_pdpotrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            throw std::exception();// test_pdpotrf_work< std::complex<double> >( params, run );
            break;
    }
}
