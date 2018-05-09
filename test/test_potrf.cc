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
extern "C" int MKL_Set_Num_Threads( int nt );
inline int slate_set_blas_num_threads( const int nt ) { return MKL_Set_Num_Threads( nt ); }
#else
inline int slate_set_blas_num_threads( const int nt ) { return -1; }
#endif

//------------------------------------------------------------------------------
template <typename scalar_t> void test_potrf_work( Params &params, bool run )
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    slate::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t nb = params.nb.value();
    int64_t lookahead = params.lookahead.value();
    lapack::Norm norm = params.norm.value();
    bool check = params.check.value()=='y';
    bool ref = params.ref.value()=='y';
    bool trace = params.trace.value()=='y';
    slate::Target target = char2target( params.target.value() );

    // mark non-standard output values
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    if ( !run )
        return;

    int64_t Am = n;
    int64_t An = n;

    // Local values
    static int i0=0, i1=1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descA_ref[9];
    int iam=0, nprocs=1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo( &iam, &nprocs );
    assert( p*q <= nprocs );
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Col", p, q );
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc( Am, nb, myrow, i0, nprow );
    int64_t nlocA = scalapack_numroc( An, nb, mycol, i0, npcol );
    scalapack_descinit( descA_tst, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info );
    assert( info==0 );
    int64_t lldA = ( int64_t )descA_tst[8];
    std::vector<scalar_t> A_tst( lldA * nlocA );
    scalapack_pplghe( &A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1 );

    // Create SLATE matrix from the ScaLAPACK layouts
    auto A = slate::HermitianMatrix<scalar_t>::fromScaLAPACK( uplo, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD );

    // if check is required, copy test data and create a descriptor for it
    std::vector<scalar_t> A_ref;
    if ( check || ref ) {
        A_ref.resize( A_tst.size() );
        A_ref = A_tst;
        scalapack_descinit( descA_ref, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info );
        assert( info==0 );
    }

    if ( trace ) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // run test
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier( MPI_COMM_WORLD );
    }
    double time = libtest::get_wtime();

    slate::potrf( A, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target} } );

    MPI_Barrier( MPI_COMM_WORLD );
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier( MPI_COMM_WORLD );
    }
    double time_tst = libtest::get_wtime() - time;

    if ( trace ) slate::trace::Trace::finish();

    // compute and save timing/performance
    double gflop = lapack::Gflop<scalar_t>::potrf( n );
    params.time.value() = time_tst;
    params.gflops.value() = gflop / time_tst;

    if ( check || ref ) {
        // A comparison with a reference routine from ScaLAPACK
        // This expects to get the original (clean) A_ref and will alter A_ref

        // Run the reference routine on A_ref
        MPI_Barrier( MPI_COMM_WORLD );
        double time = libtest::get_wtime();
        scalapack_ppotrf( uplo2str( uplo ), n, &A_ref[0], i1, i1, descA_ref, &info );
        assert( 0 == info );
        MPI_Barrier( MPI_COMM_WORLD );
        double time_ref = libtest::get_wtime() - time;

        // allocate work space
        size_t ldw = nb * ceil( ceil( mlocA / ( double ) nb ) / ( scalapack_ilcm( &nprow, &npcol ) / nprow ) );
        std::vector<real_t> worklansy( 2 * nlocA + mlocA + ldw );

        // norm of A
        real_t A_ref_norm = scalapack_plansy( norm2str( norm ), uplo2str( uplo ), n, &A_ref[0], i1, i1, descA_ref, &worklansy[0] );

        // local operation: error = A_ref = A_ref - A_tst
        blas::axpy( A_ref.size(), -1.0, &A_tst[0], 1, &A_ref[0], 1 );

        // error = norm(error)
        real_t error_norm = scalapack_plansy( norm2str( norm ), uplo2str( uplo ), n, &A_ref[0], i1, i1, descA_ref, &worklansy[0] );

        // error = error / norm;
        if ( error_norm != 0 )
            error_norm /= A_ref_norm;

        params.ref_time.value() = time_ref;
        params.ref_gflops.value() = gflop / time_ref;
        params.error.value() = error_norm;

        real_t eps = std::numeric_limits<real_t>::epsilon();
        params.okay.value() = ( params.error.value() <= 50*eps );
    }

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_potrf( Params &params, bool run )
{
    switch ( params.datatype.value() ) {
    case libtest::DataType::Integer:
        throw std::exception();
        break;

    case libtest::DataType::Single:
        test_potrf_work<float>( params, run );
        break;

    case libtest::DataType::Double:
        test_potrf_work<double>( params, run );
        break;

    case libtest::DataType::SingleComplex:
        test_potrf_work<std::complex<float>>( params, run );
        break;

    case libtest::DataType::DoubleComplex:
        test_potrf_work<std::complex<double>>( params, run );
        break;
    }
}
