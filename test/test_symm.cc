#include "slate.hh"
#include "test.hh"
#include "blas_flops.hh"

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
inline int slate_set_num_blas_threads( const int nt ) { return MKL_Set_Num_Threads( nt ); }
#else
inline int slate_set_num_blas_threads( const int nt ) { return -1; }
#endif

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_symm_work( Params &params, bool run )
{
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    //blas::Op transA = params.transA.value();
    //blas::Op transB = params.transB.value();
    blas::Side side = params.side.value();
    blas::Uplo uplo = params.uplo.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    scalar_t alpha = params.alpha.value();
    scalar_t beta = params.beta.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t nb = params.nb.value();
    int64_t lookahead = params.lookahead.value();
    bool check = params.check.value()=='y';
    bool ref = params.ref.value()=='y';
    bool trace = params.trace.value()=='y';
    slate::Target target = char2target(params.target.value());

    // mark non-standard output values
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    if( ! run )
        return;

    // for now, symm on Devices requires full tiles
    if( target == slate::Target::Devices ) {
        assert( m % nb == 0 );
        assert( n % nb == 0 );
    }

    // sizes of data
    int64_t An = ( side == blas::Side::Left ? m : n );
    int64_t Am = An;
    int64_t Bm = m;
    int64_t Bn = n;
    int64_t Cm = m;
    int64_t Cn = n;

    // local values
    static int i0=0, i1=1;

    // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    int descA_tst[9], descB_tst[9], descC_tst[9], descC_ref[9];
    int iam=0, nprocs=1;
    int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo( &iam, &nprocs );
    assert( p*q <= nprocs );
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Col", p, q );
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );
    assert( nprow==p && npcol==q );

    // matrix A, figure out local size, allocate, create descriptor, initialize
    int64_t mlocA = scalapack_numroc( Am, nb, myrow, i0, nprow );
    int64_t nlocA = scalapack_numroc( An, nb, mycol, i0, npcol );
    scalapack_descinit( descA_tst, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info );
    assert( info==0 );
    int64_t lldA = ( int64_t )descA_tst[8];
    std::vector< scalar_t > A_tst( lldA * nlocA );
    scalapack_pplghe( &A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1 );

    // matrix B, figure out local size, allocate, create descriptor, initialize
    int64_t mlocB = scalapack_numroc( Bm, nb, myrow, i0, nprow );
    int64_t nlocB = scalapack_numroc( Bn, nb, mycol, i0, npcol );
    scalapack_descinit( descB_tst, Bm, Bn, nb, nb, i0, i0, ictxt, mlocB, &info );
    assert( info==0 );
    int64_t lldB = ( int64_t )descB_tst[8];
    std::vector< scalar_t > B_tst( lldB * nlocB );
    scalapack_pplrnt( &B_tst[0], Bm, Bn, nb, nb, myrow, mycol, nprow, npcol, mlocB, iseed+1 );

    // matrix C, figure out local size, allocate, create descriptor, initialize
    int64_t mlocC = scalapack_numroc( Cm, nb, myrow, i0, nprow );
    int64_t nlocC = scalapack_numroc( Cn, nb, mycol, i0, npcol );
    scalapack_descinit( descC_tst, Cm, Cn, nb, nb, i0, i0, ictxt, mlocC, &info );
    assert( info==0 );
    int64_t lldC = ( int64_t )descC_tst[8];
    std::vector< scalar_t > C_tst( lldC * nlocC );
    scalapack_pplrnt( &C_tst[0], Cm, Cn, nb, nb, myrow, mycol, nprow, npcol, mlocC, iseed+1 );

    // if check is required, copy test data and create a descriptor for it
    std::vector< scalar_t > C_ref;
    if( check || ref ) {
        C_ref.resize( C_tst.size() );
        C_ref = C_tst;
        scalapack_descinit( descC_ref, Cm, Cn, nb, nb, i0, i0, ictxt, mlocC, &info );
        assert( info==0 );
    }

    // create SLATE matrices from the ScaLAPACK layouts
    auto A = slate::SymmetricMatrix<scalar_t>::fromScaLAPACK( uplo, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK( Bm, Bn, &B_tst[0], lldB, nb, nprow, npcol, MPI_COMM_WORLD );
    auto C = slate::Matrix<scalar_t>::fromScaLAPACK( Cm, Cn, &C_tst[0], lldC, nb, nprow, npcol, MPI_COMM_WORLD );

    if( side == blas::Side::Left )
        assert( A.mt() == C.mt() );
    else
        assert( A.mt() == C.nt() );
    assert( B.mt() == C.mt() );
    assert( B.nt() == C.nt() );

    if( trace ) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // Call the routine using ScaLAPACK layout
    MPI_Barrier( MPI_COMM_WORLD );
    double time = libtest::get_wtime();

    slate::symm( side, alpha, A, B, beta, C, {
            {slate::Option::Lookahead, lookahead},
            {slate::Option::Target, target}} );

    MPI_Barrier( MPI_COMM_WORLD );
    double time_tst = libtest::get_wtime() - time;

    if( trace ) slate::trace::Trace::finish();

    // Compute and save timing/performance
    double gflop = blas::Gflop< scalar_t >::symm( side, n, n );
    params.time.value() = time_tst;
    params.gflops.value() = gflop / time_tst;

    params.okay.value() = true;

    if( check || ref ) {
        // comparison with reference routine from ScaLAPACK

        // set MKL num threads appropriately for parallel BLAS
        int omp_num_threads;
        #pragma omp parallel
        { omp_num_threads = omp_get_num_threads(); }
        int saved_num_threads = slate_set_num_blas_threads( omp_num_threads );

        // Run the reference routine
        MPI_Barrier( MPI_COMM_WORLD );
        double time = libtest::get_wtime();
        scalapack_psymm( side2str( side ), uplo2str( uplo ), m, n, alpha,
                         &A_tst[0], i1, i1, descA_tst,
                         &B_tst[0], i1, i1, descB_tst, beta,
                         &C_ref[0], i1, i1, descC_ref );
        MPI_Barrier( MPI_COMM_WORLD );
        double time_ref = libtest::get_wtime() - time;

        // allocate work space
        std::vector< real_t > worklange( mlocC );

        // Local operation: error = C_ref - C_tst
        blas::axpy( C_ref.size(), -1.0, &C_tst[0], 1, &C_ref[0], 1 );

        // norm(C_tst)
        real_t C_tst_norm = scalapack_plange( "I", m, n, &C_tst[0], i1, i1, descC_tst, &worklange[0] );

        // norm(C_ref - C_tst)
        real_t error_norm = scalapack_plange( "I", m, n, &C_ref[0], i1, i1, descC_ref, &worklange[0] );

        if( C_tst_norm != 0 )
            error_norm /=  C_tst_norm;

        params.ref_time.value() = time_ref;
        params.ref_gflops.value() = gflop / time_ref;
        params.error.value() = error_norm;

        slate_set_num_blas_threads( saved_num_threads );

        real_t eps = std::numeric_limits< real_t >::epsilon();
        params.okay.value() = ( params.error.value() <= 50*eps );
    }

    //Cblacs_exit(1) is commented out because it does not handle re-entering ... some unknown problem
    //Cblacs_exit(1); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_symm( Params &params, bool run )
{
    switch( params.datatype.value() ) {
    case libtest::DataType::Integer:
        throw std::exception();
        break;

    case libtest::DataType::Single:
        test_symm_work< float >( params, run );
        break;

    case libtest::DataType::Double:
        test_symm_work< double >( params, run );
        break;

    case libtest::DataType::SingleComplex:
        test_symm_work< std::complex<float> >( params, run );
        break;

    case libtest::DataType::DoubleComplex:
        test_symm_work< std::complex<double> >( params, run );
        break;
    }
}
