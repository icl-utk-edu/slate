#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_trace_Trace.hh"

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

#ifdef _OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_potrf_lapack_work( Params& params, bool run )
{
    using real_t = blas::real_type<scalar_t>;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t align = params.align.value();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t lookahead = params.lookahead.value();

    int64_t nb = params.nb.value();
    int64_t nt = params.nt.value();

    int64_t n = params.dim.n();
    n = nb*nt;
    params.dim.n() = nb*nt;

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();
    params.gflops.value();

    // slate::Debug::off();

    if (! run)
        return;

    // MPI initializations
    int mpi_rank, mpi_size, retval;
    retval = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    assert(retval == MPI_SUCCESS);

    retval = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    assert(retval == MPI_SUCCESS);
    assert(mpi_size == p*q);

    //---------------------
    int64_t lda = roundup( n , align );

    double *A_tst = nullptr;
    double *A_ref = nullptr;

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    A_tst = new double[ lda*n ];
    lapack::larnv( idist, iseed, lda*n, A_tst );
    // Make diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) 
        A_tst[ i + i*lda ] += sqrt(n);

    // Copy A_tst for reference run
    if ( (params.ref.value() == 'y') && (mpi_rank == 0) ) {
        A_ref = new double[ lda*n ];
        memcpy(A_ref, A_tst, sizeof(double)*lda*n);
    }

    // Create slate matrix
    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, A_tst, lda,
                                     nb, p, q, MPI_COMM_WORLD);
    slate::trace::Trace::on();
    slate::trace::Block trace_block("MPI_Barrier");
    MPI_Barrier(MPI_COMM_WORLD);

    double start = omp_get_wtime();

    // Call routine
    slate::potrf(A,
                {{slate::Option::Lookahead, lookahead},
                 {slate::Option::Target, slate::Target::HostTask}});

    MPI_Barrier(MPI_COMM_WORLD);
    double time = omp_get_wtime() - start;
    slate::trace::Trace::finish();

    // Record gflops.
    double gflop = lapack::Gflop< scalar_t >::potrf( n );
    params.time.value() = time;
    params.gflops.value() = gflop / time;
        
    //------------------
    // Test correctness.
    if ( params.ref.value() == 'y' || params.check.value() == 'y' ) {

        // ---------- run reference
        // A.gather( &A_tst[0], lda );
        if (mpi_rank == 0) {
            libtest::flush_cache( params.cache.value() );

            double time = libtest::get_wtime();
            int64_t info_trf = lapack::potrf(uplo, n, A_ref, lda);
            time = libtest::get_wtime() - time;
            if (info_trf != 0) 
                fprintf( stderr, "potrf returned error during correctness check %lld\n", (lld) info_trf );
            params.ref_time.value() = time;
            params.ref_gflops.value() = gflop / time;

            // check error compared to reference
            real_t eps = std::numeric_limits< real_t >::epsilon();
            // A.copyFromFull(A_tst, lda);
            // error += abs_error( A_tst, A_ref );
            blas::axpy((size_t)lda*n, -1.0, A_tst, 1, A_ref, 1);
            real_t norm = lapack::lansy(lapack::Norm::Fro, uplo, n, A_tst, lda);
            real_t error = lapack::lange(lapack::Norm::Fro, n, n, A_ref, lda);
            if (norm != 0)
                error /= norm;

            if ( error > 3*eps )
                slate::Debug::diffLapackMatrices(n, n, A_tst, lda, A_ref, lda, nb, nb);
            
            params.error.value() = error;
            params.okay.value() = (error <= 3*eps);
        }
    }
    if ( (params.ref.value() == 'y') && (mpi_rank == 0) ) {
        delete[] A_ref;
        A_ref = nullptr;
    }
    delete[] A_tst;
    A_tst = nullptr;
}



// -----------------------------------------------------------------------------
void test_potrf_lapack( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            throw std::exception();// test_potrf_lapack_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_potrf_lapack_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            throw std::exception();// test_potrf_lapack_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            throw std::exception();// test_potrf_lapack_work< std::complex<double> >( params, run );
            break;
    }
}
