#ifndef TEST_HH
#define TEST_HH

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

#include "slate_Matrix.hh"

#include <iostream>
#include <iomanip>

// -----------------------------------------------------------------------------
// global variables
MPI_Comm g_mpi_comm;
int  g_mpi_rank;
int  g_mpi_size;
bool g_verbose;
int  g_num_devices = omp_get_num_devices();
int  g_host_num    = omp_get_initial_device();

// -----------------------------------------------------------------------------
// Prints test name at start and end of test.
// Since destructors are called in reverse order of constructors,
// putting this first ensures that its destructor is called last when exiting
// the test function.
class Test {
public:
    // ----------------------------------------
    Test( const char* msg ):
        msg_( msg )
    {
        if (g_mpi_rank == 0) {
            std::cout << "---------- " << msg_ << "\n" << std::flush;
        }
        MPI_Barrier( g_mpi_comm );
    }

    // ----------------------------------------
    ~Test()
    {
        std::cout << std::flush;
        MPI_Barrier( g_mpi_comm );

        if (g_mpi_rank == 0) {
            std::cout << "---------- " << msg_ << " done\n\n" << std::flush;
        }
        MPI_Barrier( g_mpi_comm );
    }

    const char* msg_;
};

// -----------------------------------------------------------------------------
// Does barrier, then prints label on rank 0 for next sub-test.
void test_barrier( const char* msg )
{
    MPI_Barrier( g_mpi_comm );
    if (g_mpi_rank == 0) {
        std::cout << "-- " << msg << "\n";
    }
}

// -----------------------------------------------------------------------------
// suppresses compiler warning if var is unused
#define unused( var ) \
    ((void)var)

// -----------------------------------------------------------------------------
// similar to assert(), but also prints out MPI rank.
#define test_assert( cond ) \
    do { \
        if (! (cond)) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": " \
                      << #cond << "\n"; \
            exit(1); \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// executes expr; asserts that the given exception was thrown.
#define test_assert_throw( expr, exception ) \
    do { \
        try { \
            expr; \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": did not throw expected exception\n"; \
        } \
        catch( exception& e ) {} \
        catch( ... ) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": wrong exception thrown\n"; \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// executes expr; asserts that no exception was thrown.
#define test_assert_no_throw( expr ) \
    do { \
        try { \
            expr; \
        } \
        catch( ... ) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": unexpected exception thrown\n"; \
        } \
    } while(0)

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::Matrix< scalar_t >& A )
{
    using blas::real;

    printf( "[\n" );
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t jb = A.tileNb(i);

            if (A.tileIsLocal( i, j )) {
                if (j > 0)
                    printf( "   " );
                auto Aij = A( i, j );
                printf( "  %-18p", (void*) Aij.data() );
                for (int64_t jj = 2; jj < jb; ++jj) { // above pointer is 2 columns
                    printf( " %9s", "" );
                }
            }
            else {
                printf( " ... " );
            }
        }
        printf( "\n" );

        int64_t ib = A.tileMb(i);
        for (int64_t ii = 0; ii < ib; ++ii) {

            // loop over block cols, then cols within block col
            for (int64_t j = 0; j < A.nt(); ++j) {
                int64_t jb = A.tileNb(i);

                if (A.tileIsLocal( i, j )) {
                    if (j > 0)
                        printf( "   " );
                    auto Aij = A( i, j );
                    for (int64_t jj = 0; jj < jb; ++jj) {
                        printf( " %9.4f", real( Aij( ii, jj ) ) );
                    }
                }
                else {
                    printf( " ... " );
                }
            }
            printf( "\n" );
        }
        if (i < A.mt()-1) {
            printf( "\n" );
        }
    }
    printf( "];\n" );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::BaseTrapezoidMatrix< scalar_t >& A )
{
    using blas::real;

    assert( A.uplo() == blas::Uplo::Lower );

    printf( "[\n" );
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        for (int64_t j = 0; j <= i && j < A.nt(); ++j) {  // lower
            int64_t jb = A.tileNb(i);

            if (A.tileIsLocal( i, j )) {
                if (j > 0)
                    printf( "   " );
                auto Aij = A( i, j );
                printf( "  %-18p", (void*) Aij.data() );
                for (int64_t jj = 2; jj < jb; ++jj) {
                    printf( " %9s", "" );
                }
            }
            else {
                printf( " ... " );
            }
        }
        printf( "\n" );

        int64_t ib = A.tileMb(i);
        for (int64_t ii = 0; ii < ib; ++ii) {

            // loop over block cols, then cols within block col
            for (int64_t j = 0; j <= i && j < A.nt(); ++j) {  // lower
                int64_t jb = A.tileNb(i);

                if (A.tileIsLocal( i, j )) {
                    if (j > 0)
                        printf( "   " );
                    auto Aij = A( i, j );
                    for (int64_t jj = 0; jj < jb; ++jj) {
                        printf( " %9.4f", real( Aij( ii, jj ) ) );
                    }
                }
                else {
                    printf( " ... " );
                }
            }
            printf( "\n" );
        }
        if (i < A.mt()-1) {
            printf( "\n" );
        }
    }
    printf( "];\n" );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( int64_t m, int64_t n, scalar_t* A, int64_t lda )
{
    using blas::real;

    printf( "[\n" );
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf( " %9.4f", real( A[ i + j*lda ] ) );
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

#endif        //  #ifndef TEST_HH
