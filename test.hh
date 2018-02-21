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

#include <iostream>
#include <iomanip>

// -----------------------------------------------------------------------------
// global variables
namespace slate {
int  g_mpi_rank;
bool g_verbose;
}

int mpi_rank;
int mpi_size;
MPI_Comm mpi_comm;

int num_devices = 1;  // todo: should be omp_get_num_devices
int host_num    = omp_get_initial_device();

// -----------------------------------------------------------------------------
// Prints test name at start and end of test.
// Since destructors are called in reverse order of constructors,
// putting this first ensures that its destructor is called last when exiting
// the test function.
class Test {
public:
    // -------------------------------------------------------------------------
    Test( const char* msg ):
        msg_( msg )
    {
        if (mpi_rank == 0) {
            std::cout << "----- " << msg_ << "\n" << std::flush;
        }
        MPI_Barrier( mpi_comm );
    }

    // -------------------------------------------------------------------------
    ~Test()
    {
        std::cout << std::flush;
        MPI_Barrier( mpi_comm );

        if (mpi_rank == 0) {
            std::cout << "----- " << msg_ << " done\n\n" << std::flush;
        }
        MPI_Barrier( mpi_comm );
    }

    const char* msg_;
};

// -----------------------------------------------------------------------------
// suppresses compiler warning if var is unused
#define unused( var ) \
    ((void)var)

// -----------------------------------------------------------------------------
// similar to assert(), but also prints out MPI rank.
#define test_assert( cond ) \
    do { \
        if (! (cond)) { \
            std::cerr << "rank " << slate::g_mpi_rank \
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
            std::cerr << "rank " << slate::g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": did not throw expected exception\n"; \
        } \
        catch( exception& e ) {} \
        catch( ... ) { \
            std::cerr << "rank " << slate::g_mpi_rank \
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
            std::cerr << "rank " << slate::g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": unexpected exception thrown\n"; \
        } \
    } while(0)

#endif        //  #ifndef TEST_HH
