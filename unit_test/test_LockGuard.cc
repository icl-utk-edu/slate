#include "slate/internal/LockGuard.hh"
#include "slate/internal/openmp.hh"

#include "unit_test.hh"

#include <unistd.h>

//------------------------------------------------------------------------------
void test_LockGuard()
{
    omp_nest_lock_t lock;
    omp_init_nest_lock( &lock );

    int sum = 0;
    int n = 20;

    #pragma omp parallel
    #pragma omp master
    {
        for (int i = 1; i <= n; ++i) {
            #pragma omp task
            {
                slate::LockGuard guard( &lock );
                // Make a race condition: read sum, pause, update sum.
                // Without the guard, the sum is usually wrong.
                int x = sum;
                usleep( 100 );
                sum = x + i;
            }
        }
    }
    //printf( "    %s sum %d\n", __func__, sum );
    test_assert( sum == n*(n + 1)/2 );
    omp_destroy_nest_lock( &lock );
}

//------------------------------------------------------------------------------
void inner( int i, int* sum, omp_nest_lock_t* lock )
{
    slate::LockGuard guard( lock );
    //printf( "    %s %d\n", __func__, i );
    int x = *sum;
    usleep( 100 );
    *sum = x + i;
}

//------------------------------------------------------------------------------
void outer( int i, int* sum, omp_nest_lock_t* lock )
{
    slate::LockGuard guard( lock );
    //printf( "    %s %d\n", __func__, i );
    inner( i, sum, lock );
}

//------------------------------------------------------------------------------
void test_nested()
{
    omp_nest_lock_t lock;
    omp_init_nest_lock( &lock );

    int sum = 0;
    int n = 20;

    #pragma omp parallel
    #pragma omp master
    {
        for (int i = 1; i <= n; ++i) {
            #pragma omp task
            outer( i, &sum, &lock );
        }
    }
    //printf( "    %s sum %d\n", __func__, sum );
    test_assert( sum == n*(n + 1)/2 );
    omp_destroy_nest_lock( &lock );
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    run_test(test_LockGuard, "LockGuard()");
    run_test(test_nested,    "LockGuard() nested");
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    //printf( "main\n" );
    return unit_test_main();  // which calls run_tests()
}
