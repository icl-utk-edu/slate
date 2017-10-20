
#ifndef SLATE_NO_OPENMP_HH
#define SLATE_NO_OPENMP_HH

#include <sys/time.h>

typedef int omp_lock_t;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
int omp_get_initial_device()
{
    return -10;
}

//------------------------------------------------------------------------------
int omp_get_max_threads()
{
    return 1;
}

//------------------------------------------------------------------------------
int omp_get_thread_num(void)
{
    return 0;
}

//------------------------------------------------------------------------------
double omp_get_wtime()
{
    struct timeval  time;
    struct timezone zone;

    gettimeofday(&time, &zone);

    double sec = time.tv_sec;
    double usec = time.tv_usec;

    return sec + usec/1000000.0;
}

//------------------------------------------------------------------------------
void omp_init_lock(omp_lock_t *lock)
{
    return;
}

//------------------------------------------------------------------------------
void omp_set_lock(omp_lock_t *lock)
{
    return;
}

//------------------------------------------------------------------------------
void omp_unset_lock(omp_lock_t *lock)
{
    return;
}

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_OPENMP_HH
