
#ifndef SLATE_NO_OPENMP_HH
#define SLATE_NO_OPENMP_HH

typedef int omp_lock_t;

#ifdef __cplusplus
extern "C" {
#endif

int omp_get_initial_device();
int omp_get_max_threads();
int omp_get_num_devices();
int omp_get_thread_num(void);

double omp_get_wtime();

void omp_destroy_lock(omp_lock_t *lock);
void omp_init_lock(omp_lock_t *lock);
void omp_set_lock(omp_lock_t *lock);
void omp_unset_lock(omp_lock_t *lock);

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_OPENMP_HH
