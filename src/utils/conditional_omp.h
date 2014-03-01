#ifdef _OPENMP
#include <omp.h>
#else
typedef int omp_int_t;
inline void omp_set_num_threads(int num_threads) {}
inline omp_int_t omp_get_thread_num()  {
    return 0;
}
inline omp_int_t omp_get_num_threads() {
    return 1;
}
inline omp_int_t omp_get_max_threads() {
    return 1;
}
#endif
