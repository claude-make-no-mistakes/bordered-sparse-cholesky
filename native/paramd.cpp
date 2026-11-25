#include "paramd/paramd.h"
#include <omp.h>

extern "C" __declspec(dllexport)
int wh_paramd(int n, int const* pointer, int const* index, int* permutation, int threads)
{
    paramd::config config;
    config.sym = true;
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    int status = 0;
    try {
        paramd::paramd(n, pointer, index, permutation, config);
    } catch (...) {
        status = 1;
    }
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
    return status;
}
