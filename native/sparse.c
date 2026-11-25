#include "amd_policy.h"
#include "amd.h"
#include "cholmod.h"
#include "colamd.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define API __declspec(dllexport)

#include "sparse_iterative.h"

extern int32_t wh_amd_order(int32_t n, const int32_t *ap, const int32_t *ai,
                            int32_t *p, double *control, double *info);

static double now(void)
{
    LARGE_INTEGER clock, frequency;
    QueryPerformanceCounter(&clock);
    QueryPerformanceFrequency(&frequency);
    return (double) clock.QuadPart / (double) frequency.QuadPart;
}

API int wh_order(int n, const int32_t *column, const int32_t *row, int32_t *output,
                 double dense, int aggressive, int candidates, int degree, int cap,
                 int use_midpoint, wh_pivot_stats *stats)
{
    double control[AMD_CONTROL];
    amd_defaults(control);
    control[AMD_DENSE] = dense;
    control[AMD_AGGRESSIVE] = aggressive;
    memset(stats, 0, sizeof(*stats));
    if (!candidates) return amd_order(n, column, row, output, control, NULL);
    int32_t *space = NULL;
    if (candidates > 1) {
        space = calloc(3 * (size_t) n, sizeof(*space));
        if (!space) return AMD_OUT_OF_MEMORY;
    }
    wh_pivot_policy policy = {
        candidates, degree, cap, use_midpoint,
        space, space ? space + n : NULL, space ? space + 2 * n : NULL, 0, stats
    };
    wh_current_policy = &policy;
    int result = wh_amd_order(n, column, row, output, control, NULL);
    wh_current_policy = NULL;
    free(space);
    return result;
}

API int wh_colamd(int n, const int32_t *column, const int32_t *row, int32_t *output,
                  double row_density, double column_density)
{
    double knobs[COLAMD_KNOBS];
    int32_t stats[COLAMD_STATS], nnz = column[n];
    size_t count = colamd_recommended(nnz, n, n);
    if (!count || count > INT32_MAX) return COLAMD_ERROR_out_of_memory;
    int32_t *work = malloc(count * sizeof(*work));
    int32_t *p = malloc((n + 1) * sizeof(*p));
    if (!work || !p) {
        free(work);
        free(p);
        return COLAMD_ERROR_out_of_memory;
    }
    memcpy(work, row, nnz * sizeof(*work));
    memcpy(p, column, (n + 1) * sizeof(*p));
    colamd_set_defaults(knobs);
    knobs[COLAMD_DENSE_ROW] = row_density;
    knobs[COLAMD_DENSE_COL] = column_density;
    int ok = colamd(n, n, (int32_t) count, work, p, knobs, stats);
    if (ok) memcpy(output, p, n * sizeof(*output));
    free(work);
    free(p);
    return stats[COLAMD_STATUS];
}

typedef struct {
    double analysis_seconds, factor_seconds, factor_nnz, flops, stored_entries, memory_peak;
    int permutation_match, status;
} wh_factor_stats;

API int wh_cholesky(int n, const int32_t *column, const int32_t *row, const double *values,
                    double *rhs, const int32_t *permutation, double *solution, wh_factor_stats *stats)
{
    cholmod_common config;
    if (!cholmod_start(&config)) return 0;
    config.print = 0;
    config.nmethods = 1;
    config.method[0].ordering = CHOLMOD_GIVEN;
    config.postorder = 0;
    config.nthreads_max = 1;
    cholmod_sparse matrix = {
        .nrow = n, .ncol = n, .nzmax = (size_t) column[n],
        .p = (void *) column, .i = (void *) row, .x = (void *) values,
        .stype = 1, .itype = CHOLMOD_INT, .xtype = CHOLMOD_REAL, .dtype = CHOLMOD_DOUBLE,
        .sorted = 1, .packed = 1
    };
    cholmod_dense b = {.nrow = n, .ncol = 1, .nzmax = n, .d = n, .x = rhs,
                       .xtype = CHOLMOD_REAL, .dtype = CHOLMOD_DOUBLE};
    memset(stats, 0, sizeof(*stats));
    double start = now();
    cholmod_factor *factor = cholmod_analyze_p(&matrix, (int32_t *) permutation, NULL, 0, &config);
    stats->analysis_seconds = now() - start;
    stats->factor_nnz = config.lnz;
    stats->flops = config.fl;
    int ok = 0;
    if (factor) {
        stats->permutation_match = !memcmp(factor->Perm, permutation, n * sizeof(*permutation));
        start = now();
        ok = stats->permutation_match && cholmod_factorize(&matrix, factor, &config) && factor->minor == (size_t) n;
        stats->factor_seconds = now() - start;
        stats->stored_entries = factor->is_super ? (double) factor->xsize : (double) factor->nzmax;
    }
    stats->status = config.status;
    stats->memory_peak = (double) config.memory_usage;
    if (ok) {
        cholmod_dense *answer = cholmod_solve(CHOLMOD_A, factor, &b, &config);
        ok = answer != NULL;
        if (answer) memcpy(solution, answer->x, n * sizeof(*solution));
        cholmod_free_dense(&answer, &config);
    }
    cholmod_free_factor(&factor, &config);
    cholmod_finish(&config);
    return ok;
}

typedef struct {
    cholmod_common config;
    cholmod_sparse *pattern;
    cholmod_factor *factor;
    int ready;
    struct wh_border *border;
} wh_chol_plan;

API void wh_chol_plan_free(wh_chol_plan *plan)
{
    if (!plan) return;
    free(plan->border);
    cholmod_free_factor(&plan->factor, &plan->config);
    cholmod_free_sparse(&plan->pattern, &plan->config);
    cholmod_finish(&plan->config);
    free(plan);
}

static wh_chol_plan *wh_make_plan(int n, const int32_t *column, const int32_t *row,
                                  int32_t *permutation, int dtype, int method, int postorder,
                                  int wide, int *status)
{
    *status = CHOLMOD_INVALID;
    if (n < 1 || column[0] != 0 || column[n] < 0) return NULL;
    *status = CHOLMOD_OUT_OF_MEMORY;
    wh_chol_plan *plan = calloc(1, sizeof(*plan));
    if (!plan) return NULL;
    if (!cholmod_start(&plan->config)) {
        free(plan);
        return NULL;
    }
    cholmod_common *config = &plan->config;
    config->print = 0;
    config->nmethods = 1;
    config->method[0].ordering = permutation ? CHOLMOD_GIVEN : CHOLMOD_AMD;
    config->postorder = postorder;
    config->nthreads_max = 1;
    config->supernodal = method;
    if (wide == 1) {
        config->nrelax[0] = 8;
        config->nrelax[1] = 32;
        config->nrelax[2] = 128;
    } else if (wide == 2) {
        config->supernodal_switch = 140.;
    }
    size_t nnz = (size_t) column[n];
    plan->pattern = cholmod_allocate_sparse(n, n, nnz, 1, 1, 1, CHOLMOD_PATTERN + dtype, config);
    if (plan->pattern) {
        memcpy(plan->pattern->p, column, ((size_t) n + 1) * sizeof(*column));
        memcpy(plan->pattern->i, row, nnz * sizeof(*row));
        plan->factor = cholmod_analyze_p(plan->pattern, permutation, NULL, 0, config);
    }
    *status = config->status;
    if (plan->factor && permutation && !postorder && memcmp(plan->factor->Perm, permutation, n * sizeof(*permutation)))
        *status = CHOLMOD_INVALID;
    if (!plan->factor || *status != CHOLMOD_OK) {
        wh_chol_plan_free(plan);
        return NULL;
    }
    return plan;
}

API wh_chol_plan *wh_chol_plan_create(int n, const int32_t *column, const int32_t *row,
                                      int32_t *permutation, int *status)
{
    return wh_make_plan(n, column, row, permutation, CHOLMOD_DOUBLE, CHOLMOD_AUTO, 0, 0, status);
}

API wh_chol_plan *wh_chol_plan_create_f32(int n, const int32_t *column, const int32_t *row,
                                          int32_t *permutation, int *status)
{
    return wh_make_plan(n, column, row, permutation, CHOLMOD_SINGLE, CHOLMOD_AUTO, 0, 0, status);
}

API wh_chol_plan *wh_chol_plan_create_tuned(int n, const int32_t *column, const int32_t *row,
                                            int32_t *permutation, int method, int postorder,
                                            int wide, int *status)
{
    return wh_make_plan(n, column, row, permutation, CHOLMOD_DOUBLE, method, postorder, wide, status);
}

API void wh_chol_plan_metrics(const wh_chol_plan *plan, double *stats)
{
    stats[0] = plan->config.fl;
    stats[1] = plan->config.lnz;
    stats[2] = plan->factor->is_super;
    stats[3] = plan->factor->nsuper;
    stats[4] = plan->factor->is_super ? plan->factor->xsize : plan->factor->nzmax;
}

static double wh_pivot(const cholmod_factor *factor, size_t index)
{
    if (factor->dtype == CHOLMOD_SINGLE) return ((const float *) factor->x)[index];
    return ((const double *) factor->x)[index];
}

static int wh_positive_diagonal(const cholmod_factor *factor)
{
    // A simplicial LDL' factor can succeed with negative pivots. Check D
    // explicitly. The supernodal real LL' blocks have column-major layout.
    if (!factor->is_super) {
        const int32_t *ptr = factor->p;
        for (size_t j = 0; j < factor->n; ++j) {
            double diagonal = wh_pivot(factor, ptr[j]);
            if (!(diagonal > 0) || !isfinite(diagonal)) return 0;
        }
    } else {
        const int32_t *columns = factor->super, *rows = factor->pi, *start = factor->px;
        for (size_t node = 0; node < factor->nsuper; ++node) {
            size_t stride = (size_t) (rows[node + 1] - rows[node]) + 1;
            size_t width = (size_t) (columns[node + 1] - columns[node]);
            for (size_t k = 0; k < width; ++k) {
                double diagonal = wh_pivot(factor, start[node] + k * stride);
                if (!(diagonal > 0) || !isfinite(diagonal)) return 0;
            }
        }
    }
    return 1;
}

static int wh_factor_plan(wh_chol_plan *plan, const void *values, int dtype, int *status)
{
    // Borrow only the numeric payload for this call. The plan owns its CSC
    // structure, and CHOLMOD sees the identical pattern for each refactor.
    cholmod_sparse numeric = *plan->pattern;
    numeric.x = (void *) values;
    numeric.xtype = CHOLMOD_REAL;
    numeric.dtype = dtype;
    plan->ready = cholmod_factorize(&numeric, plan->factor, &plan->config) && plan->factor->minor == numeric.ncol;
    *status = plan->config.status;
    if (plan->ready && !wh_positive_diagonal(plan->factor)) {
        plan->ready = 0;
        *status = CHOLMOD_NOT_POSDEF;
    }
    return plan->ready;
}

API int wh_chol_plan_factor(wh_chol_plan *plan, const double *values, int *status)
{
    return wh_factor_plan(plan, values, CHOLMOD_DOUBLE, status);
}

API int wh_chol_plan_factor_f32(wh_chol_plan *plan, const float *values, int *status)
{
    return wh_factor_plan(plan, values, CHOLMOD_SINGLE, status);
}

API void wh_chol_plan_permutation(const wh_chol_plan *plan, int32_t *permutation)
{
    memcpy(permutation, plan->factor->Perm, plan->factor->n * sizeof(*permutation));
}

API int wh_chol_plan_modify(wh_chol_plan *plan, int increase, size_t count,
                           const int32_t *column, const int32_t *sites, const double *root)
{
    if (!plan->ready || count < 1 || count > plan->factor->n || plan->factor->dtype != CHOLMOD_DOUBLE) return 0;
    // Each update column has one nonzero, in the factor's permuted coordinates.
    cholmod_sparse change = {
        .nrow = plan->factor->n, .ncol = count, .nzmax = count,
        .p = (void *) column, .i = (void *) sites, .x = (void *) root,
        .stype = 0, .itype = CHOLMOD_INT, .xtype = CHOLMOD_REAL, .dtype = CHOLMOD_DOUBLE,
        .sorted = 1, .packed = 1
    };
    plan->ready = cholmod_updown(increase, &change, plan->factor, &plan->config)
        && plan->config.status == CHOLMOD_OK && wh_positive_diagonal(plan->factor);
    return plan->ready;
}

static int wh_solve_plan(wh_chol_plan *plan, size_t count, const void *rhs,
                         void *solution, int dtype, size_t unit)
{
    size_t n = plan->pattern->nrow;
    if (!plan->ready || count < 1 || count > SIZE_MAX / n / unit || plan->factor->dtype != dtype) return 0;
    cholmod_dense b = {.nrow = n, .ncol = count, .nzmax = n * count, .d = n,
                       .x = (void *) rhs, .xtype = CHOLMOD_REAL, .dtype = dtype};
    cholmod_dense *answer = cholmod_solve(CHOLMOD_A, plan->factor, &b, &plan->config);
    if (!answer) return 0;
    for (size_t i = 0; i < count; ++i)
        memcpy((char *) solution + i * n * unit, (char *) answer->x + i * answer->d * unit, n * unit);
    cholmod_free_dense(&answer, &plan->config);
    return 1;
}

API int wh_chol_plan_solve(wh_chol_plan *plan, size_t count, const double *rhs, double *solution)
{
    return wh_solve_plan(plan, count, rhs, solution, CHOLMOD_DOUBLE, sizeof(*rhs));
}

API int wh_chol_plan_solve_f32(wh_chol_plan *plan, size_t count, const float *rhs, float *solution)
{
    return wh_solve_plan(plan, count, rhs, solution, CHOLMOD_SINGLE, sizeof(*rhs));
}

#include "sparse_border.h"
