#ifndef WORKHORSES_SPARSE_ITERATIVE_H
#define WORKHORSES_SPARSE_ITERATIVE_H

#include <float.h>

/* Symmetric upper CSC is lower CSR with the same arrays. The incomplete
   factor preserves this pattern and stores the inverse of each diagonal. */
API int wh_ic0(int n, const int32_t *row, const int32_t *column, const double *value,
               double diagonal_shift, double *factor)
{
    for (int i = 0; i < n; ++i) {
        int start = row[i], last = row[i + 1] - 1;
        if (last < start || column[last] != i) return 0;
        double diagonal = value[last] * (1 + diagonal_shift);
        for (int p = start; p < last; ++p) {
            int j = column[p];
            double entry = value[p];
            for (int a = start, b = row[j]; a < p && b < row[j + 1] - 1;) {
                if (column[a] == column[b]) {
                    entry -= factor[a++] * factor[b++];
                } else if (column[a] < column[b]) {
                    ++a;
                } else {
                    ++b;
                }
            }
            factor[p] = entry * factor[row[j + 1] - 1];
            diagonal -= factor[p] * factor[p];
        }
        if (!(diagonal > 64 * DBL_EPSILON * value[last]) || !isfinite(diagonal)) return 0;
        factor[last] = 1 / sqrt(diagonal);
    }
    return 1;
}

static double wh_precondition(int n, const int32_t *row, const int32_t *column, const double *factor,
                              const double *residual, double *z)
{
    for (int i = 0; i < n; ++i) {
        double value = residual[i];
        int last = row[i + 1] - 1;
        for (int j = row[i]; j < last; ++j) value -= factor[j] * z[column[j]];
        z[i] = value * factor[last];
    }
    double energy = 0;
    for (int i = n - 1; i >= 0; --i) {
        int last = row[i + 1] - 1;
        double value = z[i] * factor[last];
        z[i] = value;
        energy += value * residual[i];
        for (int j = row[i]; j < last; ++j) z[column[j]] -= factor[j] * value;
    }
    return energy;
}

static void wh_spmv(int n, const int32_t *row, const int32_t *column, const double *values,
                    int sites, const int32_t *where, const double *shifts,
                    const double *x, double *product)
{
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        for (int j = row[i]; j < row[i + 1]; ++j) sum += values[j] * x[column[j]];
        product[i] = sum;
    }
    for (int i = 0; i < sites; ++i) product[where[i]] += shifts[i] * x[where[i]];
}

API int wh_pcg(int n, const int32_t *row, const int32_t *column, const double *values,
               const int32_t *tri_row, const int32_t *tri_col, const double *factor,
               int sites, const int32_t *where, const double *shifts,
               const double *rhs, double matrix_norm, int limit, double tolerance,
               double *solution, double *space)
{
    double *r = space, *z = space + n, *direction = space + 2 * (size_t) n;
    double *product = space + 3 * (size_t) n;
    double rhs_norm = 0;
    for (int i = 0; i < n; ++i) {
        r[i] = rhs[i];
        solution[i] = 0;
        if (fabs(rhs[i]) > rhs_norm) rhs_norm = fabs(rhs[i]);
    }
    if (rhs_norm == 0) return 0;
    double rho = wh_precondition(n, tri_row, tri_col, factor, r, z);
    memcpy(direction, z, n * sizeof(*direction));
    for (int step = 1; step <= limit; ++step) {
        wh_spmv(n, row, column, values, sites, where, shifts, direction, product);
        double energy = 0;
        for (int i = 0; i < n; ++i) energy += direction[i] * product[i];
        if (!(energy > 0) || !(rho > 0) || !isfinite(energy) || !isfinite(rho)) return -1;
        double alpha = rho / energy;
        double rnorm = 0, xnorm = 0;
        for (int i = 0; i < n; ++i) {
            solution[i] += alpha * direction[i];
            r[i] -= alpha * product[i];
            if (fabs(r[i]) > rnorm) rnorm = fabs(r[i]);
            if (fabs(solution[i]) > xnorm) xnorm = fabs(solution[i]);
        }
        double scale = matrix_norm * xnorm + rhs_norm;
        if (!isfinite(scale)) return -1;
        int refresh = rnorm <= tolerance * scale;
        if (refresh) {
            wh_spmv(n, row, column, values, sites, where, shifts, solution, product);
            rnorm = 0;
            for (int i = 0; i < n; ++i) {
                r[i] = rhs[i] - product[i];
                if (!isfinite(r[i])) return -1;
                if (fabs(r[i]) > rnorm) rnorm = fabs(r[i]);
            }
            if (rnorm <= tolerance * scale) {
                wh_precondition(n, tri_row, tri_col, factor, r, z);
                double correction = 0;
                for (int i = 0; i < n; ++i) {
                    if (!isfinite(z[i])) return -1;
                    if (fabs(z[i]) > correction) correction = fabs(z[i]);
                }
                if (isfinite(correction) && correction <= 1e-9 * xnorm) return step;
            }
        }
        double next_rho = wh_precondition(n, tri_row, tri_col, factor, r, z);
        double beta = refresh ? 0 : next_rho / rho;
        for (int i = 0; i < n; ++i) direction[i] = z[i] + beta * direction[i];
        rho = next_rho;
    }
    return -1;
}

/* A higher-precision residual supports one correction after switching to
   mutable direct factors on difficult shift sequences. */
API int wh_extended_residual(int n, const int32_t *column, const int32_t *row, const double *value,
                             size_t count, int sites, const int32_t *where, const double *shifts,
                             const double *rhs, const double *answer, double *out)
{
    long double *accum = malloc((size_t) n * sizeof(*accum));
    if (!accum) return 0;
    for (size_t rhs_index = 0; rhs_index < count; ++rhs_index) {
        for (int i = 0; i < n; ++i) accum[i] = rhs[i];
        for (int j = 0; j < n; ++j) {
            long double x = answer[j];
            for (int p = column[j]; p < column[j + 1]; ++p) accum[row[p]] -= (long double) value[p] * x;
        }
        for (int k = 0; k < sites; ++k) accum[where[k]] -= (long double) shifts[k] * answer[where[k]];
        for (int i = 0; i < n; ++i) out[i] = (double) accum[i];
        rhs += n;
        answer += n;
        out += n;
    }
    free(accum);
    return 1;
}
#endif
