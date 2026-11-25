#ifndef WORKHORSES_SPARSE_BORDER_H
#define WORKHORSES_SPARSE_BORDER_H

/* The changing variables are a suffix of the elimination order. The leading
   factor and its rectangular coupling to the suffix are fixed for every RHS. */
#define WH_BORDER_MAX 32
typedef struct wh_border {
    int k, refine;
    size_t slot[WH_BORDER_MAX * WH_BORDER_MAX];
    long double schur[WH_BORDER_MAX * WH_BORDER_MAX];
    double norm;
    double row_norm[];
} wh_border;

API int wh_border_prepare(wh_chol_plan *plan, int k, const int32_t *sites, const double *values)
{
    if (!plan || !plan->ready || plan->border || k < 1 || k > WH_BORDER_MAX) return 0;
    cholmod_factor *f = plan->factor;
    int n = (int) f->n, first = n - k;
    if (first < 0 || f->dtype != CHOLMOD_DOUBLE || f->itype != CHOLMOD_INT) return 0;
    const int32_t *perm = f->Perm, *ap = plan->pattern->p, *ai = plan->pattern->i;
    if (memcmp(perm + first, sites, k * sizeof(*sites))) return 0;
    wh_border *b = calloc(1, sizeof(*b) + (size_t) n * sizeof(double));
    if (!b) return 0;
    b->k = k;
    for (int i = 0; i < k * k; ++i) b->slot[i] = SIZE_MAX;
    const double *fx = f->x;
    int node = 0;
    for (int j = 0; j < n; ++j) {
        int start, end, p0;
        const int32_t *rows;
        if (f->is_super) {
            const int32_t *super = f->super, *pi = f->pi, *px = f->px;
            while (j >= super[node + 1]) ++node;
            int stride = pi[node + 1] - pi[node], col = j - super[node];
            start = pi[node] + col;
            end = pi[node + 1];
            p0 = px[node] + col * stride + col;
            rows = f->s;
        } else {
            const int32_t *fp = f->p, *nz = f->nz;
            start = p0 = fp[j];
            end = start + nz[j];
            rows = f->i;
        }
        if (start >= end || rows[start] != j) goto invalid;
        int original = perm[j], last = ap[original + 1] - 1;
        if (last < ap[original] || ai[last] != original) goto invalid;
        long double diag = fx[p0];
        long double pivot = f->is_ll ? diag * diag : diag;
        if (pivot < 1e-8L * values[last]) b->refine = 1;
        if (j >= first) {
            for (int p = start; p < end; ++p)
                b->slot[(rows[p] - first) * k + j - first] = (size_t) p0 + (p - start);
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int p = ap[j]; p < ap[j + 1]; ++p) {
            b->row_norm[j] += fabs(values[p]);
            if (ai[p] != j) b->row_norm[ai[p]] += fabs(values[p]);
        }
    }
    for (int j = 0; j < n; ++j) b->norm = fmax(b->norm, b->row_norm[j]);
    for (int j = 0; j < k; ++j) {
        for (int i = j; i < k; ++i) {
            long double sum = 0;
            for (int t = 0; t <= j; ++t) {
                size_t a = b->slot[i*k+t], c = b->slot[j*k+t];
                if (a == SIZE_MAX || c == SIZE_MAX) continue;
                long double l = (!f->is_ll && i == t) ? 1 : fx[a];
                long double r = (!f->is_ll && j == t) ? 1 : fx[c];
                sum += l * r * (f->is_ll ? 1 : fx[b->slot[t*k+t]]);
            }
            b->schur[i*k+j] = sum;
        }
    }
    plan->border = b;
    return 1;
invalid:
    free(b);
    return 0;
}

static int wh_border_shift(wh_chol_plan *plan, const double *shift)
{
    wh_border *b = plan->border;
    int k = b->k;
    long double l[WH_BORDER_MAX * WH_BORDER_MAX] = {0};
    for (int j = 0; j < k; ++j) {
        for (int i = j; i < k; ++i) {
            if (b->slot[i*k+j] == SIZE_MAX) continue;
            long double v = b->schur[i*k+j];
            if (i == j) v += shift[j];
            for (int h = 0; h < j; ++h) v -= l[i*k+h] * l[j*k+h];
            if (i == j && !(v > 0)) return 0;
            l[i*k+j] = i == j ? sqrtl(v) : v / l[j*k+j];
        }
    }
    double *fx = plan->factor->x;
    for (int j = 0; j < k; ++j) {
        for (int i = j; i < k; ++i) {
            size_t at = b->slot[i*k+j];
            if (at == SIZE_MAX) continue;
            long double v = l[i*k+j];
            if (!plan->factor->is_ll) v = i == j ? v*v : v/l[j*k+j];
            fx[at] = (double) v;
            if (!isfinite(fx[at]) || (i == j && !(fx[at] > 0))) return 0;
        }
    }
    return 1;
}

API int wh_border_solve(wh_chol_plan *plan, const double *values, const int32_t *sites,
                        const double *shift, size_t count, const double *rhs,
                        double *solution, double *error)
{
    if (!plan || !plan->border || !plan->ready || !wh_border_shift(plan, shift)) return 0;
    if (!wh_solve_plan(plan, count, rhs, solution, CHOLMOD_DOUBLE, sizeof(double))) return 0;
    wh_border *b = plan->border;
    int n = (int) plan->factor->n;
    const int32_t *ap = plan->pattern->p, *ai = plan->pattern->i;
    double norm = b->norm;
    for (int j = 0; j < b->k; ++j) norm = fmax(norm, b->row_norm[sites[j]] + shift[j]);
    double *res = malloc((size_t) n * sizeof(double));
    double *correction = b->refine ? malloc((size_t) n * sizeof(double)) : NULL;
    long double *accurate = b->refine ? malloc((size_t) n * sizeof(long double)) : NULL;
    int ok = res && (!b->refine || (correction && accurate));
    *error = 0;
    for (size_t col = 0; ok && col < count; ++col, rhs += n, solution += n) {
        if (b->refine) {
            for (int j = 0; j < n; ++j) accurate[j] = rhs[j];
            for (int j = 0; j < n; ++j) {
                long double xj = solution[j], sum = 0;
                for (int p = ap[j]; p < ap[j+1]; ++p) {
                    int i = ai[p];
                    sum += (long double) values[p] * solution[i];
                    if (i != j) accurate[i] -= (long double) values[p] * xj;
                }
                accurate[j] -= sum;
            }
            for (int h = 0; h < b->k; ++h) accurate[sites[h]] -= (long double) shift[h] * solution[sites[h]];
            for (int j = 0; j < n; ++j) res[j] = (double) accurate[j];
            ok = wh_solve_plan(plan, 1, res, correction, CHOLMOD_DOUBLE, sizeof(double));
            if (!ok) break;
            for (int j = 0; j < n; ++j) solution[j] += correction[j];
        }
        memcpy(res, rhs, (size_t) n * sizeof(double));
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int p = ap[j]; p < ap[j+1]; ++p) {
                int i = ai[p];
                sum += values[p] * solution[i];
                if (i != j) res[i] -= values[p] * solution[j];
            }
            res[j] -= sum;
        }
        for (int h = 0; h < b->k; ++h) res[sites[h]] -= shift[h] * solution[sites[h]];
        double maxx = 0, maxb = 0, maxr = 0;
        for (int j = 0; j < n; ++j) {
            if (!isfinite(res[j]) || !isfinite(solution[j])) { ok = 0; break; }
            maxr = fmax(maxr, fabs(res[j]));
            maxx = fmax(maxx, fabs(solution[j]));
            maxb = fmax(maxb, fabs(rhs[j]));
        }
        double denom = norm * maxx + maxb;
        if (!isfinite(denom)) ok = 0;
        *error = fmax(*error, maxr / (denom > 0 ? denom : 1));
    }
    free(accurate);
    free(correction);
    free(res);
    return ok && *error <= 1e-11;
}
#endif
