#include "amd_policy.h"

#include <limits.h>
#include <string.h>

_Thread_local wh_pivot_policy *wh_current_policy;

typedef struct {
    const int32_t *pe, *len, *elen, *iw, *mass;
    int32_t remaining;
    int failed;
} quotient_rows;

typedef struct {
    uint64_t lower, upper;
} fill_interval;

typedef struct {
    wh_pivot_policy *policy;
    quotient_rows *graph;
    int32_t vertex, count, after;
    uint64_t weight;
} accumulator;

static void set_seen(accumulator *acc, int32_t neighbor)
{
    wh_pivot_policy *policy = acc->policy;
    int mass = acc->graph->mass[neighbor];
    if (neighbor == acc->vertex || mass <= 0 || policy->mark[neighbor] == policy->epoch) return;
    policy->mark[neighbor] = policy->epoch;
    if (acc->after < 0) {
        policy->neighbors[acc->count++] = neighbor;
        acc->weight += (uint64_t) mass;
    } else if (policy->position[neighbor] > acc->after) {
        acc->weight += (uint64_t) mass;
    }
}

static int charge(quotient_rows *g)
{
    if (g->remaining == 0) {
        g->failed = 1;
        return 0;
    }
    g->remaining--;
    return 1;
}

static int walk_row(accumulator *acc, size_t n)
{
    wh_pivot_policy *p = acc->policy;
    quotient_rows *g = acc->graph;
    if (p->epoch == INT32_MAX) {
        memset(p->mark, 0, n * sizeof(*p->mark));
        p->epoch = 0;
    }
    p->epoch++;
    int start = g->pe[acc->vertex];
    if (start < 0) return 1;
    for (int i = 0; i < g->elen[acc->vertex]; ++i) {
        if (!charge(g)) return 0;
        int element = g->iw[start + i];
        int begin = g->pe[element];
        if (begin < 0) {
            // An unrecognised quotient row cannot certify a missing edge.
            g->failed = 1;
            return 0;
        }
        for (int j = 0; j < g->len[element]; ++j) {
            if (!charge(g)) return 0;
            set_seen(acc, g->iw[begin + j]);
        }
    }
    for (int i = g->elen[acc->vertex]; i < g->len[acc->vertex]; ++i) {
        if (!charge(g)) return 0;
        set_seen(acc, g->iw[start + i]);
    }
    return 1;
}

static fill_interval score(wh_pivot_policy *policy, quotient_rows *graph, int vertex, size_t n)
{
    accumulator root = {policy, graph, vertex, 0, -1, 0};
    if (!walk_row(&root, n)) return (fill_interval){0, UINT64_MAX};
    uint64_t remaining_weight = root.weight, possible = 0;
    for (int i = 0; i < root.count; ++i) {
        int neighbor = policy->neighbors[i];
        uint64_t mass = (uint64_t) graph->mass[neighbor];
        policy->position[neighbor] = i + 1;
        remaining_weight -= mass;
        possible += mass * remaining_weight;
    }

    fill_interval bounds = {0, possible};
    remaining_weight = root.weight;
    for (int i = 0; i + 1 < root.count; ++i) {
        int neighbor = policy->neighbors[i];
        uint64_t mass = (uint64_t) graph->mass[neighbor];
        remaining_weight -= mass;
        accumulator row = {policy, graph, neighbor, 0, i + 1, 0};
        int complete = walk_row(&row, n);
        bounds.upper -= mass * row.weight;
        if (!complete) break;
        bounds.lower += mass * (remaining_weight - row.weight);
    }
    for (int i = 0; i < root.count; ++i) policy->position[policy->neighbors[i]] = 0;
    return bounds;
}

static uint64_t midpoint(fill_interval value)
{
    return value.lower + (value.upper - value.lower) / 2;
}

int32_t wh_choose_pivot(int32_t head, int32_t degree, const int32_t *next, size_t n,
                        const int32_t *pe, const int32_t *len, const int32_t *elen,
                        const int32_t *iw, const int32_t *mass)
{
    wh_pivot_policy *policy = wh_current_policy;
    if (!policy || policy->limit < 2 || next[head] < 0 || degree > policy->max_degree) return head;
    fill_interval best = {0, UINT64_MAX};
    int winner = head;
    policy->stats->decisions++;
    int candidate = head;
    for (int i = 0; i < policy->limit && candidate >= 0; ++i, candidate = next[candidate]) {
        quotient_rows graph = {pe, len, elen, iw, mass, policy->max_entries, 0};
        fill_interval value = score(policy, &graph, candidate, n);
        policy->stats->candidates++;
        policy->stats->inspected += (uint64_t) (policy->max_entries - graph.remaining);
        policy->stats->incomplete += graph.failed;
        const int improvement = policy->estimate ? midpoint(value) < midpoint(best) : value.upper < best.lower;
        if (!i || improvement) {
            best = value;
            winner = candidate;
            if (best.upper == 0) break;
        }
    }
    if (winner != head) policy->stats->changed++;
    return winner;
}
