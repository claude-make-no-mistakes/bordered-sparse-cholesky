/* Read-only lookahead for the pinned SuiteSparse AMD minimum-degree bucket. */
#ifndef WORKHORSES_AMD_POLICY_H
#define WORKHORSES_AMD_POLICY_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t decisions;
    uint64_t candidates;
    uint64_t changed;
    uint64_t inspected;
    uint64_t incomplete;
} wh_pivot_stats;

typedef struct {
    int32_t limit, max_degree, max_entries, estimate;
    int32_t *mark, *position, *neighbors;
    int32_t epoch;
    wh_pivot_stats *stats;
} wh_pivot_policy;

extern _Thread_local wh_pivot_policy *wh_current_policy;

int32_t wh_choose_pivot(int32_t head, int32_t degree, const int32_t *next, size_t n,
                        const int32_t *pe, const int32_t *len, const int32_t *elen,
                        const int32_t *iw, const int32_t *mass);

#endif
