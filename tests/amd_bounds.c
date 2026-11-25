/* Exercise the actual private scorer on a small quotient-graph witness. */
#include "../native/amd_policy.c"

#include <stdio.h>

int main(void)
{
    const int32_t start[] = {0, 3, 5, 8, 11, 12};
    const int32_t length[] = {3, 2, 3, 3, 1, 2};
    const int32_t elements[] = {0, 0, 1, 1, 0, -2};
    const int32_t adjacency[] = {2, 3, 4, 2, 3, 5, 0, 1, 5, 0, 1, 0, 2, 3};
    const int32_t mass[] = {1, 1, 2, 1, 3, 0};
    const int32_t ties[] = {1, -1, -1, -1, -1, -1};
    int32_t mark[6] = {0}, position[6] = {0}, neighbors[6] = {0};
    wh_pivot_stats stats = {0};
    wh_pivot_policy policy = {2, 64, 0, 0, mark, position, neighbors, 0, &stats};
    wh_current_policy = &policy;
    /* Eliminated element 5 joins 2 and 3. Vertex 0 has two missing
       cross-supervariable edges with weights 2*3 and 1*3. Vertex 1
       has only the connected 2/3 pair. */
    const uint64_t truth[] = {9, 0};
    for (int cap = 0; cap <= 80; ++cap) {
        for (int vertex = 0; vertex < 2; ++vertex) {
            quotient_rows rows = {start, length, elements, adjacency, mass, cap, 0};
            fill_interval bounds = score(&policy, &rows, vertex, 6);
            if (bounds.lower > truth[vertex] || bounds.upper < truth[vertex] ||
                (!rows.failed && (bounds.lower != truth[vertex] || bounds.upper != truth[vertex]))) {
                fprintf(stderr, "wrong interval at vertex %d, cap %d\n", vertex, cap);
                return 1;
            }
        }
        policy.max_entries = cap;
        stats = (wh_pivot_stats){0};
        int winner = wh_choose_pivot(0, 3, ties, 6, start, length, elements, adjacency, mass);
        if (stats.inspected > (uint64_t) (cap * 2) || (cap < 3 && winner != 0) ||
            (cap == 80 && winner != 1)) {
            fprintf(stderr, "invalid bounded pivot behavior at cap %d\n", cap);
            return 1;
        }
    }
    return 0;
}
