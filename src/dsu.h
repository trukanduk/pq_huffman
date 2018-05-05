#ifndef _DSU_H
#define _DSU_H

#include "misc.h"

typedef struct _dsu_item {
    vector_id_t parent_id; // NOTE: not pointer to save memory (if vector_id_t = int)
    int rank;
} dsu_item_t;

typedef struct _dsu {
    long long num_elements;
    dsu_item_t* items;
} dsu_t;

void dsu_init(dsu_t* dsu, long long num_elements);
void dsu_destroy(dsu_t* dsu);

vector_id_t dsu_find_set(dsu_t* dsu, vector_id_t element);
int dsu_is_one_set(dsu_t* dsu, vector_id_t first, vector_id_t second);
void dsu_union(dsu_t* dsu, vector_id_t first, vector_id_t second);

#endif // _DSU_H
