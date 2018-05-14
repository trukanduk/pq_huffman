#include "dsu.h"

#include <stdlib.h>

void dsu_init(dsu_t* dsu, long long num_elements) {
    dsu->num_elements = num_elements;
    dsu->items = malloc(sizeof(*dsu->items) * dsu->num_elements);

    dsu_item_t* items_it = dsu->items;
    for (long long item_id = 0; item_id < dsu->num_elements; ++item_id, ++items_it) {
        items_it->rank = 0;
        items_it->parent_id = item_id;
    }
}

void dsu_destroy(dsu_t* dsu) {
    dsu->num_elements = 0;
    free(dsu->items);
    dsu->items = NULL;
}

vector_id_t dsu_find_set(dsu_t* dsu, vector_id_t element) {
    // OPTIMIZE: Make non-recursive?
    dsu_item_t* item = dsu->items + element;
    if (element == item->parent_id) {
        return element;
    }

    vector_id_t root = dsu_find_set(dsu, item->parent_id);
    item->parent_id = root;
    return root;
}

int dsu_is_one_set(dsu_t* dsu, vector_id_t first, vector_id_t second) {
    return (dsu_find_set(dsu, first) == dsu_find_set(dsu, second));
}

void dsu_union(dsu_t* dsu, vector_id_t first, vector_id_t second) {
    vector_id_t first_root = dsu_find_set(dsu, first);
    vector_id_t second_root = dsu_find_set(dsu, second);
    if (first_root == second_root) {
        return;
    }

    dsu_item_t* first_root_item = dsu->items + first_root;
    dsu_item_t* second_root_item = dsu->items + second_root;
    if (first_root_item->rank < first_root_item->rank) {
        dsu_item_t* tmp = first_root_item;
        first_root_item = second_root_item;
        second_root_item = first_root_item;

        vector_id_t tmpv = first_root;
        first_root = second_root;
        second_root = tmpv;
    }

    second_root_item->parent_id = first_root;
    if (first_root_item->rank == second_root_item->rank) {
        ++first_root_item->rank;
    }
}
