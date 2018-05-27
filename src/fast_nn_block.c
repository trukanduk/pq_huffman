#include "fast_nn_block.h"

#include <stdlib.h>
#include <string.h>

void block_init(block_t* block, long long block_id, int num_dimensions, long long initial_capacity,
                int init_flags)
{
    block->id = block_id;
    block->num_dimensions = num_dimensions;
    block->capacity = 100;
    block->size = 0;
    block->data = NULL;
    block->indices = NULL;

    if (init_flags & BLOCK_INIT_WITH_DATA) {
        block->data = malloc(sizeof(*block->data) * block->num_dimensions * block->capacity);
    }
    if (init_flags & BLOCK_INIT_WITH_INDICES) {
        block->indices = malloc(sizeof(*block->indices) * block->capacity);
    }
}

void block_destroy(block_t* block) {
    block->id = -1;
    block->num_dimensions = 0;
    block->capacity = 0;
    block->size = 0;

    if (block->data) {
        free(block->data);
        block->data = NULL;
    }

    if (block->indices) {
        free(block->indices);
        block->indices = NULL;
    }
}

void block_push(block_t* block, long long index, const float* vector) {
    if (block->size >= block->capacity) {
        block_realloc(block, (block->capacity + 1) * 2);
    }

    if (block->indices) {
        block->indices[block->size] = index;
    }
    if (block->data) {
        memcpy(block->data + block->size * block->num_dimensions, vector,
               sizeof(*vector) * block->num_dimensions);
    }
    ++block->size;
}

void block_realloc(block_t* block, long long new_capacity) {
    if (block->data) {
        block->data = realloc(block->data,
                              sizeof(*block->data) * new_capacity * block->num_dimensions);
    }
    if (block->indices) {
        block->indices = realloc(block->indices,
                                 sizeof(*block->indices) * new_capacity);
    }
    block->capacity = new_capacity;
}

void block_set_id(block_t* block, long long block_id) {
    block->id = block_id;
    block->size = 0;
}
