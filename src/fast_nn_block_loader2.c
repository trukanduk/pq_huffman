#include "fast_nn_block_loader2.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "vecs_io.h"
#include "misc.h"

enum {
    BATCH_SIZE = 1000 * 1000
};

static void* block_loader2_thread(void* arg);

void block_loader2_init_from_blocks(block_loader2_t* block_loader, const char* input_filename,
                                    long long num_vectors, int num_dimensions,
                                    const blocks_info_t* blocks_info, block_t* blocks,
                                    long long num_blocks) {
    block_loader->thread = NULL;
    block_loader->input_filename = input_filename;
    block_loader->num_vectors = num_vectors;
    block_loader->num_dimensions = num_dimensions;
    block_loader->blocks_info = blocks_info;
    block_loader->num_blocks_to_load = num_blocks;
    block_loader->blocks = blocks;
    block_loader->global_indices_capacity = 500;
    block_loader->global_indices_size = 0;
    block_loader->global_indices =
            malloc(sizeof(*block_loader->global_indices) * block_loader->global_indices_capacity);
}

void block_loader2_set_start_block_id(block_loader2_t* block_loader, long long start_block) {
    for (int block_index = 0; block_index < block_loader->num_blocks_to_load; ++block_index) {
        block_set_id(block_loader->blocks + block_index, start_block + block_index);
    }
}

void block_loader2_destroy(block_loader2_t* block_loader) {
    block_loader2_join(block_loader);
    // TODO:
}

void block_loader2_start(block_loader2_t* block_loader) {
    if (block_loader->thread) {
        fprintf(stderr, "Block loader2 is not joined yet!\n");
        block_loader2_join(block_loader);
    }

    block_loader->thread = malloc(sizeof(block_loader->thread));
    pthread_create(block_loader->thread, NULL, block_loader2_thread, (void*) block_loader);
}

void block_loader2_join(block_loader2_t* block_loader) {
    if (block_loader->thread) {
        pthread_join(*block_loader->thread, NULL);
        free(block_loader->thread);
        block_loader->thread = NULL;
    }
}

static void block_loader2_push_index(block_loader2_t* block_loader, long long index) {
    if (block_loader->global_indices_size >= block_loader->global_indices_capacity) {
        block_loader->global_indices_capacity = (block_loader->global_indices_capacity + 1) * 2;
        long long new_size = sizeof(*block_loader->global_indices)
                * block_loader->global_indices_capacity;
        block_loader->global_indices = realloc(block_loader->global_indices, new_size);
    }

    block_loader->global_indices[block_loader->global_indices_size++] = index;
}

void* block_loader2_thread(void* arg) {
    block_loader2_t* block_loader = (block_loader2_t*) arg;

    FILE* file = fopen(block_loader->input_filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open %s file\n", block_loader->input_filename);
        exit(1);
    }

    for (int i = 0; i < block_loader->num_blocks_to_load; ++i) {
        assert(block_loader->blocks[i].size == 0);
        // block_loader->blocks[i].size = 0;
    }
    long long num_processed = 0;
    block_loader->global_indices_size = 0;
    float* batch = malloc(sizeof(*batch) * (block_loader->num_dimensions + 1) * block_loader->num_vectors);
    while (num_processed < block_loader->num_vectors) {
        long long current_batch_size = iminll(BATCH_SIZE, block_loader->num_vectors - num_processed);
        fread(batch, current_batch_size, sizeof(*batch) * (block_loader->num_dimensions + 1), file);

        const float* batch_it = batch;
        for (int vec_index = 0;
             vec_index != current_batch_size;
             ++vec_index, batch_it += block_loader->num_dimensions + 1)
        {
            int vector_taken = 0;
            // OPTIMIZE: common blocks prefix?
            for (int block_index = 0;
                 block_index < block_loader->num_blocks_to_load;
                 ++block_index)
            {
                block_t* block = block_loader->blocks + block_index;
                int in_block = is_vector_in_block(batch_it + 1, block_loader->num_dimensions,
                                                  block_loader->blocks_info, block->id);
                if (in_block) {
                    if (!vector_taken) {
                        block_loader2_push_index(block_loader, num_processed + vec_index);
                        vector_taken = 1;
                    }
                    block_push(block, block_loader->global_indices_size - 1, batch_it + 1);
                }
            }
        }
        assert(batch_it == batch + current_batch_size * (block_loader->num_dimensions + 1));

        num_processed += current_batch_size;
    }

    free(batch);
    batch = NULL;
    fclose(file);

    return NULL;
}
