#include "fast_nn_block_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include "misc.h"

#define NUM_ROWS_IN_BATCH 10000
#define DATASET_ITEM_SIZE 4L
#define DATASET_DIMS_ITEM_SIZE 4L
#define NUM_ROWS_IN_BATCH 10000
#define NUM_ROWS_IN_BATCH_BATCH 500

static int make_dataset_row_size(int num_dimensions) {
    return num_dimensions * DATASET_ITEM_SIZE + DATASET_DIMS_ITEM_SIZE;
}

void get_vectors_in_block(const char* input_filename, blocks_info_t* blocks_info,
                          long long* block_indices, float* block, long long* num_vectors_in_block,
                          long long max_num_vectors_in_block, int num_vectors, int num_dimensions,
                          long long block_id)
{
    if (num_vectors_in_block) {
        *num_vectors_in_block = 0LL;
    }

    FILE* f = fopen(input_filename, "rb");
    get_vectors_in_block_batches(f, blocks_info, block_indices, block, num_vectors_in_block,
                                 max_num_vectors_in_block, num_vectors, num_dimensions,
                                 block_id, NUM_ROWS_IN_BATCH);
    fclose(f);
}


void get_vectors_in_block_batches(FILE* f, blocks_info_t* blocks_info, long long* block_indices,
                                  float* block, long long* num_vectors_in_block,
                                  long long max_num_vectors_in_block, int num_vectors,
                                  int num_dimensions, long long block_id, int rows_in_batch)
{
    long long rows_processed = 0;
    long long row_size = make_dataset_row_size(num_dimensions);
    float* batch = malloc(row_size * rows_in_batch);
    while (rows_processed < num_vectors) {
        if (max_num_vectors_in_block > 0LL && (*num_vectors_in_block > max_num_vectors_in_block)) {
            break;
        }

        long long current_batch_size = iminll(rows_in_batch, num_vectors - rows_processed);
        fread(batch, row_size, current_batch_size, f);

        block_loader_filter_block(NULL, blocks_info, block_indices, block, num_vectors_in_block,
                                  max_num_vectors_in_block,  num_dimensions, block_id,
                                  rows_processed, current_batch_size, batch);
        rows_processed += current_batch_size;
    }

    free(batch);
    batch = NULL;
}

void block_loader_filter_block(pthread_mutex_t* mutex, blocks_info_t* blocks_info,
                               long long* block_indices, float* block,
                               long long* num_vectors_in_block, long long max_num_vectors_in_block,
                               int num_dimensions, long long block_id, long long batch_start_index,
                               long long rows_in_batch, float* batch_start)
{
    for (int i = 0; i < rows_in_batch; ++i) {
        float* vector_start = batch_start + i * (num_dimensions + 1) + 1;
        int is_in_block = is_vector_in_block(vector_start, num_dimensions,
                                             blocks_info, block_id);
        if (!is_in_block) {
            continue;
        }

        if (mutex) {
            pthread_mutex_lock(mutex);
        }
        if (max_num_vectors_in_block > 0 && *num_vectors_in_block >= max_num_vectors_in_block) {
            ++(*num_vectors_in_block);
            if (mutex) {
                pthread_mutex_unlock(mutex);
            }

            return;
        }

        if (block) {
            memcpy(block + (*num_vectors_in_block) * num_dimensions,
                   vector_start, num_dimensions * sizeof(*block));
        }
        if (block_indices) {
            block_indices[*num_vectors_in_block] =
                    batch_start_index + i;
        }
        ++(*num_vectors_in_block);
        if (mutex) {
            pthread_mutex_unlock(mutex);
        }
    }
}

void block_loader_init(block_loader_t* block_loader, const char* input_filename,
                       blocks_info_t* blocks_info, long long* block_indices, float* block,
                       long long* num_vectors_in_block, long long max_num_vectors_in_block,
                       int num_vectors, int num_dimensions, long long block_id)
{
    block_loader->input_filename = input_filename;
    block_loader->blocks_info = blocks_info;
    block_loader->block_indices = block_indices;
    block_loader->block = block;
    block_loader->num_vectors_in_block = num_vectors_in_block;
    block_loader->max_num_vectors_in_block = max_num_vectors_in_block;
    block_loader->num_vectors = num_vectors;
    block_loader->num_dimensions = num_dimensions;
    block_loader->block_id = block_id;
    pthread_create(&block_loader->thread, NULL, block_loader_main_thread,
                   (void*)block_loader);
}

void block_loader_join(block_loader_t* block_loader) {
    pthread_join(block_loader->thread, NULL);
}

void* block_loader_main_thread(void* arg) {
    block_loader_t* block_loader = (block_loader_t*) arg;
    get_vectors_in_block(block_loader->input_filename, block_loader->blocks_info,
                         block_loader->block_indices, block_loader->block,
                         block_loader->num_vectors_in_block, block_loader->max_num_vectors_in_block,
                         block_loader->num_vectors, block_loader->num_dimensions,
                         block_loader->block_id);
}
