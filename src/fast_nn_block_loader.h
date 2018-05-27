#ifndef _FAST_NN_BLOCK_LOADER_H_
#define _FAST_NN_BLOCK_LOADER_H_

#include "fast_nn_block.h"
#include "fast_nn_blocks_info.h"

#include <pthread.h>

void get_vectors_in_block(const char* input_filename, blocks_info_t* blocks_info,
                          long long* block_indices, float* block, long long* num_vectors_in_block,
                          long long max_num_vectors_in_block, int num_vectors, int num_dimensions,
                          long long block_id);


void get_vectors_in_block_batches(FILE* f, blocks_info_t* blocks_info, long long* block_indices,
                                  float* block, long long* num_vectors_in_block,
                                  long long max_num_vectors_in_block, int num_vectors,
                                  int num_dimensions, long long block_id, int rows_in_batch);
void block_loader_filter_block(pthread_mutex_t* mutex, blocks_info_t* blocks_info,
                               long long* block_indices, float* block,
                               long long* num_vectors_in_block, long long max_num_vectors_in_block,
                               int num_dimensions, long long block_id, long long batch_start_index,
                               long long rows_in_batch, float* batch_start);

typedef struct _block_loader {
    pthread_t thread;
    const char* input_filename;
    blocks_info_t* blocks_info;
    long long* block_indices;
    float* block;
    long long* num_vectors_in_block;
    long long max_num_vectors_in_block;
    int num_vectors;
    int num_dimensions;
    long long block_id;
    int num_threads;
} block_loader_t;

void block_loader_init(block_loader_t* block_loader, const char* input_filename,
                       blocks_info_t* blocks_info, long long* block_indices, float* block,
                       long long* num_vectors_in_block, long long max_num_vectors_in_block,
                       int num_vectors, int num_dimensions, long long block_id);
void block_loader_join(block_loader_t* block_loader);
void* block_loader_main_thread(void* arg);

#endif // _FAST_NN_BLOCK_LOADER_H_
