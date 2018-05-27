#ifndef _FAST_NN_BLOCK_LOADER2_H_
#define _FAST_NN_BLOCK_LOADER2_H_

#include "fast_nn_block.h"
#include "fast_nn_blocks_info.h"

#include <pthread.h>

typedef struct _block_loader2 {
    pthread_t* thread;
    const char* input_filename;
    long long num_vectors;
    int num_dimensions;
    const blocks_info_t* blocks_info;
    int num_blocks_to_load;
    block_t* blocks;
    long long global_indices_capacity;
    long long global_indices_size;
    long long* global_indices;
} block_loader2_t;

void block_loader2_init_from_blocks(block_loader2_t* block_loader, const char* input_filename,
                                    long long num_vectors, int num_dimensions,
                                    const blocks_info_t* block_info, block_t* blocks,
                                    long long num_blocks);
void block_loader2_set_start_block_id(block_loader2_t* block_loader, long long start_block);
void block_loader2_destroy(block_loader2_t* block_loader);

void block_loader2_start(block_loader2_t* block_loader);
void block_loader2_join(block_loader2_t* block_loader);

#endif // _FAST_NN_BLOCK_LOADER2_H_
