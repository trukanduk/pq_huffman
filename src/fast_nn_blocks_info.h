#ifndef _FAST_NN_BLOCKS_INFO_H_
#define _FAST_NN_BLOCKS_INFO_H_

#include "fast_nn_block.h"

#include <stdio.h>

typedef struct _block_dimension_info {
    int dimension;
    int num_blocks;
    double block_overlap_fraction;
    long long block_id_mask;
    float* block_starts;
    float* block_ends;
} block_dimension_info_t;

typedef struct _blocks_info {
    int num_dimensions;
    double block_overlap_fraction;
    int num_blocks_per_dim;
    long long num_blocks_total;
    block_dimension_info_t* dimension_infos;
} blocks_info_t;

void dimension_info_build(block_dimension_info_t* dimension_info, const char* input_filename,
                          long long num_vectors, int num_dimensions, int dimension_to_use,
                          int num_blocks, double block_overlap_fraction);
void dimension_info_destroy(block_dimension_info_t* dimension_info);
void blocks_info_init(blocks_info_t* blocks_info, const char* input_filename, long long num_vectors,
                      int num_dimensions, int num_dimensions_to_split, int num_blocks,
                      double block_overlap_fraction);
void blocks_info_save_filename(const blocks_info_t* blocks_info, const char* filename);
void blocks_info_save_file(const blocks_info_t* blocks_info, FILE* file);
int blocks_info_load_filename(blocks_info_t* blocks_info, const char* filename);
void blocks_info_load_file(blocks_info_t* blocks_info, FILE* file);
void blocks_info_destroy(blocks_info_t* blocks_info);
int is_vector_in_block(const float* vector, int num_dimensions, const blocks_info_t* blocks_info,
                       long long block_id);

#endif // _FAST_NN_BLOCKS_INFO_H_
