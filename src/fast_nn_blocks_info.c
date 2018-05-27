#include "fast_nn_blocks_info.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "misc.h"
#include "vecs_io.h"

#define DATASET_ITEM_SIZE 4L
#define DATASET_DIMS_ITEM_SIZE 4L
#define NUM_ROWS_IN_BATCH 10000
#define NUM_ROWS_IN_BATCH_BATCH 500

static int make_dataset_row_size(int num_dimensions) {
    return num_dimensions * DATASET_ITEM_SIZE + DATASET_DIMS_ITEM_SIZE;
}

static void load_vectors_dim_batches(FILE* input_file, long long num_vectors, int num_dimensions,
                              int dimension_to_load, float* output, int rows_in_batch)
{
    int row_size = make_dataset_row_size(num_dimensions);
    fseek(input_file, DATASET_ITEM_SIZE * dimension_to_load + DATASET_DIMS_ITEM_SIZE, SEEK_SET);
    for (int i = 0; i < num_vectors; ++i) {
        fread((void*) output++, 1, sizeof(*output), input_file);
        fseek(input_file, row_size - DATASET_ITEM_SIZE, SEEK_CUR);
    }
}

static void load_vectors_dim(const char* input_filename, long long num_vectors, int num_dimensions,
                      int dimension_to_load, float* output)
{
    FILE* f = fopen(input_filename, "rb");
    load_vectors_dim_batches(f, num_vectors, num_dimensions, dimension_to_load, output,
                             NUM_ROWS_IN_BATCH);
    fclose(f);
}

static int float_cmp(const void* left, const void* right) {
    float v = * (float*) left - * (float*) right;
    if (fabs(v) < 1e-9) {
        return 0;
    } else if (v < 0) {
        return -1;
    } else {
        return 1;
    }
}

void dimension_info_build(block_dimension_info_t* dimension_info, const char* input_filename,
                          long long num_vectors, int num_dimensions, int dimension_to_use,
                          int num_blocks, double block_overlap_fraction)
{
    float* dimension = malloc(sizeof(*dimension) * num_vectors);
    load_vectors_dim(input_filename, num_vectors, num_dimensions, dimension_to_use, dimension);
    qsort(dimension, num_vectors, sizeof(*dimension), float_cmp);
    dimension_info->dimension = dimension_to_use;
    dimension_info->num_blocks = num_blocks;
    dimension_info->block_overlap_fraction = block_overlap_fraction;
    dimension_info->block_id_mask = 1;
    for (int i = 0; i < dimension_to_use; ++i) {
        dimension_info->block_id_mask *= num_blocks;
    }
    dimension_info->block_starts = malloc(sizeof(*dimension_info->block_starts) * num_blocks * 2);
    dimension_info->block_ends = dimension_info->block_starts + num_blocks;
    printf("dim %d in [%lf, %lf], num_blocks %d, overlap %lf\n", dimension_to_use,
           dimension[0], dimension[num_vectors - 1], num_blocks, block_overlap_fraction);
    long long num_overlapped = (long long)(num_vectors * block_overlap_fraction / 2);
    for (int i = 0; i < num_blocks; ++i) {
        long long start_index = num_vectors * i / num_blocks - num_overlapped;
        start_index = iclampll(start_index, 0, num_vectors);
        dimension_info->block_starts[i] = dimension[start_index];
        long long end_index = num_vectors * (i + 1) / num_blocks + num_overlapped;
        end_index = iclampll(end_index, 0, num_vectors);
        dimension_info->block_ends[i] = dimension[iminll(end_index, num_vectors - 1)];
        printf("start %e (%lld) end %e (%lld)\n", dimension_info->block_starts[i], start_index, dimension_info->block_ends[i], end_index);
    }
    dimension_info->block_starts[0] = dimension[0] - 1.0;
    dimension_info->block_ends[num_blocks - 1] = dimension[num_vectors - 1] + 1.0;
    for(int i = 0; i < num_blocks; ++i) {
    }
    free(dimension);
}

void dimension_info_destroy(block_dimension_info_t* dimension_info) {
    dimension_info->dimension = -1;
    free(dimension_info->block_starts);
    dimension_info->block_starts = NULL;
    dimension_info->block_ends = NULL;
}

void blocks_info_init(blocks_info_t* blocks_info, const char* input_filename, long long num_vectors,
                      int num_dimensions, int num_dimensions_to_split, int num_blocks,
                      double block_overlap_fraction)
{
    blocks_info->num_dimensions = num_dimensions_to_split;
    blocks_info->block_overlap_fraction = block_overlap_fraction;
    blocks_info->num_blocks_per_dim = num_blocks;
    blocks_info->dimension_infos =
            malloc(sizeof(*blocks_info->dimension_infos) * num_dimensions_to_split);
    blocks_info->num_blocks_total = 1;
    for (int i = 0; i < num_dimensions_to_split; ++i) {
        printf("Starting block info for dimension %d", i);
        int dimension = -1;
        int has_collision = 1;
        while (has_collision) {
            dimension = rand() % num_dimensions_to_split;
            has_collision = 0;
            for (int j = 0; j < i && !has_collision; ++j) {
                has_collision = (blocks_info->dimension_infos[j].dimension == dimension);
            }
        }

        dimension_info_build(blocks_info->dimension_infos + i, input_filename, num_vectors,
                             num_dimensions, dimension, num_blocks, block_overlap_fraction);
        blocks_info->num_blocks_total *= num_blocks;
    }
}

void blocks_info_save_filename(const blocks_info_t* blocks_info, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return;
    }
    blocks_info_save_file(blocks_info, file);
    fclose(file);
}

void blocks_info_save_file(const blocks_info_t* blocks_info, FILE* file) {
    fwrite(&blocks_info->num_dimensions, 1, sizeof(blocks_info->num_dimensions), file);
    fwrite(&blocks_info->block_overlap_fraction, 1, sizeof(blocks_info->block_overlap_fraction),
           file);
    fwrite(&blocks_info->num_blocks_per_dim, 1, sizeof(blocks_info->num_blocks_per_dim), file);
    fwrite(&blocks_info->num_blocks_total, 1, sizeof(blocks_info->num_blocks_total), file);
    for (int dim_index = 0; dim_index < blocks_info->num_dimensions; ++dim_index) {
        const block_dimension_info_t* dimension_info = blocks_info->dimension_infos + dim_index;

        fwrite(&dimension_info->dimension, 1, sizeof(dimension_info->dimension), file);
        fwrite(&dimension_info->block_id_mask, 1, sizeof(dimension_info->block_id_mask), file);
        fwrite(dimension_info->block_starts, dimension_info->num_blocks * 2,
               sizeof(*dimension_info->block_starts), file);
    }
}

int blocks_info_load_filename(blocks_info_t* blocks_info, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "No dimensions info file: %s\n", filename);
        return 0;
    }
    blocks_info_load_file(blocks_info, file);
    fclose(file);
    return 1;
}

void blocks_info_load_file(blocks_info_t* blocks_info, FILE* file) {
    fread(&blocks_info->num_dimensions, 1, sizeof(blocks_info->num_dimensions), file);
    fread(&blocks_info->block_overlap_fraction, 1, sizeof(blocks_info->block_overlap_fraction),
           file);
    fread(&blocks_info->num_blocks_per_dim, 1, sizeof(blocks_info->num_blocks_per_dim), file);
    fread(&blocks_info->num_blocks_total, 1, sizeof(blocks_info->num_blocks_total), file);
    blocks_info->dimension_infos = malloc(
            sizeof(*blocks_info->dimension_infos) * blocks_info->num_dimensions);
    for (int dim_index = 0; dim_index < blocks_info->num_dimensions; ++dim_index) {
        block_dimension_info_t* dimension_info = blocks_info->dimension_infos + dim_index;

        fread(&dimension_info->dimension, 1, sizeof(dimension_info->dimension), file);
        dimension_info->num_blocks = blocks_info->num_blocks_per_dim;
        dimension_info->block_overlap_fraction = blocks_info->block_overlap_fraction;
        fread(&dimension_info->block_id_mask, 1, sizeof(dimension_info->block_id_mask), file);
        dimension_info->block_starts = malloc(
                sizeof(*dimension_info->block_starts) * dimension_info->num_blocks * 2);
        dimension_info->block_ends = dimension_info->block_starts + dimension_info->num_blocks;
        fread(dimension_info->block_starts, dimension_info->num_blocks * 2,
               sizeof(*dimension_info->block_starts), file);
    }
}

void blocks_info_destroy(blocks_info_t* blocks_info) {
    for (int i = 0; i < blocks_info->num_dimensions; ++i) {
        dimension_info_destroy(blocks_info->dimension_infos + i);
    }
    blocks_info->num_dimensions = 0;
    blocks_info->block_overlap_fraction = 0;
    blocks_info->num_blocks_total = 0;
    blocks_info->num_blocks_per_dim = 0;
    free(blocks_info->dimension_infos);
    blocks_info->dimension_infos = NULL;
}

int is_vector_in_block(const float* vector, int num_dimensions, const blocks_info_t* blocks_info,
                       long long block_id)
{
    block_dimension_info_t* dimension_infos = blocks_info->dimension_infos;
    for (int i = 0; i < blocks_info->num_dimensions; ++i, ++dimension_infos) {
        float dimension_value = vector[dimension_infos->dimension];
        int dim_block_id = block_id
                / dimension_infos->block_id_mask
                % blocks_info->num_blocks_per_dim;
        if (dimension_value < dimension_infos->block_starts[dim_block_id]
            || dimension_value > dimension_infos->block_ends[dim_block_id])
        {
            return 0;
        }
    }
    return 1;
}

