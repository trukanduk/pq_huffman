#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#include <pthread.h>
#include <semaphore.h>

#include <yael/nn.h>

#include "concurrent_queue.h"
#include "misc.h"
#include "fast_nn_temp_file.h"
#include "fast_nn_block.h"
#include "fast_nn_blocks_info.h"
#include "fast_nn_block_loader.h"
#include "fast_nn_block_loader2.h"
#include "fast_nn_result_data.h"

#define DATASET_ITEM_SIZE 4L
#define DATASET_DIMS_ITEM_SIZE 4L
#define NUM_ROWS_IN_BATCH 10000
#define NUM_ROWS_IN_BATCH_BATCH 500

static int make_dataset_row_size(int num_dimensions) {
    return num_dimensions * DATASET_ITEM_SIZE + DATASET_DIMS_ITEM_SIZE;
}

// Input data

void partial_load_vectors(const char* input_filename, int num_dimensions, int num_vectors_to_load,
                          const int* vectors_indices_to_load, float* output_vectors);
void load_vectors_dim(const char* input_filename, long long num_vectors, int num_dimensions,
                      int dimension_to_load, float* output);
void load_vectors_dim_batches(FILE* input_file, long long num_vectors, int num_dimensions,
                              int dimension_to_load, float* output, int rows_in_batch);

void partial_load_vectors(const char* input_filename, int num_dimensions, int num_vectors_to_load,
                          const int* vectors_indices_to_load, float* output_vectors)
{
    FILE* f = fopen(input_filename, "rb");
    int row_size = make_dataset_row_size(num_dimensions);
    for (
        int i = 0;
        i < num_vectors_to_load;
        ++i, ++vectors_indices_to_load, output_vectors += num_dimensions)
    {
        int vector_index = *vectors_indices_to_load;
        fseek(f, row_size * vector_index + DATASET_DIMS_ITEM_SIZE, SEEK_SET);
        fread((void*) output_vectors, num_dimensions, DATASET_ITEM_SIZE, f);
    }
    fclose(f);
}

    void load_vectors_dim(const char* input_filename, long long num_vectors, int num_dimensions,
                      int dimension_to_load, float* output)
    {
    FILE* f = fopen(input_filename, "rb");
    load_vectors_dim_batches(f, num_vectors, num_dimensions, dimension_to_load, output,
                             NUM_ROWS_IN_BATCH);
    fclose(f);
    }

    void load_vectors_dim_batches(FILE* input_file, long long num_vectors, int num_dimensions,
                              int dimension_to_load, float* output, int rows_in_batch)
    {
    int row_size = make_dataset_row_size(num_dimensions);
    fseek(input_file, DATASET_ITEM_SIZE * dimension_to_load + DATASET_DIMS_ITEM_SIZE, SEEK_SET);
    for (int i = 0; i < num_vectors; ++i) {
        fread((void*) output++, 1, sizeof(*output), input_file);
        fseek(input_file, row_size - DATASET_ITEM_SIZE, SEEK_CUR);
    }
    }

// Parse args

typedef struct _config {
    // NOTE: Required args
    const char* input_filename;
    const char* output_files_template;
    int num_nn;

    // NOTE: Optional args
    const char* temp_file;
    const char* blocks_info_cache;
    int delete_temp_file;
    int init_temp_file;
    int num_dims_to_split;
    double block_overlap_fraction;
    int num_blocks_per_dim;
    int num_threads;
    int with_blocks_stat;
    long long end_block_id;
    long long begin_block_id;
    long long num_dimensions_on_pass;

    // NOTE: computed args
    const char* output_indices_filename;
    const char* output_dists_filename;

    FILE* blocks_stat_file;

    // NOTE: dataset info
    int num_vectors;
    int num_dimensions;
} config_t;

typedef struct _dataset_metainfo {
    long long num_vectors;
    int num_dimensions;
} dataset_metainfo_t;

void print_help(const char* argv0);
dataset_metainfo_t get_metainfo(const char* input_filename);
config_t parse_args(int argc, const char* argv[]);
void config_free(config_t* config);

void print_help(const char* argv0) {
    fprintf(stderr, "Usage: %s  <input dataset path> <output files template> <num_nn> [optional args]\n", argv0);
    fprintf(stderr, "Optionl args:\n");
    fprintf(stderr, "  --temp-file <file>                         temp file path (default is nn_temp.vecs)\n");
    fprintf(stderr, "  --delete-temp-file, --no-delete-temp-file  delete temp file when done (default is true)\n");
    fprintf(stderr, "  --init-temp-file, --no-init-temp-file      clear old results of temp file (default is true)\n");
    fprintf(stderr, "  --num-dims <value>                         num dimensions to process (default is 5)\n");
    fprintf(stderr, "  --num-blocks-per-dim <value>               num blocks for each dimension (default is 2)\n");
    fprintf(stderr, "  --block-overlap-fraction <value>           fraction of dimension foroverlap (default is 0.3)\n");
    fprintf(stderr, "  --blocks-info-cache <value>                blocks info cache file (default is none)\n");
    fprintf(stderr, "  --num-dimensions-at-pass <value>           num dimensions to process on pass(default is 0)\n");
    fprintf(stderr, "  --blocks-from <value> --blocks-to <value>  start from/end on given block ids (default is all)\n");
}

dataset_metainfo_t get_metainfo(const char* input_filename) {
    FILE* f = fopen(input_filename, "rb");
    dataset_metainfo_t metainfo;
    metainfo.num_dimensions = 0;
    fread(&metainfo.num_dimensions, DATASET_DIMS_ITEM_SIZE, 1, f);
    fseek(f, 0, SEEK_END);
    long long filesize = ftell(f);
    fclose(f);

    long long row_size = make_dataset_row_size(metainfo.num_dimensions);
    metainfo.num_vectors = filesize / row_size;
    printf("num vectors %lld * row_size %lld (num_dim %d) == filesize %lld\n", metainfo.num_vectors,
           row_size, metainfo.num_dimensions, filesize);
    assert(metainfo.num_vectors * row_size == filesize);
    return metainfo;
}

config_t parse_args(int argc, const char* argv[]) {
    const char* argv0 = argv[0];
    if (argc < 3) {
        fprintf(stderr, "Too few required args\n");
        print_help(argv0);
        exit(1);
    }

    int arg_index = 1;
    config_t config;
    config.input_filename = argv[arg_index++];
    config.output_files_template = argv[arg_index++];
    config.num_nn = atoi(argv[arg_index++]);
    config.temp_file = "nn_temp.vecs";
    config.blocks_info_cache = NULL;
    config.delete_temp_file = 1;
    config.init_temp_file = 1;
    config.num_dims_to_split = 5;
    config.block_overlap_fraction = 0.3;
    config.num_blocks_per_dim = 3;
    config.num_threads = 1;
    config.with_blocks_stat = 0;
    config.begin_block_id = 0;
    config.end_block_id = -1;
    config.num_dimensions_on_pass = 0;

    config.output_indices_filename = NULL;
    config.output_dists_filename = NULL;
    config.blocks_stat_file = NULL;
    config.num_vectors = 0;
    config.num_dimensions = 0;

    int ok = 1;
    while (arg_index < argc) {
        if (!strcmp(argv[arg_index], "--with-blocks-stat")) {
            config.with_blocks_stat = 1;
        } else if (!strcmp(argv[arg_index], "--no-delete-temp-file")) {
            config.delete_temp_file = 0;
        } else if (!strcmp(argv[arg_index], "--delete-temp-file")) {
            config.delete_temp_file = 1;
        } else if (!strcmp(argv[arg_index], "--no-init-temp-file")) {
            config.init_temp_file = 0;
        } else if (!strcmp(argv[arg_index], "--init-temp-file")) {
            config.init_temp_file = 1;
        } else if (!strcmp(argv[arg_index], "--temp-file")) {
            config.temp_file = argv[++arg_index];
        } else if (!strcmp(argv[arg_index], "--blocks-info-cache")) {
            config.blocks_info_cache = argv[++arg_index];
        } else if (!strcmp(argv[arg_index], "--num-dims")) {
            config.num_dims_to_split = atoi(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--block-overlap-fraction")) {
            config.block_overlap_fraction = atof(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--num-blocks-per-dim")) {
            config.num_blocks_per_dim = atoi(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--num-threads")) {
            config.num_threads = atoi(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--blocks-from")) {
            config.begin_block_id = atoll(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--blocks-to")) {
            config.end_block_id = atoll(argv[++arg_index]) - 1;
        } else if (!strcmp(argv[arg_index], "--num-dimensions-at-pass")) {
            config.num_dimensions_on_pass = atoll(argv[++arg_index]);
        } else {
            fprintf(stderr, "Unknown arg %s\n", argv[arg_index]);
            ok = 0;
        }
        ++arg_index;
    }

    if (!ok) {
        exit(1);
    }

    config.output_indices_filename = concat(config.output_files_template, "nn_indices.ivecsl");
    config.output_dists_filename = concat(config.output_files_template, "nn_dist.fvecsl");
    if (config.with_blocks_stat) {
        char* blocks_stat_filename = concat(config.output_files_template, "blocks_stat.txt");
        config.blocks_stat_file = fopen(blocks_stat_filename, "w");
        free(blocks_stat_filename);
    }
    dataset_metainfo_t metainfo = get_metainfo(config.input_filename);
    config.num_vectors = metainfo.num_vectors;
    config.num_dimensions = metainfo.num_dimensions;
    return config;
}

void config_free(config_t* config) {
    if (config->output_indices_filename) {
        free((void*) config->output_indices_filename);
        config->output_indices_filename = NULL;
    }

    if (config->output_dists_filename) {
        free((void*) config->output_dists_filename);
        config->output_dists_filename = NULL;
    }

    if (config->blocks_stat_file) {
        fclose(config->blocks_stat_file);
        config->blocks_stat_file = NULL;
    }
}


int main(int argc, const char* argv[]);
void test_block_loaders(config_t* config, blocks_info_t* blocks_info);

void test_block_loaders(config_t* config, blocks_info_t* blocks_info) {
#define NUM_BLOCKS_PER_STEP 5
    for (
        long long block_id = 0;
        block_id < blocks_info->num_blocks_total;
        block_id += NUM_BLOCKS_PER_STEP)
    {
        long long num_blocks_this_step = \
                (block_id + NUM_BLOCKS_PER_STEP < blocks_info->num_blocks_total
                        ? NUM_BLOCKS_PER_STEP
                        : blocks_info->num_blocks_total - block_id);
        long long num_vectors_in_block[NUM_BLOCKS_PER_STEP];
        block_loader_t block_loaders[NUM_BLOCKS_PER_STEP];
        fprintf(stderr, "Starting blocks %lld - %lld of %lld\n", block_id,
                block_id + NUM_BLOCKS_PER_STEP - 1, blocks_info->num_blocks_total);
        for (int i = 0; i < num_blocks_this_step; ++i) {
            num_vectors_in_block[i] = 0LL;
            block_loader_init(&block_loaders[i], config->input_filename, blocks_info, NULL, NULL,
                              &num_vectors_in_block[i], 0LL, config->num_vectors,
                              config->num_dimensions, block_id + i);
        }

        for (int i = 0; i < num_blocks_this_step; ++i) {
            block_loader_join(&block_loaders[i]);
        }
    }
#undef NUM_BLOCKS_PER_STEP
}

static void init_block(long long capacity, int num_dimensions, float** block,
                       long long** block_indices)
{
    if (*block) {
        free(*block);
        *block = NULL;
    }
    if (*block_indices) {
        free(*block_indices);
        *block_indices = NULL;
    }

    long long block_row_size = sizeof(**block) * num_dimensions;
    *block = malloc(block_row_size * capacity);
    *block_indices = malloc(sizeof(**block_indices) * capacity);
}

float get_real_dist(float* a, float* b, int nd) {
    float result = 0.0;
    for (int i = 0; i < nd; ++i) {
        float tmp = a[i] - b[i];
        result += tmp*tmp;
    }
    return result;
}

#if 0
void run(config_t* config, blocks_info_t* blocks_info) {
#define NUM_BUFFERS 3
    block_loader_t block_loaders[NUM_BUFFERS];
    long long block_capacities[NUM_BUFFERS];
    long long* block_indices[NUM_BUFFERS];
    long long block_sizes[NUM_BUFFERS];
    float* blocks[NUM_BUFFERS];
    temp_file_loader_t temp_file_loaders[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        block_capacities[i] = (5LL * config->num_vectors + blocks_info->num_blocks_total - 1)
                / blocks_info->num_blocks_total;
        block_indices[i] = NULL;
        blocks[i] = NULL;
        block_sizes[i] = 0LL;
        init_block(block_capacities[i], config->num_dimensions, &blocks[i],
                   &block_indices[i]);
        temp_file_loader_init(&temp_file_loaders[i], config->temp_file, config->num_nn,
                              block_capacities[i]);
    }

    block_loader_init(&block_loaders[0], config->input_filename, blocks_info,
                      block_indices[0], blocks[0], &block_sizes[0],
                      block_capacities[0], config->num_vectors,
                      config->num_dimensions, 0LL);

    long long result_capacity = block_capacities[0];
    long long result_indices_row_size = sizeof(int) * (config->num_nn + 1);
    long long result_dists_row_size = sizeof(float) * (config->num_nn + 1);
    int* result_indices = malloc(result_indices_row_size * result_capacity);
    float* result_dists = malloc(result_dists_row_size * result_capacity);

    long long begin_block_id = config->begin_block_id;
    if (begin_block_id > blocks_info->num_blocks_total) {
        begin_block_id = blocks_info->num_blocks_total;
    }

    long long end_block_id = config->end_block_id;
    if (end_block_id < 0 || end_block_id > blocks_info->num_blocks_total) {
        end_block_id = blocks_info->num_blocks_total;
    } else if (end_block_id > config->begin_block_id) {
        end_block_id = config->begin_block_id;
    }
    for (long long block_id = begin_block_id; block_id < end_block_id; ++block_id) {
        if (block_id % 100 == 0) {
            printf("Start block %6lld / %6lld (%6lld / %6lld)\n",
                block_id - begin_block_id, end_block_id - begin_block_id,
                block_id, blocks_info->num_blocks_total);
        }

        int active_buffer = block_id % NUM_BUFFERS;
        int next_buffer = (active_buffer + 1) % NUM_BUFFERS;
        int prev_buffer = (active_buffer - 1 + NUM_BUFFERS) % NUM_BUFFERS;

        // printf("Starting block %lld\n", block_id);
        if (block_id + 1 != blocks_info->num_blocks_total) {
            temp_file_loader_save_join(&temp_file_loaders[next_buffer]);
            block_loader_init(&block_loaders[next_buffer], config->input_filename, blocks_info,
                              block_indices[next_buffer], blocks[next_buffer],
                              &block_sizes[next_buffer], block_capacities[next_buffer],
                              config->num_vectors, config->num_dimensions, block_id + 1);
        }

        block_loader_join(&block_loaders[active_buffer]);

        // TODO: always return real size of block
        while (block_sizes[active_buffer] > block_capacities[active_buffer]) {
            fprintf(stderr, "No enough block size. Capacity is %lld\n",
                    block_capacities[active_buffer]);

            block_capacities[active_buffer] *= 2;
            init_block(block_capacities[active_buffer], config->num_dimensions,
                       &blocks[active_buffer], &block_indices[active_buffer]);

            block_loader_init(&block_loaders[active_buffer], config->input_filename, blocks_info,
                              block_indices[active_buffer], blocks[active_buffer],
                              &block_sizes[active_buffer], block_capacities[active_buffer],
                              config->num_vectors, config->num_dimensions, block_id);
            block_loader_join(&block_loaders[active_buffer]);
        }

        temp_file_loader_load(&temp_file_loaders[active_buffer], block_sizes[active_buffer],
                              block_indices[active_buffer]);

        if (block_sizes[active_buffer] > result_capacity) {
            free(result_indices);
            result_indices = NULL;

            free(result_dists);
            result_dists = NULL;

            result_capacity = block_sizes[active_buffer];

            result_indices = malloc(result_indices_row_size * result_capacity);
            result_dists = malloc(result_dists_row_size * result_capacity);
        }

        int block_size = block_sizes[active_buffer];
        if (config->blocks_stat_file) {
            fprintf(config->blocks_stat_file, "%d ", block_size);
        }
        // printf("Starting knn for block %lld of size %d\n", block_id, block_size);
        int num_nn_real = iminll(config->num_nn, block_size - 1);
        knn_full_thread(2, block_size, block_size, config->num_dimensions,
                        num_nn_real + 1, blocks[active_buffer],
                        blocks[active_buffer], NULL, result_indices, result_dists,
                        config->num_threads);
        // printf("Done knn for block %lld\n", block_id);

        temp_file_loader_load_join(&temp_file_loaders[active_buffer]);

        // printf("Starting merge for block %lld\n", block_id);
        long long temp_file_row_size = make_temp_file_row_size(config->num_nn);
        nn_item_t temp_item;
        for (long long vec_index = 0; vec_index < block_size; ++vec_index) {
            nn_item_t* heap = temp_file_loaders[active_buffer].vectors
                    + config->num_nn * vec_index;
            int* vec_indices = result_indices + (num_nn_real + 1) * vec_index;
            float* vec_dists = result_dists + (num_nn_real + 1) * vec_index;
            for (int nn_index = 1; nn_index <= num_nn_real; ++nn_index) {
                temp_item.index = block_indices[active_buffer][vec_indices[nn_index]];
                temp_item.dist = vec_dists[nn_index];

                // float real_dist = get_real_dist(
                //         blocks[active_buffer] + vec_index * config->num_dimensions,
                //         blocks[active_buffer] + vec_indices[nn_index] * config->num_dimensions,
                //         config->num_dimensions);
                fast_nn_heap_push(heap, &temp_item, config->num_nn);
            }
        }
        // printf("Done merge for block %lld\n", block_id);

        temp_file_loader_save(&temp_file_loaders[active_buffer]);
        temp_file_loader_save_join(&temp_file_loaders[active_buffer]);
    }

    free(result_indices);
    result_indices = NULL;

    free(result_dists);
    result_dists = NULL;

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        temp_file_loader_save_join(&temp_file_loaders[i]);
        temp_file_loader_destroy(&temp_file_loaders[i]);

        free(blocks[i]);
        blocks[i] = NULL;

        free(block_indices[i]);
        block_indices[i] = NULL;
    }
#undef NUM_BUFFERS
}
#else

static void run_block(const config_t* config, const blocks_info_t* blocks_info,
                      const block_t* block, result_data_t* result_data) {
    result_data_realloc(result_data, block->size * (config->num_nn + 1));

    if (config->blocks_stat_file) {
        fprintf(config->blocks_stat_file, "%d ", block->size);
    }

    // printf("Starting knn for block %lld of size %d\n", block_id, block->size);
    int num_nn_real = iminll(config->num_nn, block->size - 1);
    knn_full_thread(2, block->size, block->size, config->num_dimensions,
                    num_nn_real + 1, block->data, block->data, NULL,
                    result_data->indices, result_data->dists, config->num_threads);
    // printf("Done knn for block %lld\n", block->id);
}

static void run_merge_block(const config_t* config, const result_data_t* result_data,
                            const block_loader2_t* block_loader, const block_t* block,
                            temp_file_loader_t* temp_file_loader) {
    // printf("Starting merge for block %lld\n", block_id);
    nn_item_t temp_item;
    int num_nn_real = iminll(config->num_nn, block->size - 1);
    for (long long vec_index = 0; vec_index < block->size; ++vec_index) {
        long long vec_temp_index = block->indices[vec_index];
        nn_item_t* heap = temp_file_loader->vectors + config->num_nn * vec_temp_index;

        int* vec_indices = result_data->indices + (num_nn_real + 1) * vec_index;
        float* vec_dists = result_data->dists + (num_nn_real + 1) * vec_index;

        for (int nn_index = 1; nn_index <= num_nn_real; ++nn_index) {
            long long block_index = vec_indices[nn_index];
            long long temp_index = block->indices[block_index];
            long long global_index = block_loader->global_indices[temp_index];
            temp_item.index = global_index;
            temp_item.dist = vec_dists[nn_index];

            fast_nn_heap_push(heap, &temp_item, config->num_nn);
        }
    }
}

static void run(const config_t* config, const blocks_info_t* blocks_info) {
#define NUM_BUFFERS 3
    block_t* blocks;
    block_loader2_t block_loaders[NUM_BUFFERS];
    temp_file_loader_t temp_file_loaders[NUM_BUFFERS];
    long long num_blocks_on_pass = 1;
    for (int i = 0; i < config->num_dimensions_on_pass; ++i) {
        num_blocks_on_pass *= config->num_blocks_per_dim;
    }
    blocks = malloc(sizeof(*blocks) * num_blocks_on_pass * NUM_BUFFERS);
    long long block_initial_capacity =
            (5LL * config->num_vectors + blocks_info->num_blocks_total - 1)
                    / blocks_info->num_blocks_total;
    for (int i = 0; i < num_blocks_on_pass * NUM_BUFFERS; ++i) {
        block_init(blocks + i, 0, config->num_dimensions, block_initial_capacity, BLOCK_INIT_ALL);
    }

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        block_loader2_init_from_blocks(block_loaders + i, config->input_filename,
                                       config->num_vectors, config->num_dimensions, blocks_info,
                                       blocks + i * num_blocks_on_pass, num_blocks_on_pass);
        block_loader2_set_common_prefix(
                block_loaders + i, blocks_info->num_dimensions - config->num_dimensions_on_pass);
        temp_file_loader_init(&temp_file_loaders[i], config->temp_file, config->num_nn,
                              block_initial_capacity);
    }

    result_data_t* result_data = malloc(sizeof(*result_data) * num_blocks_on_pass);
    for (int i = 0; i < num_blocks_on_pass; ++i) {
        result_data_init(result_data + i, block_initial_capacity * config->num_nn);
    }

    long long begin_block_id = config->begin_block_id;
    if (begin_block_id > blocks_info->num_blocks_total) {
        begin_block_id = blocks_info->num_blocks_total;
    }

    long long end_block_id = config->end_block_id;
    if (end_block_id < 0 || end_block_id > blocks_info->num_blocks_total) {
        end_block_id = blocks_info->num_blocks_total;
    } else if (end_block_id > config->begin_block_id) {
        end_block_id = config->begin_block_id;
    }
    block_loader2_set_start_block_id(block_loaders, begin_block_id);
    block_loader2_start(block_loaders);

    for (long long block_id = begin_block_id;
         block_id < end_block_id;
         block_id += num_blocks_on_pass)
    {
        if (block_id % 10 == 0) {
            printf("Start block %6lld / %6lld (%6lld / %6lld)\n",
                block_id - begin_block_id, end_block_id - begin_block_id,
                block_id, blocks_info->num_blocks_total);
        }

        int active_buffer = block_id % NUM_BUFFERS;
        int next_buffer = (active_buffer + 1) % NUM_BUFFERS;
        int prev_buffer = (active_buffer - 1 + NUM_BUFFERS) % NUM_BUFFERS;

        // printf("Starting block %lld\n", block_id);
        if (block_id + num_blocks_on_pass < end_block_id) {
            block_loader2_join(block_loaders + next_buffer);
            block_loader2_set_start_block_id(block_loaders + next_buffer,
                                             block_id + num_blocks_on_pass);
            block_loader2_start(block_loaders + next_buffer);
        }

        block_loader2_join(block_loaders + active_buffer);
        temp_file_loader_save_join(temp_file_loaders + prev_buffer);

        temp_file_loader_load(temp_file_loaders + active_buffer,
                              block_loaders[active_buffer].global_indices_size,
                              block_loaders[active_buffer].global_indices);


        for (int block_index = 0;
             block_index < num_blocks_on_pass && block_id + block_index < end_block_id;
             ++block_index)
        {
            block_t* block = block_loaders[active_buffer].blocks + block_index;
            run_block(config, blocks_info, block, result_data + block_index);
        }
        // printf("Done knn for block %lld\n", block_id);

        temp_file_loader_load_join(&temp_file_loaders[active_buffer]);

        for (int block_index = 0;
             block_index < num_blocks_on_pass && block_id + block_index < end_block_id;
             ++block_index)
        {
            block_t* block = block_loaders[active_buffer].blocks + block_index;
            run_merge_block(config, result_data + block_index, block_loaders + active_buffer,
                            block, &temp_file_loaders[active_buffer]);
        }

        // printf("Done merge for block %lld\n", block_id);
        ////////////////////////////////////////////////////////////////////////

        temp_file_loader_save(&temp_file_loaders[active_buffer]);
        // temp_file_loader_save_join(&temp_file_loaders[active_buffer]);
    }

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        temp_file_loader_save_join(&temp_file_loaders[i]);
        temp_file_loader_destroy(&temp_file_loaders[i]);

        // TODO: cleanup
    }
#undef NUM_BUFFERS
}
#endif

int main(int argc, const char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);

    config_t config = parse_args(argc, argv);
    if (config.init_temp_file) {
        init_temp_file(config.temp_file, config.num_vectors, config.num_nn);
        printf("Temp file: Done\n");
    }

    blocks_info_t blocks_info;
    if (!config.blocks_info_cache
        || !blocks_info_load_filename(&blocks_info, config.blocks_info_cache))
    {
        blocks_info_init(&blocks_info, config.input_filename, config.num_vectors,
                         config.num_dimensions, config.num_dims_to_split, config.num_blocks_per_dim,
                         config.block_overlap_fraction);
        if (config.blocks_info_cache) {
            blocks_info_save_filename(&blocks_info, config.blocks_info_cache);
        }
        printf("Blocks info: Done\n");
    } else {
        printf("Blocks info: Loaded\n");
    }

    run(&config, &blocks_info);
    // test_block_loaders(&config, &blocks_info);

    temp_file_to_result(config.temp_file, config.output_indices_filename,
                        config.output_dists_filename, config.num_vectors, config.num_nn);

    if (config.delete_temp_file) {
        remove(config.temp_file);
    }
    config_free(&config);
    return 0;
}
