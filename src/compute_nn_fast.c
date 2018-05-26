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

#define DATASET_ITEM_SIZE 4L
#define DATASET_DIMS_ITEM_SIZE 4L
#define NUM_ROWS_IN_BATCH 10000
#define NUM_ROWS_IN_BATCH_BATCH 500

int make_dataset_row_size(int num_dimensions) {
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
#if 0
    assert(DATASET_ITEM_SIZE == DATASET_DIMS_ITEM_SIZE); // This method should be rewritten else
    int row_size = make_dataset_row_size(num_dimensions);
    float* batch = malloc(rows_in_batch * row_size);

    long long rows_processed = 0;
    while (rows_processed < num_vectors) {
        int current_batch_size = iminll(rows_in_batch, num_vectors - rows_processed);
        fread((void*) batch, current_batch_size, row_size, input_file);

        for (int i = 0; i < current_batch_size; ++i) {
            *(output++) = batch[dimension_to_load + 1];
        }

        rows_processed += current_batch_size;
    }

    free(batch);
#else
    int row_size = make_dataset_row_size(num_dimensions);
    fseek(input_file, DATASET_ITEM_SIZE * dimension_to_load + DATASET_DIMS_ITEM_SIZE, SEEK_SET);
    for (int i = 0; i < num_vectors; ++i) {
        fread((void*) output++, 1, sizeof(*output), input_file);
        fseek(input_file, row_size - DATASET_ITEM_SIZE, SEEK_CUR);
    }
#endif
}

// Blocks splits

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
int is_vector_in_block(float* vector, int num_dimensions, blocks_info_t* blocks_info,
                       long long block_id);
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

int float_cmp(const void* left, const void* right);

int float_cmp(const void* left, const void* right) {
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

int is_vector_in_block(float* vector, int num_dimensions, blocks_info_t* blocks_info,
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

// Temp file


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

int main(int argc, const char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);

    config_t config = parse_args(argc, argv);
    if (config.init_temp_file) {
        init_temp_file(config.temp_file, config.num_vectors, config.num_nn);
    }
    printf("Temp file: Done\n");

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
    }
    printf("Blocks info: Done\n");

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
