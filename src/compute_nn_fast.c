#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#include <pthread.h>
#include <semaphore.h>

#include <yael/nn.h>

// #define _QUEUE_DEBUG

// Utils

int imin(int a, int b) {
    return (a < b ? a : b);
}

long long iminll(long long a, long long b) {
    return (a < b ? a : b);
}

#ifndef __linux__
#   error "need usleep function!"
#endif
#include <unistd.h>
int usleep(unsigned usec);

#define MS 1000

#define DATASET_ITEM_SIZE 4L
#define DATASET_DIMS_ITEM_SIZE 4L
#define NUM_ROWS_IN_BATCH 10000
#define NUM_ROWS_IN_BATCH_BATCH 500

int make_dataset_row_size(int num_dimensions) {
    return num_dimensions * DATASET_ITEM_SIZE + DATASET_DIMS_ITEM_SIZE;
}

// Max heap

typedef struct _temp_file_item {
    long long index;
    float dist;
} temp_file_item_t;

int search_indices_collisions(temp_file_item_t* begin, temp_file_item_t* end,
                              const temp_file_item_t* item);
void heap_push(temp_file_item_t* heap, const temp_file_item_t* item, int num_nn);
void heap_push_impl(temp_file_item_t* heap, const temp_file_item_t* item, int num_nn);
void heap_sort(temp_file_item_t* heap, int num_nn, long long* indices, float* dists);

int search_indices_collisions(temp_file_item_t* begin, temp_file_item_t* end,
                              const temp_file_item_t* item)
{
    for (temp_file_item_t* i = begin; i != end; ++i) {
        if (i->index == item->index) {
            return 1;
        }
    }

    return 0;
}

void heap_push(temp_file_item_t* heap, const temp_file_item_t* item, int num_nn) {
    if (item->dist >= heap[0].dist || search_indices_collisions(heap, heap + num_nn, item)) {
        return; // Drop this new item
    }

    heap_push_impl(heap, item, num_nn);
}

void heap_push_impl(temp_file_item_t* heap, const temp_file_item_t* item, int num_nn) {
    int item_index = 0;
    while (1) {
        int left_child_index = 2 * (item_index + 1) - 1;
        int right_child_index = 2 * (item_index + 1);

        int index_to_swap = item_index;
        if (right_child_index < num_nn
            && heap[right_child_index].dist > heap[left_child_index].dist
            && heap[right_child_index].dist > item->dist)
        {
            index_to_swap = right_child_index;
        } else if (left_child_index < num_nn && heap[left_child_index].dist > item->dist) {
            index_to_swap = left_child_index;
        } else {
            break;
        }

        heap[item_index] = heap[index_to_swap];
        item_index = index_to_swap;
    }

    heap[item_index] = *item;
}

void heap_sort(temp_file_item_t* heap, int num_nn, long long* indices, float* dists) {
    while (--num_nn >= 0) {
        indices[num_nn] = heap[0].index;
        dists[num_nn] = heap[0].dist;

        heap_push_impl(heap, &heap[num_nn], num_nn);
    }
}

// Concurrect queue

typedef struct _concurrent_queue {
    pthread_mutex_t mutex;
    sem_t sem;
    int size;
    int first_datum_index;
    int next_push_index;
    int capacity;
    void** data;
} concurrent_queue_t;

void concurrent_queue_init(concurrent_queue_t* queue, int count);
void concurrent_queue_destroy(concurrent_queue_t* queue);
int concurrent_queue_empty(concurrent_queue_t* queue);
int concurrent_queue_empty_noblock(concurrent_queue_t* queue);
int concurrent_queue_full(concurrent_queue_t* queue);
int concurrent_queue_full_noblock(concurrent_queue_t* queue);
int concurrent_queue_size(concurrent_queue_t* queue);
int concurrent_queue_size_noblock(concurrent_queue_t* queue);
void concurrent_queue_clear(concurrent_queue_t* queue);
void concurrent_queue_clear_noblock(concurrent_queue_t* queue);
int concurrent_queue_push(concurrent_queue_t* queue, void* datum, int round_time, int num_rounds);
int concurrent_queue_try_push(concurrent_queue_t* queue, void* datum);
int concurrent_queue_try_push_noblock(concurrent_queue_t* queue, void* datum);
void* concurrent_queue_try_pop(concurrent_queue_t* queue);
void* concurrent_queue_try_pop_noblock(concurrent_queue_t* queue);

void concurrent_queue_print(concurrent_queue_t* queue, const char* action);

void concurrent_queue_print(concurrent_queue_t* queue, const char* action) {
#ifdef _QUEUE_DEBUG
    int semv = -1;
    sem_getvalue(&queue->sem, &semv);
    printf("Queue %s c=%2d, s=%2d, sem=%2d, d=%2d, p=%2d\n", action, queue->capacity, queue->size,
           semv, queue->first_datum_index, queue->next_push_index);
#endif
}

void concurrent_queue_init(concurrent_queue_t* queue, int count) {
    pthread_mutex_init(&queue->mutex, NULL);
    sem_init(&queue->sem, 0, 0);
    queue->size = 0;
    queue->first_datum_index = 0;
    queue->next_push_index = 0;
    queue->capacity = count;
    queue->data = malloc(sizeof(*queue->data) * count);
}

void concurrent_queue_destroy(concurrent_queue_t* queue) {
    pthread_mutex_destroy(&queue->mutex);
    free(queue->data);
    sem_destroy(&queue->sem);
    queue->size = 0;
    queue->first_datum_index = 0;
    queue->next_push_index = 0;
    queue->capacity = 0;
    queue->data = NULL;
}

int concurrent_queue_empty(concurrent_queue_t* queue) {
    return concurrent_queue_empty_noblock(queue);
}

int concurrent_queue_empty_noblock(concurrent_queue_t* queue) {
    return (queue->size == 0);
}

int concurrent_queue_full(concurrent_queue_t* queue) {
    return concurrent_queue_full_noblock(queue);
}

int concurrent_queue_full_noblock(concurrent_queue_t* queue) {
    return (queue->size == queue->capacity);
}

int concurrent_queue_size(concurrent_queue_t* queue) {
    return concurrent_queue_size_noblock(queue);
}

int concurrent_queue_size_noblock(concurrent_queue_t* queue) {
    return queue->size;
}

void concurrent_queue_clear(concurrent_queue_t* queue) {
    pthread_mutex_lock(&queue->mutex);
    concurrent_queue_clear_noblock(queue);
    pthread_mutex_unlock(&queue->mutex);
}

void concurrent_queue_clear_noblock(concurrent_queue_t* queue) {
    queue->size = 0;
    queue->first_datum_index = queue->next_push_index;
}

int concurrent_queue_push(concurrent_queue_t* queue, void* datum, int round_time, int num_rounds) {
    if (num_rounds < 0) {
        num_rounds = INT_MAX;
    }

    int round = 0;
    int result = 0;
    while (round++ < num_rounds && !(result = concurrent_queue_try_push(queue, datum))) {
        fprintf(stderr, "Cannot push task to concurrent queue\n");
        usleep(round_time * MS);
    }
    return result;
}

int concurrent_queue_try_push(concurrent_queue_t* queue, void* datum) {
    pthread_mutex_lock(&queue->mutex);
    int result = concurrent_queue_try_push_noblock(queue, datum);
    pthread_mutex_unlock(&queue->mutex);
    return result;
}

int concurrent_queue_try_push_noblock(concurrent_queue_t* queue, void* datum) {
    if (concurrent_queue_full_noblock(queue)) {
        return 0;
    }

    queue->data[queue->next_push_index] = datum;
    queue->next_push_index = (queue->next_push_index + 1) % queue->capacity;
    ++queue->size;
    sem_post(&queue->sem);
    concurrent_queue_print(queue, "push");
    return 1;
}

void* concurrent_queue_pop(concurrent_queue_t* queue) {
    void* result = NULL;
    int i = rand() % 100;
    while (!result) {
        sem_wait(&queue->sem);
        result = concurrent_queue_try_pop(queue);
        if (!result) {
            fprintf(stderr, "Queue is broken! got empty result from queue");
            sem_post(&queue->sem);
        }
    }

    return result;
}

void* concurrent_queue_try_pop(concurrent_queue_t* queue) {
    pthread_mutex_lock(&queue->mutex);
    void* result = concurrent_queue_try_pop_noblock(queue);
    pthread_mutex_unlock(&queue->mutex);
    return result;
}

void* concurrent_queue_try_pop_noblock(concurrent_queue_t* queue) {
    if (concurrent_queue_empty_noblock(queue)) {
        return NULL;
    }

    void* result = queue->data[queue->first_datum_index];
    queue->first_datum_index = (queue->first_datum_index + 1) % queue->capacity;
    --queue->size;
    concurrent_queue_print(queue, "pop");
    return result;
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
    int num_splits;
    long long block_id_finder;
    float* block_starts;
    float* block_ends;
} block_dimension_info_t;

typedef struct _blocks_info {
    int num_dimensions;
    int num_splits;
    int num_blocks_per_dim;
    long long num_blocks_total;
    block_dimension_info_t* dimension_infos;
} blocks_info_t;

void dimension_info_build(block_dimension_info_t* dimension_info, const char* input_filename,
                          long long num_vectors, int num_dimensions, int dimension_to_use,
                          int num_blocks, int num_splits);
void dimension_info_destroy(block_dimension_info_t* dimension_info);
void blocks_info_init(blocks_info_t* blocks_info, const char* input_filename, long long num_vectors,
                      int num_dimensions, int num_dimensions_to_split, int num_blocks,
                      int num_splits);
void blocks_info_destroy(blocks_info_t* blocks_info);
int is_vector_in_block(float* vector, int num_dimensions, blocks_info_t* blocks_info,
                       long long block_id);
void get_vectors_in_block(const char* input_filename, blocks_info_t* blocks_info,
                          long long* block_indices, float* block, long long* num_vectors_in_block,
                          long long max_num_vectors_in_block, int num_vectors, int num_dimensions,
                          long long block_id, int num_threads);

typedef struct _batch_filter_worker_task {
    float* batch_start;
    int batch_size;
    long long batch_start_index;
    int done;
    sem_t* sem;
    int is_done_task;
} batch_filter_worker_task_t;

typedef struct _batch_filter_worker_arg {
    concurrent_queue_t* queue;
    pthread_mutex_t* mutex;
    long long* num_vectors_in_block;
    long long max_num_vectors_in_block;
    float* block;
    long long* block_indices;
    blocks_info_t* blocks_info;
    int num_dimensions;
    long long block_id;
    int done;
} batch_filter_worker_arg_t;


void* block_loader_worker_thread(void* arg);
void get_vectors_in_block_batches_mt(FILE* f, blocks_info_t* blocks_info, long long* block_indices,
                                     float* block, long long* num_vectors_in_block,
                                     long long max_num_vectors_in_block, int num_vectors,
                                     int num_dimensions, long long block_id, int num_threads,
                                     int rows_in_batch);
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
                       int num_vectors, int num_dimensions, long long block_id, int num_threads);
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
                          int num_blocks, int num_splits)
{
    float* dimension = malloc(sizeof(*dimension) * num_vectors);
    load_vectors_dim(input_filename, num_vectors, num_dimensions, dimension_to_use, dimension);
    qsort(dimension, num_vectors, sizeof(*dimension), float_cmp);
    dimension_info->dimension = dimension_to_use;
    dimension_info->num_blocks = num_blocks;
    dimension_info->num_splits = num_splits;
    dimension_info->block_id_finder = 1;
    for (int i = 0; i < dimension_to_use; ++i) {
        dimension_info->block_id_finder *= num_blocks;
    }
    dimension_info->block_starts = malloc(sizeof(*dimension_info->block_starts) * num_blocks * 2);
    dimension_info->block_ends = dimension_info->block_starts + num_blocks;
    long long num_vectors_to_split = num_vectors * (num_splits - 1) / num_splits;
    printf("dim %d, nvts %lld\n", dimension_to_use, num_vectors_to_split);
    for (int i = 0; i < num_blocks; ++i) {
        long long start_index = num_vectors_to_split * i / (num_blocks - 1);
        dimension_info->block_starts[i] = dimension[start_index];
        long long end_index = start_index + num_vectors / num_splits;
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
                      int num_splits)
{
    blocks_info->num_dimensions = num_dimensions_to_split;
    blocks_info->num_splits = num_splits;
    blocks_info->num_blocks_per_dim = num_blocks;
    blocks_info->dimension_infos =
            malloc(sizeof(*blocks_info->dimension_infos) * num_dimensions_to_split);
    blocks_info->num_blocks_total = 1;
    for (int i = 0; i < num_dimensions_to_split; ++i) {
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
                             num_dimensions, dimension, num_blocks, num_splits);
        blocks_info->num_blocks_total *= num_blocks;
    }
}

void blocks_info_destroy(blocks_info_t* blocks_info) {
    for (int i = 0; i < blocks_info->num_dimensions; ++i) {
        dimension_info_destroy(blocks_info->dimension_infos + i);
    }
    blocks_info->num_dimensions = 0;
    blocks_info->num_splits = 0;
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
                / dimension_infos->block_id_finder
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
                          long long block_id, int num_threads)
{
    if (num_vectors_in_block) {
        *num_vectors_in_block = 0LL;
    }

    FILE* f = fopen(input_filename, "rb");
    if (num_threads > 0) {
        get_vectors_in_block_batches_mt(f, blocks_info, block_indices, block, num_vectors_in_block,
                                        max_num_vectors_in_block, num_vectors, num_dimensions,
                                        block_id, num_threads, NUM_ROWS_IN_BATCH);
    } else {
        get_vectors_in_block_batches(f, blocks_info, block_indices, block, num_vectors_in_block,
                                     max_num_vectors_in_block, num_vectors, num_dimensions,
                                     block_id, NUM_ROWS_IN_BATCH);
    }
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

void get_vectors_in_block_batches_mt(FILE* f, blocks_info_t* blocks_info, long long* block_indices,
                                     float* block, long long* num_vectors_in_block,
                                     long long max_num_vectors_in_block, int num_vectors,
                                     int num_dimensions, long long block_id, int num_threads,
                                     int rows_in_batch)
{
    int rows_in_batch_batch = NUM_ROWS_IN_BATCH_BATCH;
    int num_tasks = (rows_in_batch + rows_in_batch_batch - 1) / rows_in_batch_batch;
    *num_vectors_in_block = 0;
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);
    concurrent_queue_t queue;
    concurrent_queue_init(&queue, 2 * num_tasks + num_threads);
    pthread_t* threads = malloc(sizeof(*threads) * num_threads);
    batch_filter_worker_arg_t* thread_args = malloc(sizeof(*thread_args) * num_threads);
    batch_filter_worker_task_t* tasks = malloc(sizeof(*tasks) * num_tasks);
    batch_filter_worker_task_t* tasks_swap = malloc(sizeof(*tasks_swap) * num_tasks);
    sem_t tasks_sem_pool[2];
    sem_init(&tasks_sem_pool[0], 0, 0);
    sem_init(&tasks_sem_pool[1], 0, 0);
    sem_t* tasks_sem = &tasks_sem_pool[0];
    sem_t* tasks_sem_swap = &tasks_sem_pool[1];

    for (int i = 0; i < num_threads; ++i) {
        thread_args[i].queue = &queue;
        thread_args[i].mutex = num_threads > 1 ? &mutex : NULL;
        thread_args[i].blocks_info = blocks_info;
        thread_args[i].num_dimensions = num_dimensions;
        thread_args[i].block_id = block_id;
        thread_args[i].block = block;
        thread_args[i].block_indices = block_indices;
        thread_args[i].num_vectors_in_block = num_vectors_in_block;
        thread_args[i].max_num_vectors_in_block = max_num_vectors_in_block;
        thread_args[i].done = 0;

        pthread_create(threads + i, NULL, block_loader_worker_thread, (void*) (thread_args + i));
    }

    for (int i = 0; i < num_tasks; ++i) {
        tasks[i].batch_start = NULL;
        tasks[i].batch_start_index = 0;
        tasks[i].batch_size = rows_in_batch_batch;
        tasks[i].done = 1;
        tasks[i].sem = NULL;
        tasks[i].is_done_task = 0;
    }
    memcpy(tasks_swap, tasks, sizeof(*tasks_swap) * num_tasks);

    int row_size = make_dataset_row_size(num_dimensions);
    float* batch = malloc(row_size * rows_in_batch);
    float* batch_swap = malloc(row_size * rows_in_batch);

    long long rows_processed = 0;
    long long prev_batch_size = 0;
    int prev_num_tasks = 0;
    while (rows_processed < num_vectors) {
        if (max_num_vectors_in_block > 0LL) {
            pthread_mutex_lock(&mutex);
            int got_max_num_vectors = (*num_vectors_in_block > max_num_vectors_in_block);
            pthread_mutex_unlock(&mutex);
            if (got_max_num_vectors) {
                break;
            }
        }

        long long current_batch_size = iminll(rows_in_batch, num_vectors - rows_processed);
        fread(batch_swap, current_batch_size, row_size, f);

        int current_num_tasks = 0;
        for (
            long long row_i = 0;
            row_i < current_batch_size;
            row_i += rows_in_batch_batch, ++current_num_tasks)
        {
            assert(current_num_tasks < num_tasks);

            tasks_swap[current_num_tasks].batch_start = batch_swap + (num_dimensions + 1) * row_i;
            tasks_swap[current_num_tasks].batch_start_index = rows_processed + row_i;
            tasks_swap[current_num_tasks].batch_size =
                    iminll(current_batch_size - row_i, rows_in_batch_batch);
            tasks_swap[current_num_tasks].done = 0;
            tasks_swap[current_num_tasks].sem = tasks_sem_swap;
            // printf("Push %x\n", tasks_swap[current_num_tasks].batch_start_index);
            concurrent_queue_push(&queue, (void*)(tasks_swap + current_num_tasks), 10, -1);
        }
        // printf("Num tasks: %d\n", current_num_tasks);
        for (int i = 0; i < prev_num_tasks; ++i) {
            // int v;
            // sem_getvalue(tasks_sem, &v);
            // printf("Wait %d round for %x, %d, %d\n", i, tasks_sem, v, prev_num_tasks);
            sem_wait(tasks_sem);
        }
        // printf("Got tasks: %d\n", prev_num_tasks);
        int semv;
        sem_getvalue(tasks_sem, &semv);
        assert(!semv);
        for (int i = 0; i < prev_num_tasks; ++i) {
            assert(tasks[i].done);
        }

        float* tmp_batch = batch;
        batch = batch_swap;
        batch_swap = tmp_batch;

        batch_filter_worker_task_t* tmp_tasks = tasks;
        tasks = tasks_swap;
        tasks_swap = tmp_tasks;

        sem_t* tmp_sem = tasks_sem;
        tasks_sem = tasks_sem_swap;
        tasks_sem_swap = tmp_sem;

        prev_batch_size = current_batch_size;
        prev_num_tasks = current_num_tasks;
        rows_processed += current_batch_size;
    }

    for (int i = 0; i < num_threads; ++i) {
        tasks_swap[i].is_done_task = 1;
        concurrent_queue_push(&queue, (void*)(tasks_swap + i), 10, -1);
    }
    for (int i = 0; i < prev_num_tasks; ++i) {
        // printf("Wait %d round for %x\n", i, tasks_sem);
        sem_wait(tasks_sem);
    }
    for (int i = 0; i < num_threads; ++i) {
        thread_args[i].done = 1;
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    sem_destroy(&tasks_sem_pool[0]);
    sem_destroy(&tasks_sem_pool[1]);
    free(threads);
    free(thread_args);
    concurrent_queue_destroy(&queue);
    free(tasks);
    free(tasks_swap);
    free(batch);
    free(batch_swap);
}

void* block_loader_worker_thread(void* arg) {
    batch_filter_worker_arg_t* arg_data = (batch_filter_worker_arg_t*) arg;
    while (!arg_data->done) {
        void* task_raw = concurrent_queue_pop(arg_data->queue);
        if (!task_raw) {
            fprintf(stderr, "queue is empty task\n");
            usleep(15 * MS);
            continue;
        }

        batch_filter_worker_task_t* task = (batch_filter_worker_task_t*) task_raw;
        if (task->is_done_task) {
            break;
        }

        block_loader_filter_block(arg_data->mutex, arg_data->blocks_info, arg_data->block_indices,
                                  arg_data->block, arg_data->num_vectors_in_block,
                                  arg_data->max_num_vectors_in_block, arg_data->num_dimensions,
                                  arg_data->block_id, task->batch_start_index, task->batch_size,
                                  task->batch_start);

        task->done = 1;
        // int v;
        // sem_getvalue(task->sem, &v);
        // printf("Done and post about %d to %x, %d\n", task->batch_start_index, task->sem, v);
        sem_post(task->sem);
    }
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
                       int num_vectors, int num_dimensions, long long block_id, int num_threads)
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
    block_loader->num_threads = num_threads;
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
                         block_loader->block_id, block_loader->num_threads);
}

// Temp file

void init_temp_file(const char* temp_file_name, long long num_vectors, int num_nn);
int make_temp_file_row_size(int num_nn);
void init_empty_batch_for_temp_file(temp_file_item_t* batch, int num_nn, long long rows_in_batch);
void init_temp_file_batches(FILE* temp_file, long long num_vectors, int num_nn,
                            long long rows_in_batch);

void partial_load_temp_file(const char* temp_file_name, int num_nn, long long num_vectors_to_load,
                            const long long* vectors_indices_to_load,
                            temp_file_item_t* output_vectors);
void partial_save_temp_file(const char* temp_file_name, int num_nn, long long num_vectors_to_save,
                            const long long* vectors_indices_to_save,
                            temp_file_item_t* vectors_to_save);

typedef struct _temp_file_loader {
    const char* temp_file_name;
    int num_nn;
    temp_file_item_t* vectors;
    long long vectors_capacity;
    long long num_vectors_to_load;
    long long* vectors_indices;
    pthread_t thread;
    int thread_active;
} temp_file_loader_t;

void temp_file_loader_init(temp_file_loader_t* loader, const char* temp_file_name, int num_nn,
                           long long initial_capacity);
void _temp_file_loader_alloc_vectors(temp_file_loader_t* loader, long long capacity);
void temp_file_loader_load(temp_file_loader_t* loader, long long num_vectors_to_load,
                           long long* vectors_indices);
void temp_file_loader_load_join(temp_file_loader_t* loader);
void temp_file_loader_save(temp_file_loader_t* loader);
void temp_file_loader_save_join(temp_file_loader_t* loader);
void temp_file_loader_join(temp_file_loader_t* loader);
void temp_file_loader_destroy(temp_file_loader_t* loader);

void* _temp_file_loader_load_thread(void* arg);
void* _temp_file_loader_save_thread(void* arg);

void temp_file_to_result(const char* temp_file, const char* indices_file, const char* dists_file,
                         long long num_vectors, int num_nn);
void temp_file_to_result_batches(FILE* temp_file, FILE* indices_file, FILE* dists_file,
                                 long long num_vectors, int num_nn, long long batch_size);


void init_temp_file(const char* temp_file_name, long long num_vectors, int num_nn) {
    FILE* f = fopen(temp_file_name, "wb");
    init_temp_file_batches(f, num_vectors, num_nn, NUM_ROWS_IN_BATCH);
    fclose(f);
}

int make_temp_file_row_size(int num_nn) {
    return num_nn * sizeof(temp_file_item_t);
}

void init_empty_batch_for_temp_file(temp_file_item_t* batch, int num_nn, long long rows_in_batch) {
    int row_size = make_temp_file_row_size(num_nn);

    // NOTE: init first row directly
    for (int i = 0; i < num_nn; ++i) {
        batch[i].index = UINT_MAX;
        batch[i].dist = INFINITY;
    }
    // NOTE: copy prefix of batch
    int batch_filled_rows = 1;
    while (batch_filled_rows < rows_in_batch) {
        int rows_to_copy = iminll(batch_filled_rows, rows_in_batch - batch_filled_rows);
        memcpy(batch + num_nn * batch_filled_rows, batch, row_size * rows_to_copy);
        batch_filled_rows += rows_to_copy;
    }
}

void init_temp_file_batches(FILE* temp_file, long long num_vectors, int num_nn,
                            long long rows_in_batch)
{
    if (num_vectors < rows_in_batch) {
        rows_in_batch = num_vectors;
    }

    long long row_size = make_temp_file_row_size(num_nn);
    temp_file_item_t* batch = malloc(row_size * rows_in_batch);
    init_empty_batch_for_temp_file(batch, num_nn, rows_in_batch);

    long long num_rows_written = 0;
    while (num_rows_written < num_vectors) {
        long long num_rows_to_write = iminll(rows_in_batch, num_vectors - num_rows_written);
        fwrite(batch, row_size, num_rows_to_write, temp_file);
        num_rows_written += num_rows_to_write;
    }

    free(batch);
}

void partial_load_temp_file(const char* temp_file_name, int num_nn, long long num_vectors_to_load,
                            const long long* vectors_indices_to_load,
                            temp_file_item_t* output_vectors)
{
    FILE* f = fopen(temp_file_name, "rb");

    long long row_size = make_temp_file_row_size(num_nn);
    for (
        long long i = 0;
        i < num_vectors_to_load;
        ++i, ++vectors_indices_to_load, output_vectors += num_nn)
    {
        long long vector_index = *vectors_indices_to_load;
        fseek(f, row_size * vector_index, SEEK_SET);
        fread((void*) output_vectors, 1, row_size, f);
    }

    fclose(f);
}

void partial_save_temp_file(const char* temp_file_name, int num_nn, long long num_vectors_to_save,
                            const long long* vectors_indices_to_save,
                            temp_file_item_t* vectors_to_save)
{
    FILE* f = fopen(temp_file_name, "r+b");

    long long row_size = make_temp_file_row_size(num_nn);
    for (
        long long i = 0;
        i < num_vectors_to_save;
        ++i, ++vectors_indices_to_save, vectors_to_save += num_nn)
    {
        long long vector_index = *vectors_indices_to_save;
        fseek(f, row_size * vector_index, SEEK_SET);
        fwrite((void*) vectors_to_save, 1, row_size, f);
    }

    fclose(f);
}


void temp_file_loader_init(temp_file_loader_t* loader, const char* temp_file_name, int num_nn,
                           long long initial_capacity)
{
    loader->temp_file_name = temp_file_name;
    loader->num_nn = num_nn;
    loader->vectors_capacity = 0;
    loader->vectors = NULL;
    _temp_file_loader_alloc_vectors(loader, initial_capacity);
    loader->num_vectors_to_load = 0;
    loader->vectors_indices = NULL;
    loader->thread_active = 0;
}

void _temp_file_loader_alloc_vectors(temp_file_loader_t* loader, long long capacity) {
    if (loader->vectors_capacity >= capacity) {
        return;
    }

    if (loader->vectors) {
        free(loader->vectors);
        loader->vectors = NULL;
    }
    loader->vectors_capacity = capacity;
    long long row_size = make_temp_file_row_size(loader->num_nn);
    loader->vectors = malloc(row_size * loader->vectors_capacity);
}

void temp_file_loader_load(temp_file_loader_t* loader, long long num_vectors_to_load,
                           long long* vectors_indices)
{
    if (loader->thread_active) {
        fprintf(stderr, "Loader is busy. Wait for thread join");
        temp_file_loader_join(loader);
    }
    loader->num_vectors_to_load = num_vectors_to_load;
    loader->vectors_indices = vectors_indices;
    if (loader->num_vectors_to_load > loader->vectors_capacity) {
        _temp_file_loader_alloc_vectors(loader, loader->num_vectors_to_load);
    }
    loader->thread_active = 1;
    pthread_create(&loader->thread, NULL, _temp_file_loader_load_thread, (void*)loader);
}

void temp_file_loader_load_join(temp_file_loader_t* loader) {
    temp_file_loader_join(loader);
}

void temp_file_loader_save(temp_file_loader_t* loader) {
    if (loader->thread_active) {
        fprintf(stderr, "Loader is busy. Wait for thread join");
        temp_file_loader_join(loader);
    }

    loader->thread_active = 1;
    pthread_create(&loader->thread, NULL, _temp_file_loader_save_thread, (void*)loader);
}

void temp_file_loader_save_join(temp_file_loader_t* loader) {
    temp_file_loader_join(loader);
}

void temp_file_loader_join(temp_file_loader_t* loader) {
    if (loader->thread_active) {
        pthread_join(loader->thread, NULL);
    }
    loader->thread_active = 0;
}

void temp_file_loader_destroy(temp_file_loader_t* loader) {
    temp_file_loader_join(loader);
    loader->temp_file_name = NULL;
    loader->num_nn = 0;
    if (loader->vectors) {
        free(loader->vectors);
    }
    loader->vectors = NULL;
    loader->vectors_capacity = 0;
    loader->num_vectors_to_load = 0;
    loader->vectors_indices = NULL;
}

void* _temp_file_loader_load_thread(void* arg) {
    temp_file_loader_t* loader = (temp_file_loader_t*)arg;
    partial_load_temp_file(loader->temp_file_name, loader->num_nn, loader->num_vectors_to_load,
                           loader->vectors_indices, loader->vectors);
}

void* _temp_file_loader_save_thread(void* arg) {
    temp_file_loader_t* loader = (temp_file_loader_t*)arg;
    partial_save_temp_file(loader->temp_file_name, loader->num_nn, loader->num_vectors_to_load,
                           loader->vectors_indices, loader->vectors);
}

// Temp file to result
void temp_file_to_result(const char* temp_file, const char* indices_file, const char* dists_file,
                         long long num_vectors, int num_nn)
{
    FILE* temp_f = fopen(temp_file, "rb");
    FILE* indices_f = fopen(indices_file, "wb");
    FILE* dists_f = fopen(dists_file, "wb");

    temp_file_to_result_batches(temp_f, indices_f, dists_f, num_vectors, num_nn, NUM_ROWS_IN_BATCH);

    fclose(temp_f);
    fclose(indices_f);
    fclose(dists_f);
}

void temp_file_to_result_batches(FILE* temp_file, FILE* indices_file, FILE* dists_file,
                                 long long num_vectors, int num_nn, long long batch_size)
{
    long long tempfile_row_size = make_temp_file_row_size(num_nn);
    long long indices_row_size = sizeof(long long) * num_nn;
    long long dists_row_size = sizeof(float) * num_nn;

    temp_file_item_t* tempfile_batch = malloc(tempfile_row_size * batch_size);
    long long* indices_batch = malloc(indices_row_size * batch_size);
    float* dists_batch = malloc(dists_row_size * batch_size);

    int num_vectors_int = num_vectors;
    fwrite(&num_vectors_int, 1, sizeof(num_vectors_int), indices_file);
    fwrite(&num_nn, 1, sizeof(num_nn), indices_file);
    fwrite(&num_vectors_int, 1, sizeof(num_vectors_int), dists_file);
    fwrite(&num_nn, 1, sizeof(num_nn), dists_file);

    long long rows_processed = 0;
    while (rows_processed < num_vectors) {
        long long current_batch_size = iminll(batch_size, num_vectors - rows_processed);

        fread(tempfile_batch, tempfile_row_size, current_batch_size, temp_file);

        for (long long row_index = 0; row_index < current_batch_size; ++row_index) {
            heap_sort(tempfile_batch + num_nn * row_index, num_nn,
                      indices_batch + num_nn * row_index, dists_batch + num_nn * row_index);
        }

        fwrite(indices_batch, indices_row_size, current_batch_size, indices_file);
        fwrite(dists_batch, dists_row_size, current_batch_size, dists_file);

        rows_processed += current_batch_size;
    }

    free(dists_batch);
    free(indices_batch);
    free(tempfile_batch);
}

// Parse args

typedef struct _config {
    // NOTE: Required args
    const char* input_filename;
    const char* output_files_template;
    int num_nn;

    // NOTE: Optional args
    const char* temp_file;
    int delete_temp_file;
    int num_dims_to_split;
    int num_splits_per_dim;
    int num_blocks_per_dim;
    int num_threads;
    int with_blocks_stat;

    // NOTE: computed args
    const char* output_indices_filename;
    const char* output_dists_filename;

    FILE* blocks_stat_file;

    // NOTE: dataset info
    int num_vectors;
    int num_dimensions;
} config_t;

typedef struct _dataset_metainfo {
    int num_vectors;
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
    fprintf(stderr, "  --num-dims <value>                         num dimensions to process (default is 5)\n");
    // fprintf(stderr, "  --num-splits-per-dim <value>               num parts to split each dimension (default is 2)\n");
    // fprintf(stderr, "  --num-blocks-per-dim <value>               num blocks for each dimension (default is 2)\n");
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
    printf("num vectors %lld * row_size %lld (num_dim %lld) == filesize %lld\n", metainfo.num_vectors,
           row_size, metainfo.num_dimensions, filesize);
    assert(metainfo.num_vectors * row_size == filesize);
    return metainfo;
}

static const char* concat(const char* prefix, const char* suffix) {
    int prefix_length = strlen(prefix);
    int suffix_length = strlen(suffix);

    char* result = malloc(sizeof(char) * (prefix_length + suffix_length + 1));
    memcpy(result, prefix, sizeof(char) * prefix_length);
    memcpy(result + prefix_length, suffix, sizeof(char) * suffix_length);
    result[prefix_length + suffix_length] = '\0';
    return result;
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
    config.delete_temp_file = 1;
    config.num_dims_to_split = 5;
    config.num_splits_per_dim = 2;
    config.num_blocks_per_dim = 3;
    config.num_threads = 1;
    config.with_blocks_stat = 0;

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
        } else if (!strcmp(argv[arg_index], "--temp-file")) {
            config.temp_file = argv[++arg_index];
        } else if (!strcmp(argv[arg_index], "--num-dims")) {
            config.num_dims_to_split = atoi(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--num-splits-per-dim")) {
            config.num_splits_per_dim = atoi(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--num-blocks-per-dim")) {
            config.num_blocks_per_dim = atoi(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--num-threads")) {
            config.num_threads = atoi(argv[++arg_index]);
        } else {
            fprintf(stderr, "Unknown arg %s\n", argv[arg_index]);
            ok = 0;
        }
        ++arg_index;
    }

    if (!ok) {
        exit(1);
    }

    config.output_indices_filename = concat(config.output_files_template, "nn_indices.lvecsl");
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
                              config->num_dimensions, block_id + i, config->num_threads);
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
                      config->num_dimensions, 0LL, config->num_threads);

    long long result_capacity = block_capacities[0];
    long long result_indices_row_size = sizeof(int) * (config->num_nn + 1);
    long long result_dists_row_size = sizeof(float) * (config->num_nn + 1);
    int* result_indices = malloc(result_indices_row_size * result_capacity);
    float* result_dists = malloc(result_dists_row_size * result_capacity);

    for (long long block_id = 0; block_id < blocks_info->num_blocks_total; ++block_id) {
        int active_buffer = block_id % NUM_BUFFERS;
        int next_buffer = (active_buffer + 1) % NUM_BUFFERS;
        int prev_buffer = (active_buffer - 1 + NUM_BUFFERS) % NUM_BUFFERS;

        // printf("Starting block %lld\n", block_id);
        if (block_id + 1 != blocks_info->num_blocks_total) {
            temp_file_loader_save_join(&temp_file_loaders[next_buffer]);
            block_loader_init(&block_loaders[next_buffer], config->input_filename, blocks_info,
                              block_indices[next_buffer], blocks[next_buffer],
                              &block_sizes[next_buffer], block_capacities[next_buffer],
                              config->num_vectors, config->num_dimensions, block_id + 1, 0);
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
                              config->num_vectors, config->num_dimensions, block_id,
                              config->num_threads);
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
        temp_file_item_t temp_item;
        for (long long vec_index = 0; vec_index < block_size; ++vec_index) {
            temp_file_item_t* heap = temp_file_loaders[active_buffer].vectors
                    + config->num_nn * vec_index;
            int* vec_indices = result_indices + (num_nn_real + 1) * vec_index;
            float* vec_dists = result_dists + (num_nn_real + 1) * vec_index;
            for (int nn_index = 1; nn_index <= num_nn_real; ++nn_index) {
                // FIXME: Mapping to original indices!
                temp_item.index = block_indices[active_buffer][vec_indices[nn_index]];
                temp_item.dist = vec_dists[nn_index];

                float real_dist = get_real_dist(
                        blocks[active_buffer] + vec_index * config->num_dimensions,
                        blocks[active_buffer] + vec_indices[nn_index] * config->num_dimensions,
                        config->num_dimensions);
                heap_push(heap, &temp_item, config->num_nn);
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
    config_t config = parse_args(argc, argv);
    init_temp_file(config.temp_file, config.num_vectors, config.num_nn);

    blocks_info_t blocks_info;
    blocks_info_init(&blocks_info, config.input_filename, config.num_vectors, config.num_dimensions,
                     config.num_dims_to_split, config.num_blocks_per_dim,
                     config.num_splits_per_dim);

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
