#include "fast_nn_temp_file.h"

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#include "vecs_io.h"
// Max heap

int fast_nn_heap_search_indices_collisions(nn_item_t* begin, nn_item_t* end,
                                           const nn_item_t* item)
{
    for (nn_item_t* i = begin; i != end; ++i) {
        if (i->index == item->index) {
            return 1;
        }
    }

    return 0;
}

static void fast_nn_heap_push_impl(nn_item_t* heap, const nn_item_t* item, int num_nn) {
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

void fast_nn_heap_push(nn_item_t* heap, const nn_item_t* item, int num_nn) {
    if (item->dist >= heap[0].dist || fast_nn_heap_search_indices_collisions(heap, heap + num_nn, item)) {
        return; // Drop this new item
    }

    fast_nn_heap_push_impl(heap, item, num_nn);
}

void fast_nn_heap_sort(nn_item_t* heap, int num_nn, vector_id_t* indices, float* dists) {
    while (--num_nn >= 0) {
        indices[num_nn] = heap[0].index;
        dists[num_nn] = heap[0].dist;

        fast_nn_heap_push_impl(heap, &heap[num_nn], num_nn);
    }
}

// Temp file

enum {
    DEFAULT_BATCH_SIZE = 10000
};

static void* temp_file_loader_load_thread(void* arg) {
    temp_file_loader_t* loader = (temp_file_loader_t*)arg;
    partial_load_temp_file(loader->temp_file_name, loader->num_nn, loader->num_vectors_to_load,
                           loader->vectors_indices, loader->vectors);
}

static void* temp_file_loader_save_thread(void* arg) {
    temp_file_loader_t* loader = (temp_file_loader_t*)arg;
    partial_save_temp_file(loader->temp_file_name, loader->num_nn, loader->num_vectors_to_load,
                           loader->vectors_indices, loader->vectors);
}

void init_temp_file(const char* temp_file_name, long long num_vectors, int num_nn) {
    FILE* f = fopen(temp_file_name, "wb");
    init_temp_file_batches(f, num_vectors, num_nn, DEFAULT_BATCH_SIZE);
    fclose(f);
}

int make_temp_file_row_size(int num_nn) {
    return num_nn * sizeof(nn_item_t);
}

void init_empty_batch_for_temp_file(nn_item_t* batch, int num_nn, long long rows_in_batch) {
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
    nn_item_t* batch = malloc(row_size * rows_in_batch);
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
                            nn_item_t* output_vectors)
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
                            nn_item_t* vectors_to_save)
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
    pthread_create(&loader->thread, NULL, temp_file_loader_load_thread, (void*)loader);
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
    pthread_create(&loader->thread, NULL, temp_file_loader_save_thread, (void*)loader);
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

// Temp file to result
void temp_file_to_result(const char* temp_file, const char* indices_file, const char* dists_file,
                         long long num_vectors, int num_nn)
{
    FILE* temp_f = fopen(temp_file, "rb");
    FILE* indices_f = fopen(indices_file, "wb");
    FILE* dists_f = fopen(dists_file, "wb");

    temp_file_to_result_batches(temp_f, indices_f, dists_f, num_vectors, num_nn, DEFAULT_BATCH_SIZE);

    fclose(temp_f);
    fclose(indices_f);
    fclose(dists_f);
}

void temp_file_to_result_batches(FILE* temp_file, FILE* indices_file, FILE* dists_file,
                                 long long num_vectors, int num_nn, long long batch_size)
{
    long long tempfile_row_size = make_temp_file_row_size(num_nn);
    long long indices_row_size = sizeof(vector_id_t) * num_nn;
    long long dists_row_size = sizeof(float) * num_nn;

    nn_item_t* tempfile_batch = malloc(tempfile_row_size * batch_size);
    vector_id_t* indices_batch = malloc(indices_row_size * batch_size);
    float* dists_batch = malloc(dists_row_size * batch_size);

    save_vecs_light_meta_file(indices_file, num_vectors, num_nn);
    save_vecs_light_meta_file(dists_file, num_vectors, num_nn);

    long long rows_processed = 0;
    while (rows_processed < num_vectors) {
        long long current_batch_size = iminll(batch_size, num_vectors - rows_processed);

        fread(tempfile_batch, tempfile_row_size, current_batch_size, temp_file);

        for (long long row_index = 0; row_index < current_batch_size; ++row_index) {
            fast_nn_heap_sort(tempfile_batch + num_nn * row_index, num_nn,
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

