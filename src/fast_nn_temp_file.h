#ifndef _FAST_NN_TEMP_FILE_H
#define _FAST_NN_TEMP_FILE_H

#include <stdio.h>
#include <pthread.h>

#include "misc.h"
// Max heap
// TODO: generalize and move to common code

typedef struct _nn_item {
    vector_id_t index;
    float dist;
} nn_item_t;

int fast_nn_heap_search_indices_collisions(nn_item_t* begin, nn_item_t* end,
                                           const nn_item_t* item);
void fast_nn_heap_push(nn_item_t* heap, const nn_item_t* item, int num_nn);
void fast_nn_heap_sort(nn_item_t* heap, int num_nn, vector_id_t* indices, float* dists);

// Temp file

void init_temp_file(const char* temp_file_name, long long num_vectors, int num_nn);
int make_temp_file_row_size(int num_nn);
void init_empty_batch_for_temp_file(nn_item_t* batch, int num_nn, long long rows_in_batch);
void init_temp_file_batches(FILE* temp_file, long long num_vectors, int num_nn,
                            long long rows_in_batch);

void partial_load_temp_file(const char* temp_file_name, int num_nn, long long num_vectors_to_load,
                            const long long* vectors_indices_to_load,
                            nn_item_t* output_vectors);
void partial_save_temp_file(const char* temp_file_name, int num_nn, long long num_vectors_to_save,
                            const long long* vectors_indices_to_save,
                            nn_item_t* vectors_to_save);

typedef struct _temp_file_loader {
    const char* temp_file_name;
    int num_nn;
    nn_item_t* vectors;
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

void temp_file_to_result(const char* temp_file, const char* indices_file, const char* dists_file,
                         long long num_vectors, int num_nn);
void temp_file_to_result_batches(FILE* temp_file, FILE* indices_file, FILE* dists_file,
                                 long long num_vectors, int num_nn, long long batch_size);


#endif // _FAST_NN_TEMP_FILE_H