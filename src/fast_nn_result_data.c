#include "fast_nn_result_data.h"

#include <stdio.h>
#include <stdlib.h>

void result_data_init(result_data_t* result_data, long long initial_capacity) {
    result_data->capacity = 0;
    result_data->indices = NULL;
    result_data->dists = NULL;

    result_data_realloc(result_data, initial_capacity);
}

static void result_data_free_data(result_data_t* result_data) {
    if (result_data->indices) {
        free(result_data->indices);
        result_data->indices = NULL;
    }

    if (result_data->dists) {
        free(result_data->dists);
        result_data->dists = NULL;
    }
}

void result_data_realloc(result_data_t* result_data, long long new_capacity) {
    if (result_data->capacity >= new_capacity) {
        return;
    }
    result_data_free_data(result_data);

    result_data->capacity = new_capacity;
    result_data->indices = malloc(sizeof(*result_data->indices) * result_data->capacity);
    result_data->dists = malloc(sizeof(*result_data->dists) * result_data->capacity);
}

void result_data_destroy(result_data_t* result_data) {
    result_data_free_data(result_data);
    result_data->capacity = 0;
}
