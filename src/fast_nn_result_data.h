#ifndef _FAST_NN_RESUT_DATA_H_
#define _FAST_NN_RESUT_DATA_H_

typedef struct _result_data {
    long long capacity;
    int* indices;
    float* dists;
} result_data_t;

void result_data_init(result_data_t* result_data, long long initial_capacity);
void result_data_realloc(result_data_t* result_data, long long minimal_capacity);
void result_data_destroy(result_data_t* result_data);

#endif // _FAST_NN_RESUT_DATA_H_
