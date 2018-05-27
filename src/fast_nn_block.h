#ifndef _FAST_NN_BLOCK_H_
#define _FAST_NN_BLOCK_H_

typedef struct _block {
    long long id;
    int num_dimensions;
    long long capacity;
    long long* indices;
    float* data;
    long long size;
} block_t;

enum {
    BLOCK_INIT_WITH_DATA = 0x01,
    BLOCK_INIT_WITH_INDICES = 0x02,

    BLOCK_INIT_ALL = 0xff
};

void block_init(block_t* block, long long block_id, int num_dimensions, long long initial_capacity,
                int init_flags);
void block_destroy(block_t* block);
void block_push(block_t* block, long long index, const float* vector);
void block_realloc(block_t* block, long long new_capacity);
void block_set_id(block_t* block, long long block_id);

#endif // _FAST_NN_BLOCK_H_
