#ifndef _CONCURRENT_QUEUE_H
#define _CONCURRENT_QUEUE_H

#include <pthread.h>
#include <semaphore.h>

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
void* concurrent_queue_pop(concurrent_queue_t* queue);
void* concurrent_queue_try_pop(concurrent_queue_t* queue);
void* concurrent_queue_try_pop_noblock(concurrent_queue_t* queue);

void concurrent_queue_print(concurrent_queue_t* queue, const char* action);


#endif // _CONCURRENT_QUEUE_H