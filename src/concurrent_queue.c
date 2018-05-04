#include "concurrent_queue.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "misc.h"

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
