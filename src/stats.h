#ifndef _HUFFMAN_STATS_H
#define _HUFFMAN_STATS_H

#include <stdio.h>

typedef struct _huffman_stats {
    long long num_vectors;
    int m;
    int k_star;

    double sum_length;
    double* partial_lengths;
} huffman_stats_t;

void huffman_stats_init(huffman_stats_t* stats, long long num_vectors, int m, int k_star);
void huffman_stats_destroy(huffman_stats_t* stats);

void huffman_stats_push(huffman_stats_t* stats, int part, double length);

void huffman_stats_print(const huffman_stats_t* stats);
void huffman_stats_print_filename(const huffman_stats_t* stats, const char* filename);
void huffman_stats_print_file(const huffman_stats_t* stats, FILE* file);

#endif // _HUFFMAN_STATS_H
