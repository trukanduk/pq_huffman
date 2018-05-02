#include "stats.h"

#include <stdlib.h>

void huffman_stats_init(huffman_stats_t* stats, long long num_vectors, int m, int k_star) {
    stats->num_vectors = num_vectors;
    stats->m = m;
    stats->k_star = k_star;

    stats->sum_length = 0.0;
    stats->partial_lengths = malloc(sizeof(*stats->partial_lengths) * stats->m);
    for (double* length_it = stats->partial_lengths;
         length_it != stats->partial_lengths + stats->m;
         ++length_it)
    {
        *length_it = 0.0;
    }
}

void huffman_stats_destroy(huffman_stats_t* stats) {
    stats->num_vectors = 0;
    stats->m = 0;
    stats->k_star = 0;
    stats->sum_length = 0.0;
    if (stats->partial_lengths) {
        free(stats->partial_lengths);
        stats->partial_lengths = NULL;
    }
}

void huffman_stats_push(huffman_stats_t* stats, int part, double value) {
    stats->sum_length += value;
    stats->partial_lengths[part] = value;
}

void huffman_stats_print(const huffman_stats_t* stats) {
    huffman_stats_print_file(stats, stdout);
}

void huffman_stats_print_filename(const huffman_stats_t* stats, const char* filename) {
    FILE* file = fopen(filename, "a");
    huffman_stats_print_file(stats, file);
    fclose(file);
}

static void huffman_stats_print_stats_impl(FILE* file, double bit_length,
                                           const huffman_stats_t* stats) {
    fprintf(file, "\"length_bit\": %.1lf, \"length_bytes\": %.1lf, ", bit_length, bit_length / 8);
    fprintf(file, "\"compression_rate\": %.3lf, \"bits_per_byte\": %.2lf",
            bit_length / (stats->num_vectors * stats->m * 8),
            bit_length / (stats->num_vectors * stats->m));
}

void huffman_stats_print_file(const huffman_stats_t* stats, FILE* file) {
    fprintf(file, "{\"num_vectors\": %lld, \"m\": %d, \"k_star\": %d, ",
            stats->num_vectors, stats->m, stats->k_star);
    huffman_stats_print_stats_impl(file, stats->sum_length, stats);
    fprintf(file, ", \"partials\": [");
    for (int i = 0; i < stats->m; ++i) {
        if (i > 0) {
            fprintf(file, ", ");
        }
        fprintf(file, "{");
        huffman_stats_print_stats_impl(file, stats->partial_lengths[i], stats);
        fprintf(file, "}");
    }
    fprintf(file, "]}\n");
}
