#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bitstream.h"
#include "huffman.h"
#include "stats.h"

typedef struct _config {
    const char* pq_input_template;
    const char* output_template;
    int m;
    int k_star;
    int only_estimate;
    int sort;
    int context;

    char* pq_input_indices;
    // char* pq_input_centroids;
    char* output_codebooks;
    char* output_encoded_indices;
    char* output_stats;

    long long num_vectors;
} config_t;

static char* concat(const char* left, const char* right) {
    int left_length = strlen(left);
    int right_length = strlen(right);
    char* result = malloc(sizeof(char) * (left_length + right_length + 1));
    char* res_it = result;
    for (const char* left_it = left; *left_it; ++left_it) {
        *(res_it++) = *left_it;
    }
    for (const char* right_it = right; *right_it; ++right_it) {
        *(res_it++) = *right_it;
    }
    *res_it = '\0';
    return result;
}

static void print_help(const char* argv0) {
    fprintf(stderr, "Usage: %s <pq-output-template> <output-template> <m>"
                    " [--no-sort] [--no-context] [--only-estimate]\n", argv0);
    exit(1);
}

static long long load_num_vectors(const char* filename, int m) {
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    fclose(f);

    return file_size / m;
}

static void parse_args(config_t* config, int argc, const char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Too few positional arguments\n");
        print_help(argv[0]);
    }

    config->pq_input_template = argv[1];
    config->output_template = argv[2];
    config->m = atoi(argv[3]);
    config->only_estimate = 0;
    config->sort = 1;
    config->context = 1;

    for (const char** arg = argv + 4; *arg; ++arg) {
        if (!strcmp(*arg, "--only-estimate")) {
            config->only_estimate = 1;
        } else if (!strcmp(*arg, "--no-sort")) {
            config->sort = 0;
            fprintf(stderr, "WARNING: --no-sort may not be implemented yet\n");
        } else if (!strcmp(*arg, "--no-context")) {
            config->context = 0;
            fprintf(stderr, "WARNING: --no-context may not be implemented yet\n");
        } else {
            fprintf(stderr, "Unknown arg: %s\n", *arg);
            print_help(argv[0]);
        }
    }

    config->k_star = (1 << 8); // TODO:

    config->pq_input_indices = concat(config->pq_input_template, "pq_indices.bvecsl");
    // config->pq_input_centroids = concat(config->pq_input_template, "pq_centroids.fvecsl");
    config->output_codebooks = concat(config->output_template, "simple_huffman_codebooks.bin");
    config->output_encoded_indices = concat(config->output_template, "simple_huffman_indices.bin");
    config->output_stats = concat(config->output_template, "simple_huffman_stats.txt");

    config->num_vectors = load_num_vectors(config->pq_input_indices, config->m);
}

static void config_free(config_t* config) {
    if (config->pq_input_indices) {
        free(config->pq_input_indices);
        config->pq_input_indices = NULL;
    }
    // if (config->pq_input_centroids) {
    //     free(config->pq_input_centroids);
    //     config->pq_input_centroids = NULL;
    // }
    if (config->output_codebooks) {
        free(config->output_codebooks);
        config->output_codebooks = NULL;
    }
    if (config->output_encoded_indices) {
        free(config->output_encoded_indices);
        config->output_encoded_indices = NULL;
    }
    if (config->output_stats) {
        free(config->output_stats);
        config->output_stats = NULL;
    }
}

static void run(const config_t* config) {
    long long data_size = config->m * config->num_vectors;
    huffman_stats_t encode_stats;
    huffman_stats_init(&encode_stats, config->num_vectors, config->m, config->k_star);

    byte_t* data = malloc(sizeof(*data) * data_size);

    // TODO: Don't load this shit to memory
    FILE* indices_file = fopen(config->pq_input_indices, "rb");
    fread(data, sizeof(byte_t), config->num_vectors * config->m, indices_file);
    fclose(indices_file);

    double* stats = malloc(sizeof(*stats) * config->m * config->k_star);
    for (int i = 0; i < config->m * config->k_star; ++i) {
        stats[i] = 0.0;
    }

    for (byte_t* vec_it = data; vec_it != data + data_size; vec_it += config->m) {
        for (int i = 0; i < config->m; ++i) {
            stats[config->k_star * i + *(vec_it + i)] += 1;
        }
    }
    for (int i = 0; i < config->m; ++i) {
        double stats_sum = 0.0;
        for (double* stat_it = stats + config->k_star * i;
             stat_it != stats + config->k_star * (i + 1);
             ++stat_it)
        {
            stats_sum += *stat_it;
        }
        assert((long long) stats_sum == config->num_vectors);
    }

    huffman_codebook_t* codebooks = malloc(sizeof(*codebooks) * config->m);
    for (int i = 0; i < config->m; ++i) {
        double* stats_part = stats + config->k_star * i;
        huffman_codebook_encode_init(&codebooks[i], config->k_star, stats_part);
        double estimation = huffman_estimate_size(&codebooks[i], stats_part);
        huffman_stats_push(&encode_stats, i, estimation);
    }
    huffman_stats_print(&encode_stats);
    huffman_stats_print_filename(&encode_stats, config->output_stats);
    huffman_stats_destroy(&encode_stats);

    if (config->only_estimate) {
        // TODO: free resources
        return;
    }

    FILE* codebooks_file = fopen(config->output_codebooks, "wb");
    for (int i = 0; i < config->m; ++i) {
        huffman_codebook_save(codebooks + i, codebooks_file);
    }
    fclose(codebooks_file);

    FILE* encoded_indices_file = fopen(config->output_encoded_indices, "wb");
    bit_stream_t* stream = bit_stream_create_from_file(encoded_indices_file);
    for (byte_t* vector = data; vector != data + data_size; vector += config->m) {
        for (int i = 0; i < config->m; ++i) {
            const huffman_code_item_t* item = &codebooks[i].items[vector[i]];
            bit_stream_write(stream, item->code, item->bit_length);
        }
    }
    stream = bit_stream_destroy(stream);
    fclose(encoded_indices_file);

    for (int i = 0; i < config->m; ++i) {
        huffman_codebook_destroy(&codebooks[i]);
    }
    free(data);
}

int main(int argc, const char* argv[]) {
    config_t config;
    parse_args(&config, argc, argv);

    run(&config);

    config_free(&config);
    return 0;
}
