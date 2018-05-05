#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bitstream.h"
#include "huffman.h"
#include "stats.h"
#include "misc.h"
#include "vecs_io.h"

enum {
    SORT_NOSORT = 0,
    SORT_SORT = 1,
    SORT_SHUFFLE = 2
};

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
    config->sort = SORT_SORT;
    config->context = 1;

    for (const char** arg = argv + 4; *arg; ++arg) {
        if (!strcmp(*arg, "--only-estimate")) {
            config->only_estimate = 1;
        } else if (!strcmp(*arg, "--no-sort")) {
            config->sort = SORT_NOSORT;
        } else if (!strcmp(*arg, "--shuffle")) {
            config->sort = SORT_SHUFFLE;
        } else if (!strcmp(*arg, "--no-context")) {
            config->context = 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", *arg);
            print_help(argv[0]);
        }
    }

    config->k_star = (1 << 8); // TODO:

    config->pq_input_indices = concat(config->pq_input_template, "pq_indices.bvecsl");
    // config->pq_input_centroids = concat(config->pq_input_template, "pq_centroids.fvecsl");
    config->output_codebooks = concat(config->output_template, "huffman_codebooks.bin");
    config->output_encoded_indices = concat(config->output_template, "huffman_indices.bin");
    config->output_stats = concat(config->output_template, "huffman_stats.txt");

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

static double* collect_non_context_stats(const config_t* config, const byte_t* data) {
    double* stats = malloc(sizeof(*stats) * config->m * config->k_star);
    for (int i = 0; i < config->m * config->k_star; ++i) {
        stats[i] = 0.0;
    }

    for (const byte_t* vec_it = data;
         vec_it != data + config->m * config->num_vectors;
         vec_it += config->m)
    {
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
    return stats;
}

static double* collect_context_stats(const config_t* config, const byte_t* data) {
    int alphabet_size = config->k_star * config->k_star;
    double* stats = malloc(sizeof(*stats) * config->m * alphabet_size);
    for (int i = 0; i < config->m * alphabet_size; ++i) {
        stats[i] = 0.0;
    }

    const byte_t* current_vec = data + config->m;
    const byte_t* prev_vec = data;
    for (;
         current_vec != data + config->m * config->num_vectors;
         current_vec += config->m, prev_vec += config->m)
    {
        for (int i = 0; i < config->m; ++i) {
            stats[i * alphabet_size + (prev_vec[i] << BYTE_NUM_BITS) + current_vec[i]] += 1;
        }
    }
    for (int i = 0; i < config->m; ++i) {
        double stats_sum = 0.0;
        for (double* stat_it = stats + alphabet_size * i;
             stat_it != stats + alphabet_size * (i + 1);
             ++stat_it)
        {
            stats_sum += *stat_it;
        }
        assert(((long long) stats_sum + 1) == config->num_vectors);
    }

    return stats;
}

static void encode_non_context_data(const config_t* config, const byte_t* data,
                                    const huffman_codebook_t* codebooks, bit_stream_t* stream) {
    for (const byte_t* vector = data;
         vector != data + config->m * config->num_vectors;
         vector += config->m)
    {
        for (int i = 0; i < config->m; ++i) {
            const huffman_code_item_t* item = &codebooks[i].items[vector[i]];
            bit_stream_write(stream, item->code, item->bit_length);
        }
    }
}

static void encode_context_data(const config_t* config, const byte_t* data,
                                const huffman_codebook_t* codebooks, bit_stream_t* stream) {
    const byte_t* vec_it = data;
    const byte_t* prev_vec_it = NULL;
    for (;
         vec_it != data + config->m * config->num_vectors;
         prev_vec_it = vec_it, vec_it += config->m)
    {
        for (int i = 0; i < config->m; ++i) {
            if (prev_vec_it) {
                int code_index = (prev_vec_it[i] << 8) + vec_it[i];
                const huffman_code_item_t* item = &codebooks[i].items[code_index];
                bit_stream_write(stream, item->code, item->bit_length);
            } else {
                bit_stream_write(stream, vec_it + i, BYTE_NUM_BITS);
            }
        }
    }
}

static void shuffle(const config_t* config, byte_t* data) {
    byte_t* tmp = malloc(sizeof(byte_t) * config->m);
    long long vec_size = config->m * sizeof(byte_t);
    for (long long i = config->num_vectors - 1; i >= 1; --i) {
        long long j = 1LL * rand() * i / RAND_MAX;

        memmove(tmp, data + i * config->m, vec_size);
        memmove(data + i * config->m, data + j * config->m, vec_size);
        memmove(data + j * config->m, tmp, vec_size);
    }
    free(tmp);
}

static int sort_indices_comparator_m;
static int sort_indices_comparator(const void* left, const void* right) {
    return strncmp(left, right, sort_indices_comparator_m);
}

static void run(const config_t* config) {
    long long data_size = config->m * config->num_vectors;
    huffman_stats_t encode_stats;
    huffman_stats_init(&encode_stats, config->num_vectors, config->m, config->k_star);

    byte_t* data = load_vecs_light_filename(config->pq_input_indices, sizeof(byte_t), config->num_vectors * config->m);
    // byte_t* data = malloc(sizeof(*data) * data_size);
    // // TODO: Don't load this shit to memory
    // FILE* indices_file = fopen(config->pq_input_indices, "rb");
    // fread(data, sizeof(byte_t), config->num_vectors * config->m, indices_file);
    // fclose(indices_file);

    if (config->sort == SORT_SHUFFLE) {
        shuffle(config, data);
    } else if (config->sort == SORT_SORT) {
        sort_indices_comparator_m = config->m;
        qsort(data, config->num_vectors, config->m * sizeof(byte_t), sort_indices_comparator);
    }

    double* stats;
    int alphabet_size = config->k_star;
    if (config->context) {
        stats = collect_context_stats(config, data);
        alphabet_size *= config->k_star;
    } else {
        stats = collect_non_context_stats(config, data);
    }

    huffman_codebook_t* codebooks = malloc(sizeof(*codebooks) * config->m);
    for (int i = 0; i < config->m; ++i) {
        double* stats_part = stats + alphabet_size * i;
        if (config->context) {
            huffman_codebook_context_encode_init(codebooks + i, config->k_star, stats_part);
        } else {
            huffman_codebook_encode_init(codebooks + i, config->k_star, stats_part);
        }

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
    unsigned int m_int = config->m;
    fwrite(&m_int, sizeof(m_int), 1, codebooks_file);
    for (int i = 0; i < config->m; ++i) {
        huffman_codebook_save(codebooks + i, codebooks_file);
    }
    fclose(codebooks_file);

    FILE* encoded_indices_file = fopen(config->output_encoded_indices, "wb");
    unsigned long long num_vectors_ll = config->num_vectors;
    fwrite(&num_vectors_ll, sizeof(num_vectors_ll), 1, encoded_indices_file);
    bit_stream_t* stream = bit_stream_create_from_file(encoded_indices_file);
    if (config->context) {
        encode_context_data(config, data, codebooks, stream);
    } else {
        encode_non_context_data(config, data, codebooks, stream);
    }
    stream = bit_stream_destroy(stream);
    fclose(encoded_indices_file);

    for (int i = 0; i < config->m; ++i) {
        huffman_codebook_destroy(&codebooks[i]);
    }
    free(stats);
    free(data);
}

int main(int argc, const char* argv[]) {
    srand(time(NULL));

    config_t config;
    parse_args(&config, argc, argv);

    run(&config);

    config_free(&config);
    return 0;
}
