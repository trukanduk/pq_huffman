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

typedef struct _config {
    const char* pq_input_template;
    const char* nn_input_template;
    const char* output_template;
    int m;
    int k_star;

    char* pq_input_indices;
    char* nn_input_indices;
    char* output_stats;

    long long num_vectors;
    int num_nn;
} config_t;

static void print_help(const char* argv0) {
    fprintf(stderr, "Usage: %s <pq-input-template> <nn-input-template>"
            " <output-template> <m>\n", argv0);
    exit(1);
}

static void parse_args(config_t* config, int argc, const char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Too few positional arguments\n");
        print_help(argv[0]);
    }

    config->pq_input_template = argv[1];
    config->nn_input_template = argv[2];
    config->output_template = argv[3];
    config->m = atoi(argv[4]);


    config->k_star = (1 << 8); // TODO:

    config->pq_input_indices = concat(config->pq_input_template, "pq_indices.bvecsl");
    config->nn_input_indices = concat(config->nn_input_template, "nn_indices.ivecsl");
    config->output_stats = concat(config->output_template, "huffman_stats.txt");

    config->num_vectors = load_num_elements(config->pq_input_indices, config->m);
    config->num_nn = load_num_elements(config->nn_input_indices,
                                       config->num_vectors * sizeof(vector_id_t));
}

static void config_free(config_t* config) {
    if (config->pq_input_indices) {
        free(config->pq_input_indices);
        config->pq_input_indices = NULL;
    }
    if (config->nn_input_indices) {
        free(config->nn_input_indices);
        config->nn_input_indices = NULL;
    }
    if (config->output_stats) {
        free(config->output_stats);
        config->output_stats = NULL;
    }
}

static void collect_stats_at_kth_nn(const config_t* config, const byte_t* pq_indices,
                                    const vector_id_t* nn_indices, double* stats, int nn_index) {
    long long num_stats_per_codebook = config->k_star * config->k_star;
    long long num_stats_per_nn = config->m * num_stats_per_codebook;
    for (int i = 0; i < num_stats_per_nn; ++i) {
        stats[i] = 0.0;
    }

    const vector_id_t* nn_it;
    const byte_t* source_vector_it;
    for (nn_it = nn_indices, source_vector_it = pq_indices;
         source_vector_it != pq_indices + config->m * config->num_vectors;
         nn_it += config->num_nn, source_vector_it += config->m)
    {
        const byte_t* target_vector = pq_indices + config->m * nn_it[nn_index];
        for (int i = 0; i < config->m; ++i) {
            double* codebook_stats = stats + num_stats_per_codebook * i;
            codebook_stats[source_vector_it[i] * config->k_star + target_vector[i]] += 1;
        }
    }
    for (int i = 0; i < config->m; ++i) {
        double stats_sum = 0.0;
        for (double* stat_it = stats + num_stats_per_codebook * i;
             stat_it != stats + num_stats_per_codebook * (i + 1);
             ++stat_it)
        {
            stats_sum += *stat_it;
        }
        assert((long long) stats_sum == config->num_vectors);
    }
}

static void run(const config_t* config) {
    long long data_size = config->m * config->num_vectors;

    byte_t* pq_indices = load_vecs_light_filename(config->pq_input_indices, sizeof(byte_t),
                                                  config->num_vectors * config->m);
    vector_id_t* nn_indices = load_vecs_light_filename(config->nn_input_indices,
                                                       sizeof(vector_id_t),
                                                       config->num_vectors * config->num_nn);

    printf("assfgdsfgsd\n");

    FILE* stats_file = fopen(config->output_stats, "a");
    fprintf(stats_file, "[");

    double* stats = malloc(sizeof(*stats) * config->m * config->k_star * config->k_star);
    huffman_codebook_t* codebooks = malloc(sizeof(*codebooks) * config->m);
    for (int nn_index = 0; nn_index < config->num_nn; ++nn_index) {
        collect_stats_at_kth_nn(config, pq_indices, nn_indices, stats, nn_index);
        huffman_stats_t encode_stats;
        huffman_stats_init(&encode_stats, config->num_vectors, config->m, config->k_star);
        for (int part_index = 0; part_index < config->m; ++part_index) {
            double* stats_part = stats + config->k_star * config->k_star * part_index;
            huffman_codebook_t* codebook = codebooks + part_index;
            huffman_codebook_context_encode_init(codebook, config->k_star, stats_part);
            double estimation = huffman_estimate_size(codebook, stats_part);
            huffman_stats_push(&encode_stats, part_index, estimation);
        }
        huffman_stats_print(&encode_stats);
        huffman_stats_print_file(&encode_stats, stats_file);
        huffman_stats_destroy(&encode_stats);

        for (int part_index = 0; part_index < config->m; ++part_index) {
            huffman_codebook_destroy(codebooks + part_index);
        }

        fprintf(stats_file, ", ");
        printf("\n");
    }
    fprintf(stats_file, "]\n");
    fclose(stats_file);
    stats_file = NULL;

    free(stats);
    free(pq_indices);
    free(nn_indices);
}

int main(int argc, const char* argv[]) {
    srand(time(NULL));

    config_t config;
    parse_args(&config, argc, argv);

    run(&config);

    config_free(&config);
    return 0;
}
