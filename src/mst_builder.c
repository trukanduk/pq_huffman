#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mst.h"
#include "misc.h"
#include "vecs_io.h"

typedef struct _config {
    const char* nn_input_template;
    const char* pq_input_template; // NOTE: May be null - if no pq penalties expected
    const char* output_template;
    int num_nn_to_load;
    float pq_penalty; // NOTE: Not used by now

    char* nn_indices_filename;
    char* nn_dist_filename;
    char* pq_indices_filename; // NOTE: May be null
    char* output_tree_filename;
    char* output_stats_filename;
    char* output_stats_num_children_filename;

    long long num_vectors;
    int pq_m; // NOTE: Used only iff pq_input_template is given
} config_t;

static void print_help(const char* argv0) {
    fprintf(stderr, "Usage: %s <nn-input-template> <output-template> <num-nn-to-use>"
            " [--pq-template <pq output template>]"
            " [--pq-penalty <pq penalty size>]\n", argv0);
    exit(1);
}

static void parse_args(config_t* config, int argc, const char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Too few arguments\n");
        print_help(argv[0]);
    }

    config->nn_input_template = argv[1];
    config->output_template = argv[2];
    config->num_nn_to_load = atoi(argv[3]);

    config->pq_input_template = NULL;
    config->pq_penalty = 0.0;

    for (const char** arg = argv + 4; *arg; ++arg) {
        if (!strcmp(*arg, "--pq-template")) {
            ++arg;
            config->pq_input_template = *arg;
        } else if (!strcmp(*arg, "--pq-penalty")) {
            ++arg;
            config->pq_penalty = atof(*arg);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", *arg);
            print_help(argv[0]);
        }
    }

    config->nn_indices_filename = concat(config->nn_input_template, "nn_indices.ivecsl");
    config->nn_dist_filename = concat(config->nn_input_template, "nn_dist.fvecsl");
    if (config->pq_input_template) {
        config->pq_indices_filename = concat(config->pq_input_template, "pq_indices.bvecsl");
        config->pq_m = load_vecs_num_dimensions_filename(config->pq_indices_filename);
    }
    config->output_tree_filename = concat(config->output_template, "mst.tree");
    config->output_stats_filename = concat(config->output_template, "stats.json");
    config->output_stats_num_children_filename = concat(config->output_template,
                                                        "stats_num_children.json");

    config->num_vectors = load_vecs_num_vectors_filename(config->nn_indices_filename);
}

static void config_destroy(config_t* config) {
    config->num_vectors = 0;
    config->pq_m = 0;

    free(config->nn_indices_filename);
    config->nn_indices_filename = NULL;

    free(config->nn_dist_filename);
    config->nn_dist_filename = NULL;

    free(config->pq_indices_filename);
    config->pq_indices_filename = NULL;

    free(config->output_tree_filename);
    config->output_tree_filename = NULL;

    free(config->output_stats_filename);
    config->output_stats_filename = NULL;

    free(config->output_stats_num_children_filename);
    config->output_stats_num_children_filename = NULL;
}

static void run(const config_t* config) {
    byte_t* pq_indices = NULL;
    if (config->pq_indices_filename) {
        pq_indices = load_vecs_light_filename(config->pq_indices_filename, sizeof(byte_t),
                                              NULL, NULL);
    }
    mst_edge_t* edges = load_mst_edges_from_nn_filenames(config->num_vectors,
                                                         config->num_nn_to_load, config->pq_m,
                                                         config->pq_penalty, pq_indices,
                                                         config->nn_indices_filename,
                                                         config->nn_dist_filename);
    tree_t tree;
    minimum_spanning_tree(&tree, config->num_vectors, config->num_nn_to_load, edges);

    free(edges);
    edges = NULL;

    tree_save_filename(&tree, config->output_tree_filename);
    huffman_stats_t indices_stats;
    huffman_stats_t children_stats;
    tree_estimate_huffman_encoding(&indices_stats, &children_stats, &tree, config->pq_m,
                                   pq_indices);
    huffman_stats_print(&indices_stats);
    huffman_stats_print_filename(&indices_stats, config->output_stats_filename);
    huffman_stats_print_filename(&children_stats, config->output_stats_num_children_filename);

    huffman_stats_destroy(&indices_stats);
    huffman_stats_destroy(&children_stats);

    if (pq_indices) {
        free(pq_indices);
        pq_indices = NULL;
    }
}

int main(int argc, const char* argv[]) {
    config_t config;
    parse_args(&config, argc, argv);

    run(&config);

    config_destroy(&config);
    return 0;
}
