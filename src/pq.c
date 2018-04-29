#include "pq.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <yael/kmeans.h>

typedef struct _config {
    const char* input_filename;
    const char* output_template;
    int m;
    int num_bits_per_code;
    int num_threads;
    int use_opq;
    int compute_dist;

    long long num_vectors;
    int num_dimensions;
    int num_dimensions_per_part;
    int num_clusters;
    char* output_vectors;
    char* output_centroids;
    char* output_error;
} config_t;

enum {
    LOAD_BATCH_SIZE = 128 * 1024
};

static char* concat(const char* a, const char* b) {
    int alen = strlen(a);
    int blen = strlen(b);
    char* result = malloc(sizeof(char) * (alen + blen + 1));
    for (int i = 0; i < alen; ++i) {
        result[i] = a[i];
    }
    for (int i = 0; i < blen; ++i) {
        result[alen + i] = b[i];
    }
    result[alen + blen] = '\0';
    return result;
}

static long long minll(long long a, long long b) {
    return (a < b ? a : b);
}

static void load_input_file_meta(const char* input_filename, long long* num_vectors,
                                 int* num_dimensions) {
    FILE* f = fopen(input_filename, "rb");
    fread(num_dimensions, 1, sizeof(*num_dimensions), f);
    long long file_row_size = *num_dimensions * sizeof(float) + sizeof(int);
    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    *num_vectors = file_size / file_row_size;
    assert(file_size % file_row_size == 0);
    fclose(f);
}

static void load_vectors_dimensions(const char* input_filename, long long num_vectors,
                                    int from_dimension, int num_dimensions, float* output) {
    FILE* f = fopen(input_filename, "rb");
    int file_num_dimensions;
    fread(&file_num_dimensions, 1, sizeof(file_num_dimensions), f);
    fseek(f, 0, SEEK_SET);
    long long file_row_size = file_num_dimensions * sizeof(float) + sizeof(int);
    float* batch = malloc(file_row_size * LOAD_BATCH_SIZE);
    long long got_vectors = 0LL;
    while (got_vectors < num_vectors) {
        long long current_batch_size = minll(num_vectors - got_vectors, LOAD_BATCH_SIZE);
        long long red_rows = fread(batch, current_batch_size, file_row_size, f);
        for (long long row_index = 0; row_index < red_rows; ++row_index) {
            memcpy(output + (got_vectors + row_index) * num_dimensions,
                   batch + (file_num_dimensions + 1) * row_index,
                   num_dimensions);
        }
        got_vectors += red_rows;
    }
    free(batch);
    batch = NULL;
    fclose(f);
}

static void print_help(const char* argv0) {
    fprintf(stderr,
            "Usage: ./%s <input file> <output template> <m> "
            "[--num-threads <nt>] [--compute-dist]\n",
            argv0);
    exit(1);
}

static void parse_args(config_t* config, int argc, const char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Expected 3 positionsl arguments\n");
        print_help(argv[0]);
    }

    int show_help = 0;
    config->output_vectors = NULL;
    config->output_centroids = NULL;
    config->output_error = NULL;
    config->input_filename = argv[1];
    config->output_template = argv[2];
    config->m = atoi(argv[3]);
    config->compute_dist = 0;
    config->num_bits_per_code = 8; // TODO:
    config->num_threads = 1; // TODO:
    config->use_opq = 0; // TODO:
    if (config->m <= 0) {
        fprintf(stderr, "Invalid m: %d\n", config->m);
        print_help(argv[0]);
    }

    for (int arg_index = 4; arg_index < argc; ++arg_index) {
        if (!strcmp(argv[arg_index], "--num-threads")) {
            config->num_threads = atoi(argv[++arg_index]);
        } else if (!strcmp(argv[arg_index], "--compute-dist")) {
            config->compute_dist = 1;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[arg_index]);
            show_help = 1;
        }
    }
    if (show_help) {
        print_help(argv[0]);
    }

    load_input_file_meta(config->input_filename, &config->num_vectors, &config->num_dimensions);
    config->num_dimensions_per_part = config->num_dimensions / config->m;
    assert(config->num_dimensions % config->m == 0);
    config->num_clusters = (1 << config->num_bits_per_code);
    config->output_vectors = concat(config->output_template, "pq_indices.bvecsl");
    config->output_centroids = concat(config->output_template, "pq_centroids.fvecsl");
    config->output_error = concat(config->output_template, "pq_error");
}

static void config_free(config_t* config) {
    if (config->output_vectors) {
        free(config->output_vectors);
        config->output_vectors = NULL;
    }
    if (config->output_centroids) {
        free(config->output_centroids);
        config->output_centroids = NULL;
    }
    if (config->output_error) {
        free(config->output_error);
        config->output_error = NULL;
    }
}

static void plus(float* dists_sum, const float* dists, long long num_vectors) {
    if (!dists_sum) {
        return;
    }

    float* sum_it = dists_sum;
    const float* dist_it = dists;
    for (; sum_it != dists_sum + num_vectors; ++sum_it, ++dist_it) {
        *sum_it += *dist_it;
    }
}

static void copy_cluster_indices(unsigned char* result, const int* clusters,
                                 long long num_vectors, int m) {
    unsigned char* res_it = result;
    const int* clusters_it = clusters;
    for (; clusters_it != clusters + num_vectors; res_it += m, ++clusters_it) {
        *res_it = *clusters_it;
    }
}

static void save_indices(const char* output_filename, const unsigned char* result,
                         long long num_vectors, int m) {
    FILE* f = fopen(output_filename, "wb");
    fwrite(result, m * sizeof(*result), num_vectors, f);
    fclose(f);
}

static void save_error(const char* output_filename, const float* dists_sum, long long num_vectors) {
    double sum = 0.0;
    for (const float* dist_it = dists_sum; dist_it != dists_sum + num_vectors; ++dist_it) {
        sum += sqrt(*dist_it);
    }

    FILE* f = fopen(output_filename, "a");
    fprintf(f, "%lf\n", sum);
    fclose(f);
}

static void run(const config_t* config) {
    float* data = malloc(sizeof(*data) * config->num_vectors * config->num_dimensions_per_part);
    unsigned char* result = malloc(sizeof(*result) * config->num_vectors * config->m);
    int* clusters = malloc(sizeof(*clusters) * config->num_vectors);
    float* centroids =
            malloc(sizeof(*centroids) * config->num_clusters * config->num_dimensions_per_part);
    float* dists = NULL;
    float* dists_sum = NULL;
    if (config->compute_dist) {
        dists = malloc(sizeof(*dists) * config->num_vectors);
        dists_sum = malloc(sizeof(*dists_sum) * config->num_vectors);
    }
    FILE* centroids_file = fopen(config->output_centroids, "wb");

    for (int part_index = 0; part_index < config->m; ++part_index) {
        printf("Starting part %d\n", part_index);
        load_vectors_dimensions(config->input_filename, config->num_vectors,
                                config->num_dimensions_per_part * part_index,
                                config->num_dimensions_per_part, data);
        kmeans(config->num_dimensions_per_part, config->num_vectors, config->num_clusters,
               -1, data, config->num_threads | KMEANS_INIT_BERKELEY, time(NULL), 1, centroids,
               dists, clusters, NULL);
        copy_cluster_indices(result + part_index, clusters, config->num_vectors, config->m);
        fwrite(centroids, sizeof(*centroids),
               config->num_clusters * config->num_dimensions_per_part, centroids_file);
        if (config->compute_dist) {
            plus(dists_sum, dists, config->num_vectors);
        }
    }
    fclose(centroids_file);
    centroids_file = NULL;

    save_indices(config->output_vectors, result, config->num_vectors, config->m);

    if (config->compute_dist) {
        save_error(config->output_error, dists_sum, config->num_vectors);

        free(dists);
        dists = NULL;
        free(dists_sum);
        dists_sum = NULL;
    }
    free(centroids);
    centroids = NULL;
    free(clusters);
    clusters = NULL;
    free(result);
    result = NULL;
    free(data);
    data = NULL;
}

int main(int argc, const char* argv[]) {
    config_t config;
    parse_args(&config, argc, argv);

    run(&config);

    config_free(&config);
    return 0;
}
