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
#include "mst.h"

enum {
    SORT_NOSORT = 0,
    SORT_SORT = 1,
    SORT_SHUFFLE = 2
};

typedef struct _config {
    const char* pq_input_template;
    const char* tree_input;
    const char* output_template;
    int m;
    int k_star;
    int only_estimate;
    int sort;
    int context;

    char* pq_input_indices;
    // char* pq_input_centroids;
    char* output_codebooks;
    char* output_children_codebook;
    char* output_encoded_indices;
    char* output_encoded_children;
    char* output_stats;
    char* output_children_stats;

    long long num_vectors;
} config_t;

static void print_help(const char* argv0) {
    fprintf(stderr, "Usage: %s <pq-output-template> <output-template> <m>"
                    " [--no-sort] [--no-context] [--only-estimate] [--tree <tree path>]\n", argv0);
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
    config->tree_input = NULL;
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
        } else if (!strcmp(*arg, "--tree")) {
            ++arg;
            config->tree_input = *arg;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", *arg);
            print_help(argv[0]);
        }
    }

    config->k_star = (1 << 8); // TODO:

    config->pq_input_indices = concat(config->pq_input_template, "pq_indices.bvecsl");
    // config->pq_input_centroids = concat(config->pq_input_template, "pq_centroids.fvecsl");
    config->output_codebooks = concat(config->output_template, "huffman_codebooks.bin");
    config->output_children_codebook = concat(config->output_template,
                                               "huffman_children_codebooks.bin");
    config->output_encoded_indices = concat(config->output_template, "huffman_indices.bin");
    config->output_encoded_children = concat(config->output_template, "huffman_children.bin");
    config->output_stats = concat(config->output_template, "huffman_stats.txt");
    config->output_children_stats = concat(config->output_template, "huffman_children_stats.txt");

    config->num_vectors = load_vecs_num_vectors_filename(config->pq_input_indices);
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
    if (config->output_children_codebook) {
        free(config->output_children_codebook);
        config->output_children_codebook = NULL;
    }
    if (config->output_encoded_indices) {
        free(config->output_encoded_indices);
        config->output_encoded_indices = NULL;
    }
    if (config->output_encoded_children) {
        free(config->output_encoded_children);
        config->output_encoded_children = NULL;
    }
    if (config->output_stats) {
        free(config->output_stats);
        config->output_stats = NULL;
    }
    if (config->output_children_stats) {
        free(config->output_children_stats);
        config->output_children_stats = NULL;
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

static double* collect_context_stats(const config_t* config, const byte_t* data,
                                     const vector_id_t* vertices, const int* num_children,
                                     int* num_roots_out) {
    int alphabet_size = config->k_star * config->k_star;
    double* stats = malloc(sizeof(*stats) * config->m * alphabet_size);
    for (int i = 0; i < config->m * alphabet_size; ++i) {
        stats[i] = 0.0;
    }

    int num_roots = 1;
    if (vertices) {
        num_roots = tree_collect_indices_stats(config->num_vectors, config->m, data, vertices,
                                               num_children, stats);
    } else {
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
    }
    if (num_roots_out) {
        *num_roots_out = num_roots;
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

static void encode_tree_data(const config_t* config, const byte_t* data,
                             const huffman_codebook_t* codebooks, bit_stream_t* stream,
                             const vector_id_t* vertices, const int* num_children) {
    tree_traverser_t traverser;
    tree_traverser_init(&traverser, config->num_vectors);

    const int* num_children_it = num_children;
    const vector_id_t* vertices_it = vertices;
    long long total_written = 0;
    for (long long vec_index = 0; vec_index < config->num_vectors; ++vec_index) {
        vector_id_t current_vector_id = vertices_it ? *vertices_it : vec_index;
        const byte_t* current_vector = data + 1LL * current_vector_id * config->m;

        vector_id_t prev_vector_id = tree_traverser_get_active_parent(&traverser);
        const byte_t* prev_vector = prev_vector_id != TRAVERSER_NO_PARENT_VECTOR
                ? data + 1LL * prev_vector_id * config->m
                : NULL;
        for (int part_index = 0; part_index < config->m; ++part_index) {
            const huffman_codebook_t* codebook = codebooks + part_index;
            if (prev_vector) {
                long long item_index = config->k_star * prev_vector[part_index]
                        + current_vector[part_index];
                const huffman_code_item_t* item = &codebook->items[item_index];
                bit_stream_write(stream, item->code, item->bit_length);
                assert(item->bit_length > 0);
                total_written += item->bit_length;
            } else {
                bit_stream_write(stream, current_vector + part_index, BYTE_NUM_BITS);
                total_written += BYTE_NUM_BITS;
            }
        }

        int current_num_children = num_children_it ? *num_children_it : 1;
        tree_traverser_push_vertex(&traverser, current_vector_id, current_num_children);

        if (num_children) {
            ++num_children_it;
        }
        if (vertices) {
            ++vertices_it;
        }
    }
    assert(traverser.stack_size == 0);

    // printf("WRITTEN %lld\n", total_written);
    tree_traverser_destroy(&traverser);
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

    byte_t* data = load_vecs_light_filename(config->pq_input_indices, sizeof(byte_t), NULL, NULL);

    if (config->sort == SORT_SHUFFLE) {
        shuffle(config, data);
    } else if (config->sort == SORT_SORT) {
        sort_indices_comparator_m = config->m;
        qsort(data, config->num_vectors, config->m * sizeof(byte_t), sort_indices_comparator);
    }

    double* stats;
    int alphabet_size = config->k_star;
    vector_id_t* vertices = NULL;
    int* num_children = NULL;
    if (config->tree_input) {
        tree_t tree;
        tree_load_filename(&tree, config->tree_input);
        vertices = malloc(sizeof(*vertices) * config->num_vectors);
        num_children = malloc(sizeof(*num_children) * config->num_vectors);
        tree_collect_vertices_dfs(&tree, vertices, num_children);
        tree_destroy(&tree);
    }
    int num_roots = 0;
    if (config->context) {
        stats = collect_context_stats(config, data, vertices, num_children, &num_roots);
        alphabet_size *= config->k_star;
        encode_stats.num_roots = num_roots;

        // huffman_counts_context_dump(stats, config->k_star, stdout);
    } else {
        stats = collect_non_context_stats(config, data);
    }

    if (num_children) {
        int children_alphabet_size = 0;
        double* symbol_counts = tree_collect_num_children_stats(config->num_vectors, num_children,
                                                                &children_alphabet_size);
        huffman_codebook_t codebook;
        huffman_codebook_encode_init(&codebook, children_alphabet_size, symbol_counts);

        FILE* codebook_file = fopen(config->output_children_codebook, "wb");
        huffman_codebook_save(&codebook, codebook_file);
        fclose(codebook_file);

        FILE* children_file = fopen(config->output_encoded_children, "wb");
        bit_stream_t* bitstream = bit_stream_create_from_file(children_file);
        for (const int* children_it = num_children;
             children_it != num_children + config->num_vectors;
             ++children_it)
        {
            const huffman_code_item_t* item = &codebook.items[*children_it];
            bit_stream_write(bitstream, item->code, item->bit_length);
        }
        bit_stream_destroy(bitstream);
        fclose(children_file);

        huffman_stats_t stats;
        huffman_stats_init(&stats, config->num_vectors, 1, config->k_star);
        double estimation = huffman_estimate_size(&codebook, symbol_counts);
        huffman_stats_push(&stats, 0, estimation);
        huffman_stats_print_filename(&stats, config->output_children_stats);
        huffman_stats_destroy(&stats);

        huffman_codebook_destroy(&codebook);
        free(symbol_counts);
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
    long long sum_length = 0;
    for (int i = 0; i < config->m; ++i) {
        for (int j = 0; j < 256 * 256; ++j) {
            sum_length += codebooks[i].items[j].bit_length;
        }
        huffman_codebook_save(codebooks + i, codebooks_file);
    }
    printf("SUM LENGTH %lld, MEAN %lf\n", sum_length, 1.0 * sum_length / 256 / 256 / config->m);
    fclose(codebooks_file);

    // huffman_codebook_dump(codebooks, stdout);

    FILE* encoded_indices_file = fopen(config->output_encoded_indices, "wb");
    unsigned long long num_vectors_ll = config->num_vectors;
    fwrite(&num_vectors_ll, sizeof(num_vectors_ll), 1, encoded_indices_file);
    bit_stream_t* stream = bit_stream_create_from_file(encoded_indices_file);
    if (config->tree_input) {
        printf("use tree encoder\n");
        encode_tree_data(config, data, codebooks, stream, vertices, num_children);
    } else if (config->context) {
        printf("use context encoder\n");
        encode_context_data(config, data, codebooks, stream);
    } else {
        printf("use non-context encoder\n");
        encode_non_context_data(config, data, codebooks, stream);
    }
    stream = bit_stream_destroy(stream);
    fclose(encoded_indices_file);

    for (int i = 0; i < config->m; ++i) {
        huffman_codebook_destroy(&codebooks[i]);
    }
    if (config->tree_input) {
        free(vertices);
        free(num_children);
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
