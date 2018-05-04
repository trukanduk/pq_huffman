#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bitstream.h"
#include "huffman.h"

enum {
    ACTION_DECODE = 0, // NOTE: Read input file and decode the data.
                       //       Drop data if no output expected
    ACTION_CHECK = 1, // NOTE: Fully read to memory, decode, sort result,
                      //       then read source indices, sort them and compare
                      //       byte-by-byte. Needed for tests

    DEFAULT_BATCH_SIZE = 1000 * 1000 // NOTE: 1M vectors
};

typedef struct _config {
    const char* input_huffman_template;
    const char* output_filename; // NOTE: NULL if no output required
    const char* pq_indices_filename; // NOTE: Nullable - has sense only
                                     //       for action=ACTION_CHECK
    int action;
    int m;
    long long num_vectors;

    char* input_huffman_indices_filename;
    char* input_huffman_codebooks_filename;
} config_t;

// TODO: Move to common file!
static char* concat(const char* left, const char* right) {
    int left_length = strlen(left);
    int right_length = strlen(right);
    char* result = malloc(sizeof(*result) * (left_length + right_length + 1));
    char* res_it = result;
    for (const char* c = left; *c; ++c, ++res_it) {
        *res_it = *c;
    }
    for (const char* c = right; *c; ++c, ++res_it) {
        *res_it = *c;
    }
    *res_it = '\0';
    return result;
}

static void print_help(const char* argv0) {
    fprintf(stderr, "Usage: %s <huffman output template> [--output-file <output file>]"
            " [--check-file <pq indices file>]\n");
    exit(1);
}

static int load_m_from_codebooks_file(const char* input_huffman_codebooks_filename) {
    FILE* f = fopen(input_huffman_codebooks_filename, "rb");
    unsigned int m = 0;
    fread(&m, sizeof(m), 1, f);
    fclose(f);
    return m;
}

static long long load_num_vectors_from_indices_file(const char* input_huffman_indices_filename) {
    FILE* f = fopen(input_huffman_indices_filename, "rb");
    unsigned long long num_vectors = 0;
    fread(&num_vectors, sizeof(num_vectors), 1, f);
    fclose(f);
    return num_vectors;
}

static void parse_args(config_t* config, int argc, const char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Too few arguments\n");
        print_help(argv[0]);
    }
    config->input_huffman_template = argv[1];

    config->num_vectors = 0;
    config->action = ACTION_DECODE;
    config->output_filename = NULL;
    config->pq_indices_filename = NULL;

    config->input_huffman_indices_filename = NULL;
    config->input_huffman_codebooks_filename = NULL;

    int show_help = 0;
    for (const char** arg = argv + 2; *arg; ++arg) {
        if (!strcmp(*arg, "--output-file")) {
            ++arg;
            config->output_filename = *arg;
        } else if (!strcmp(*arg, "--check-file")) {
            ++arg;
            config->action = ACTION_CHECK;
            config->pq_indices_filename = *arg;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", *arg);
            show_help = 1;
        }
    }

    if (show_help) {
        print_help(argv[0]);
    }

    config->input_huffman_indices_filename = concat(config->input_huffman_template, "huffman_indices.bin");
    config->input_huffman_codebooks_filename = concat(config->input_huffman_template, "huffman_codebooks.bin");

    config->num_vectors = load_num_vectors_from_indices_file(config->input_huffman_indices_filename);
    config->m = load_m_from_codebooks_file(config->input_huffman_codebooks_filename);
}

static void config_free(config_t* config) {
    if (config->input_huffman_indices_filename) {
        free(config->input_huffman_indices_filename);
        config->input_huffman_indices_filename = NULL;
    }

    if (config->input_huffman_codebooks_filename) {
        free(config->input_huffman_codebooks_filename);
        config->input_huffman_codebooks_filename = NULL;
    }
}

typedef struct _codebooks_array {
    huffman_codebook_t* codebooks;
    huffman_decoder_t** decoders;
    int m;
} codebooks_array_t;

static void codebooks_array_init(codebooks_array_t* codebooks_array, const config_t* config) {
    codebooks_array->m = config->m;
    codebooks_array->codebooks = malloc(sizeof(*codebooks_array->codebooks) * codebooks_array->m);
    codebooks_array->decoders = malloc(sizeof(*codebooks_array->decoders) * codebooks_array->m);
    FILE* codebooks_file = fopen(config->input_huffman_codebooks_filename, "rb");

    fseek(codebooks_file, sizeof(int), SEEK_CUR); // NOTE: Skip m

    for (int part_index = 0; part_index < codebooks_array->m; ++part_index) {
        huffman_codebook_load(codebooks_array->codebooks + part_index, codebooks_file);
        codebooks_array->decoders[part_index] =
                huffman_decoder_create(codebooks_array->codebooks + part_index);
    }
    fclose(codebooks_file);
}

static void codebooks_array_destroy(codebooks_array_t* codebooks_array) {
    for (int part_index = 0; part_index < codebooks_array->m; ++part_index) {
        codebooks_array->decoders[part_index] =
                huffman_decoder_destroy(codebooks_array->decoders[part_index]);
        huffman_codebook_destroy(codebooks_array->codebooks + part_index);
    }

    free(codebooks_array->decoders);
    codebooks_array->decoders = NULL;

    free(codebooks_array->codebooks);
    codebooks_array->codebooks = NULL;

    codebooks_array->m = 0;
}

static long long minll(long long left, long long right) {
    return (left < right ? left : right);
}

static int sort_indices_comparator_m;
static int sort_indices_comparator(const void* left, const void* right) {
    return strncmp(left, right, sort_indices_comparator_m);
}

static void run(const config_t* config) {
    codebooks_array_t codebooks_array;
    codebooks_array_init(&codebooks_array, config);

    printf("Codebooks loaded\n");

    FILE* output_file = NULL;
    if (config->output_filename) {
        output_file = fopen(config->output_filename, "wb");
    }

    long long batch_size = DEFAULT_BATCH_SIZE;
    if (config->action == ACTION_CHECK) {
        batch_size = config->num_vectors; // NOTE: Need to store all in memory :(
    }
    byte_t* decoded_batch = calloc(sizeof(byte_t) * config->m, batch_size);

    FILE* encoded_file = fopen(config->input_huffman_indices_filename, "rb");
    fseek(encoded_file, sizeof(long long), SEEK_CUR); // NOTE: Skip num vectors
    bit_stream_t* encoded_stream = bit_stream_create_from_file(encoded_file);

    long long got_vectors = 0;
    while (got_vectors < config->num_vectors) {
        // NOTE: Fill single batch on iteration.
        long long current_batch_size = minll(config->num_vectors - got_vectors, batch_size);

        byte_t* batch_it = decoded_batch;
        for (long long vec_index = 0; vec_index < current_batch_size; ++vec_index) {
            for (int part_index = 0; part_index < config->m; ++part_index, ++batch_it) {

                int symbol = HUFFMAN_NO_SYMBOL;
                while (symbol == HUFFMAN_NO_SYMBOL) {
                    int bit_value = bit_stream_read_bit(encoded_stream);
                    symbol = huffman_decoder_push_bit(codebooks_array.decoders[part_index],
                                                      bit_value);
                }

                assert(symbol != HUFFMAN_INVALID_SYMBOL);
                *batch_it = symbol;
            }
        }
        assert(batch_it == decoded_batch + config->m * current_batch_size);

        if (output_file) {
            fwrite(decoded_batch, sizeof(*decoded_batch) * config->m, current_batch_size,
                   output_file);
        }
        got_vectors += current_batch_size;
    }

    printf("Decoded\n");

    if (config->action == ACTION_CHECK) {
        assert(config->pq_indices_filename != NULL);

        long long indices_size = config->m * config->num_vectors;
        byte_t* orig_indices = malloc(sizeof(*orig_indices) * indices_size);

        FILE* indices_file = fopen(config->pq_indices_filename, "rb");
        fread(orig_indices, sizeof(byte_t), config->num_vectors * config->m, indices_file);
        fclose(indices_file);

        sort_indices_comparator_m = config->m;
        qsort(orig_indices, config->num_vectors, config->m * sizeof(byte_t),
              sort_indices_comparator);
        printf("Orig sorted\n");

        qsort(decoded_batch, config->num_vectors, config->m * sizeof(byte_t),
              sort_indices_comparator);
        printf("Decoded sorted\n");

        assert(!strncmp(orig_indices, decoded_batch, sizeof(byte_t) * indices_size));
        printf("Check OK!\n");

        free(orig_indices);
        orig_indices = NULL;
    }

    encoded_stream = bit_stream_destroy(encoded_stream);
    fclose(encoded_file);
    encoded_file = NULL;

    free(decoded_batch);
    decoded_batch = NULL;

    if (output_file) {
        fclose(output_file);
        output_file = NULL;
    }

    codebooks_array_destroy(&codebooks_array);
}

int main(int argc, const char* argv[]) {
    config_t config;
    parse_args(&config, argc, argv);

    run(&config);

    config_free(&config);
    return 0;
}
