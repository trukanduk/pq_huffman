#include "vecs_io.h"

#include <assert.h>
#include <stdlib.h>

typedef unsigned int header_item_t;

byte_t* load_vecs_light_filename(const char* filename, size_t element_size,
                                 long long* num_elements_out, int* num_dimensions_out) {
    FILE* f = fopen(filename, "rb");
    byte_t* result = load_vecs_light_file(f, element_size, num_elements_out, num_dimensions_out);
    fclose(f);
    return result;
}

byte_t* load_vecs_light_file(FILE* file, size_t element_size,
                             long long* num_elements_out, int* num_dimensions_out)
{
    long long num_elements;
    int num_dimensions;
    load_vecs_light_meta_file(file, &num_elements, &num_dimensions);
    if (num_elements_out) {
        *num_elements_out = num_elements;
    }
    if (num_dimensions_out) {
        *num_dimensions_out = num_dimensions;
    }

    byte_t* data = malloc(element_size * num_elements * num_dimensions);
    long long got_bytes = fread(data, element_size, num_elements * num_dimensions, file);
    assert(got_bytes == num_elements * num_dimensions);

    return data;
}

void load_vecs_light_meta_filename(const char* filename, long long* num_elements_out,
                                   int* num_dimensions_out) {
    FILE* f = fopen(filename, "rb");
    load_vecs_light_meta_file(f, num_elements_out, num_dimensions_out);
    fclose(f);
}

void load_vecs_light_meta_file(FILE* file, long long* num_elements_out,
                               int* num_dimensions_out) {
    header_item_t num_elements = 0; // NOTE: Should be replaced with long long in future
    fread(&num_elements, sizeof(num_elements), 1, file);
    if (num_elements_out) {
        *num_elements_out = num_elements;
    }

    header_item_t num_dimensions = 0;
    fread(&num_dimensions, sizeof(num_dimensions), 1, file);
    if (num_dimensions_out) {
        *num_dimensions_out = num_dimensions;
    }
}

long long load_vecs_num_vectors_filename(const char* filename) {
    long long result = 0;
    load_vecs_light_meta_filename(filename, &result, NULL);
    return result;
}

int load_vecs_num_dimensions_filename(const char* filename) {
    int result = 0;
    load_vecs_light_meta_filename(filename, NULL, &result);
    return result;
}

void save_vecs_light_meta_file(FILE* file, long long num_elements, int num_dimensions) {
    header_item_t num_elements_item = num_elements;
    fwrite(&num_elements_item, sizeof(num_elements_item), 1, file);

    header_item_t num_dimensions_item = num_dimensions;
    fwrite(&num_dimensions_item, sizeof(num_dimensions_item), 1, file);
}
