#include "vecs_io.h"

#include <assert.h>
#include <stdlib.h>

typedef unsigned int header_item_t;

byte_t* load_vecs_light_filename(const char* filename, size_t element_size,
                                 long long num_elements) {
    FILE* f = fopen(filename, "rb");
    byte_t* result = load_vecs_light_file(f, element_size, num_elements);
    fclose(f);

    return result;
}

byte_t* load_vecs_light_file(FILE* file, size_t element_size, long long num_elements) {
    byte_t* data = malloc(element_size * num_elements);
    long long got_bytes = fread(data, element_size, num_elements, file);
    assert(got_bytes == num_elements && "Cannot read xvecsl-file");
    return data;
}
