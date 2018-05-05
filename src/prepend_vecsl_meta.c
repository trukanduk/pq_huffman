#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vecs_io.h"

enum {
    XVECSL_SUFFIX_LENGTH = 7 // .xvecsl
};

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <filename> <num_vectors> <num_dimensions>\n", argv[0]);
        return 1;
    }
    const char* filename = argv[1];
    int num_vectors = atoi(argv[2]);
    int num_dimensions = atoi(argv[3]);

    int filename_length = strlen(filename);
    const char* extension = filename + filename_length - XVECSL_SUFFIX_LENGTH;
    assert(extension[0] == '.');
    assert(!strcmp(extension + 2, "vecsl"));
    char vecs_type = extension[1];
    long long element_size = 0;
    if (vecs_type == 'i' || vecs_type == 'f') {
        element_size = 4;
    } else if (vecs_type == 'l') {
        element_size = 8;
    } else if (vecs_type == 'b') {
        element_size = 1;
    } else {
        fprintf(stderr, "Unknown extension type: '%s'\n", extension);
        return 1;
    }

    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "File %s not found\n", filename);
        return 1;
    }
    long long num_elements = num_vectors * num_dimensions;
    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    if (file_size == num_elements * element_size + 8) {
        printf("File %s already processed\n", filename);
        return 0;
    } else if (file_size != num_elements * element_size) {
        fprintf(stderr, "Invalid size/dimension of file %s. File size if %lld (expected %lld)\n",
                filename, file_size, num_elements * element_size);
        return 1;
    }

    fseek(f, 0, SEEK_SET);
    byte_t* data = malloc(element_size * num_vectors * num_dimensions);
    long long got_elements = fread(data, element_size, num_elements, f);
    if (got_elements != num_elements) {
        fprintf(stderr, "Cannot read enough data from %s: requested %lld, got %lld",
                filename, num_elements, got_elements);
        return 1;
    }
    fclose(f);

    f = fopen(filename, "wb");
    save_vecs_light_meta_file(f, num_vectors, num_dimensions);
    fwrite(data, element_size, num_elements, f);
    fclose(f);

    return 0;
}
