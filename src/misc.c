#include "misc.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int imin(int a, int b) {
    return (a < b ? a : b);
}

long long iminll(long long a, long long b) {
    return (a < b ? a : b);
}

long long iclampll(long long value, long long min_value, long long max_value) {
    if (value <= min_value) {
        return min_value;
    } else if (value >= max_value) {
        return max_value;
    } else {
        return value;
    }
}

char* concat(const char* prefix, const char* suffix) {
    int prefix_length = strlen(prefix);
    int suffix_length = strlen(suffix);
    char* result = malloc(sizeof(*result) * (prefix_length + suffix_length + 1));
    char* result_it = result;
    for (const char* prefix_it = prefix; *prefix_it; ++prefix_it, ++result_it) {
        *result_it = *prefix_it;
    }
    for (const char* suffix_it = suffix; *suffix_it; ++suffix_it, ++result_it) {
        *result_it = *suffix_it;
    }
    *result_it = '\0';
    return result;
}

long long load_num_elements(const char* filename, long long element_size) {
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    fclose(f);

    long long result = file_size / element_size;
    assert(result * element_size == file_size);
    return result;
}
