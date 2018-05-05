#ifndef _HUFFMAN_MISC_H
#define _HUFFMAN_MISC_H

typedef unsigned int vector_id_t;

int imin(int a, int b);
long long iminll(long long a, long long b);
long long iclampll(long long value, long long min_value, long long max_value);

// NOTE: Allocates with malloc
char* concat(const char* prefix, const char* suffix);

long long load_num_elements(const char* filename, long long element_size);

#ifndef __linux__
#   error "need usleep function!"
#endif
#include <unistd.h>

int usleep(unsigned usec);

#define MS 1000

typedef unsigned char byte_t;

#endif // _HUFFMAN_MISC_H
