#ifndef _HUFFMAN_MISC_H
#define _HUFFMAN_MISC_H

typedef unsigned int vector_id_t;

int imin(int a, int b);
long long iminll(long long a, long long b);
long long iclampll(long long value, long long min_value, long long max_value);

// NOTE: Allocates with malloc
char* concat(const char* prefix, const char* suffix);

#ifndef __linux__
#   error "need usleep function!"
#endif
#include <unistd.h>

int usleep(unsigned usec);

#define MS 1000

#endif // _HUFFMAN_MISC_H
