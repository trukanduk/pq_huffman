#ifndef _BITSTREAM_H
#define _BITSTREAM_H

#include <stdio.h>

#include "misc.h"

enum {
    BYTE_NUM_BITS = sizeof(byte_t) * 8
};

typedef struct _bit_stream bit_stream_t;

bit_stream_t* bit_stream_create_from_file(FILE* file);
bit_stream_t* bit_stream_create_from_file_buffered(FILE* file, long long buffer_size_bytes);
bit_stream_t* bit_stream_destroy(bit_stream_t* stream); // NOTE: Always returns NULL
bit_stream_t* bit_stream_destroy_file(bit_stream_t* stream, int close_file); // NOTE: Always returns NULL

int bit_stream_flush(bit_stream_t* stream, int add_pad);
int bit_stream_write(bit_stream_t* stream, const byte_t* data, long long bit_length);
void bit_stream_read(bit_stream_t* stream, byte_t* data, long long bit_length);
int bit_stream_read_bit(bit_stream_t* stream);

#endif // _BITSTREAM_H
