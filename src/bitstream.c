#include "bitstream.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

// #define _BITSTREAM_DEBUG

struct _bit_stream {
    FILE* file;

    byte_t* buffer;
    int mode;
    int buffer_capacity;
    int buffer_filled_bits;
    int buffer_read_position;
};

enum {
    DEFAULT_BUFFER_SIZE = 1024,

    BIT_STREAM_MODE_UNKNOWN = 0,
    BIT_STREAM_MODE_READ = 1,
    BIT_STREAM_MODE_WRITE = 2
};

bit_stream_t* bit_stream_create_from_file(FILE* file) {
    return bit_stream_create_from_file_buffered(file, DEFAULT_BUFFER_SIZE);
}

bit_stream_t* bit_stream_create_from_file_buffered(FILE* file, int buffer_size_bytes) {
    bit_stream_t* stream = malloc(sizeof(bit_stream_t));
    stream->file = file;
    stream->buffer = malloc(sizeof(byte_t) * buffer_size_bytes);
    stream->mode = BIT_STREAM_MODE_UNKNOWN;
    stream->buffer_capacity = buffer_size_bytes * BYTE_NUM_BITS;
    stream->buffer_filled_bits = 0;
    stream->buffer_read_position = 0;

    return stream;
}

bit_stream_t* bit_stream_destroy(bit_stream_t* stream) {
    return bit_stream_destroy_file(stream, 0);
}

bit_stream_t* bit_stream_destroy_file(bit_stream_t* stream, int close_file) {
    if (stream->mode == BIT_STREAM_MODE_WRITE) {
        bit_stream_flush(stream, 1);
    }
    if (close_file) {
        fclose(stream->file);
    }
    free(stream->buffer);
    stream->buffer_capacity = 0;
    stream->buffer_filled_bits = 0;
    stream->buffer_read_position = 0;

    free(stream);
    return NULL;
}

static void bit_stream_write_impl(bit_stream_t* stream, const byte_t* data, int bit_length) {
    assert(stream->mode == BIT_STREAM_MODE_UNKNOWN || stream->mode == BIT_STREAM_MODE_WRITE);
    assert(stream->buffer_filled_bits + bit_length <= stream->buffer_capacity &&
           "bit_stream buffer overflow");

    stream->mode = BIT_STREAM_MODE_WRITE;
    int buffer_byte = stream->buffer_filled_bits / BYTE_NUM_BITS;
    int buffer_bit_offset = stream->buffer_filled_bits % BYTE_NUM_BITS;

    for (int data_bit = 0; data_bit < bit_length; ++data_bit) {
        // FIXME: bit significance is platform-dependent in general. So we ca write bytes
        //        in reversed order for some non-standard platform.
        // NOTE: write bit-by-bit to simplify the process.
        byte_t data_byte = data[data_bit / BYTE_NUM_BITS];
        int data_bit_offset = data_bit % BYTE_NUM_BITS;

        if ((data_byte >> (BYTE_NUM_BITS - data_bit_offset - 1)) & 0x01) {
            stream->buffer[buffer_byte] |= (1 << (BYTE_NUM_BITS - buffer_bit_offset - 1));
        } else {
            stream->buffer[buffer_byte] &= ~(1 << (BYTE_NUM_BITS - buffer_bit_offset - 1));
        }

        ++buffer_bit_offset;
        if (buffer_bit_offset == BYTE_NUM_BITS) {
            buffer_bit_offset = 0;
            ++buffer_byte;
        }
    }

    stream->buffer_filled_bits += bit_length;
}

int bit_stream_flush(bit_stream_t* stream, int add_pad) {
    if (add_pad && stream->buffer_filled_bits % BYTE_NUM_BITS != 0) {
        byte_t pad = 0;
        int pad_size = BYTE_NUM_BITS - stream->buffer_filled_bits % BYTE_NUM_BITS;
        bit_stream_write_impl(stream, &pad, pad_size);
        assert(stream->buffer_filled_bits % BYTE_NUM_BITS == 0 &&
               "bit_stream: something wrong with padding");
    }

    int num_bytes_to_write = stream->buffer_filled_bits / BYTE_NUM_BITS;
    int bytes_written = fwrite(stream->buffer, sizeof(byte_t), num_bytes_to_write, stream->file);
    assert(bytes_written == num_bytes_to_write && "bit_stream: cannot flush");
    if (stream->buffer_filled_bits % BYTE_NUM_BITS != 0) {
        stream->buffer[0] = stream->buffer[num_bytes_to_write];
    }

    stream->buffer_filled_bits -= (num_bytes_to_write * BYTE_NUM_BITS);

    return 1; // TODO
}

int bit_stream_write(bit_stream_t* stream, const byte_t* data, int bit_length) {
    while (bit_length > 0) {
        int batch_length = bit_length;
        // imin(bit_length, stream->buffer_capacity - stream->buffer_filled_bits);
        if (stream->buffer_filled_bits + batch_length > stream->buffer_capacity) {
            batch_length = (stream->buffer_capacity - stream->buffer_filled_bits) / BYTE_NUM_BITS * BYTE_NUM_BITS;
        }

        bit_stream_write_impl(stream, data, batch_length);

        if (stream->buffer_filled_bits + BYTE_NUM_BITS > stream->buffer_capacity) {
            bit_stream_flush(stream, 0);
        }

        bit_length -= batch_length;
        data += batch_length / BYTE_NUM_BITS;
    }

    return 1; // TODO:
}

static void bit_stream_read_buffer(bit_stream_t* stream) {
    stream->buffer_filled_bits =
            fread(stream->buffer, sizeof(byte_t), stream->buffer_capacity / BYTE_NUM_BITS, stream->file) * BYTE_NUM_BITS;
    if (stream->buffer_filled_bits != stream->buffer_capacity && ferror(stream->file)) {
        perror("bit_stream: error reading file");
    }
    stream->buffer_read_position = 0;
}

void bit_stream_read(bit_stream_t* stream, byte_t* data, int bit_length) {
    for (int bit_index = 0; bit_index < bit_length; ++bit_index) {
        int byte_offset = bit_index / BYTE_NUM_BITS;
        int bit_offset = bit_index % BYTE_NUM_BITS;
        int bit_value = bit_stream_read_bit(stream);
        if (bit_value) {
            data[byte_offset] |= (1 << (BYTE_NUM_BITS - bit_offset - 1));
        } else {
            data[byte_offset] &= ~(1 << (BYTE_NUM_BITS - bit_offset - 1));
        }
    }
}

int bit_stream_read_bit(bit_stream_t* stream) {
    assert(stream->mode == BIT_STREAM_MODE_UNKNOWN || stream->mode == BIT_STREAM_MODE_READ);

    if (stream->buffer_read_position == stream->buffer_filled_bits) {
        bit_stream_read_buffer(stream);
        if (stream->buffer_read_position == stream->buffer_filled_bits) {
            return EOF;
        }
    }

    int byte_num = stream->buffer_read_position / BYTE_NUM_BITS;
    int bit_num = stream->buffer_read_position % BYTE_NUM_BITS;
    ++stream->buffer_read_position;
    return (stream->buffer[byte_num] >> (BYTE_NUM_BITS - bit_num - 1)) & 0x01;
}

#ifdef _BITSTREAM_TEST

int main() {
    FILE* f = fopen("/tmp/foo.bin", "wb");
    bit_stream_t* stream = bit_stream_create_from_file_buffered(f, 2);
    byte_t data[7] = {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde};
    bit_stream_write(stream, data, 4); // 0x1
    bit_stream_write(stream, data + 1, 12); // 0x345
    bit_stream_write(stream, data + 3, 28); // 0x789abcd
    stream = bit_stream_destroy(stream);
    fclose(f);

    // return 0;
    f = fopen("/tmp/foo.bin", "rb");
    stream = bit_stream_create_from_file_buffered(f, 2);
    char got_str[4 + 12 + 28 + 1];
    int str_index = 0;
    int bit;
    while ((bit = bit_stream_read_bit(stream)) != EOF) {
        got_str[str_index++] = '0' + bit;
    }
    got_str[str_index] = '\0';
    printf("Red %d bits: '%s'\n", str_index, got_str);
    //                              2   3   4   6   7   8   9   a   b   c   d pad
    assert(!strcmp(&got_str[0], "000100110100010101111000100110101011110011010000"));
    assert(bit_stream_read_bit(stream) == EOF);
    stream = bit_stream_destroy(stream);
    fclose(f);

    return 0;
}

#endif // _BITSTREAM_TEST
