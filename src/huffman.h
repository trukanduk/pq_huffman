#ifndef _HUFFMAN_H
#define _HUFFMAN_H

#include "bitstream.h"

enum {
    HUFFMAN_NO_SYMBOL = -1,
    HUFFMAN_INVALID_SYMBOL = -2
};

typedef struct _huffman_decoder huffman_decoder_t;

typedef struct _huffman_code_item {
    const byte_t* code;
    int bit_length;
} huffman_code_item_t;

typedef struct _huffman_codebook {
    byte_t* codefield;
    int alphabet_size;
    huffman_code_item_t* items;
} huffman_codebook_t;

void huffman_codebook_encode_init(huffman_codebook_t* codebook, int alphabet_size, float* symbol_counts);
void huffman_codebook_destroy(huffman_codebook_t* codebook); // NOTE: returns NULL

huffman_decoder_t* huffman_decoder_create(huffman_codebook_t* codebook);
huffman_decoder_t* huffman_decoder_destroy(huffman_decoder_t* decoder); // NOTE: returns NULL

void huffman_decoder_reset(huffman_decoder_t* decoder);
int huffman_decoder_push_bit(huffman_decoder_t* decoder, int bit_value);
int huffman_decoder_push_bits(huffman_decoder_t* decoder, const byte_t* bits, int bit_length);

#endif // _HUFFMAN_H
