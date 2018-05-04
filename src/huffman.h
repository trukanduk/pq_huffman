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
    int is_context;
    int num_items; // NOTE: = alphabet_size * (is_context ? alphabet_size : 1)
    huffman_code_item_t* items;
} huffman_codebook_t;

// NOTE: Debug functions:
void huffman_dump_code(const huffman_code_item_t* item, FILE* f);
void huffman_codebook_dump(const huffman_codebook_t* codebook, FILE* f);

void huffman_codebook_save(const huffman_codebook_t* codebook, FILE* file);
void huffman_codebook_load(huffman_codebook_t* codebook, FILE* file);

void huffman_codebook_encode_init(huffman_codebook_t* codebook, int alphabet_size, const double* symbol_counts);
void huffman_codebook_context_encode_init(huffman_codebook_t* codebook, int alphabet_size, const double* symbol_counts);
void huffman_codebook_destroy(huffman_codebook_t* codebook);

double huffman_estimate_size(const huffman_codebook_t* codebook, const double* symbol_counts);

huffman_decoder_t* huffman_decoder_create(const huffman_codebook_t* codebook);
huffman_decoder_t* huffman_decoder_destroy(huffman_decoder_t* decoder); // NOTE: returns NULL

void huffman_decoder_reset(huffman_decoder_t* decoder);
void huffman_decoder_set_prev_symbol(huffman_decoder_t* decoder, int prev_symbol);
int huffman_decoder_push_bit(huffman_decoder_t* decoder, int bit_value);
int huffman_decoder_push_bits(huffman_decoder_t* decoder, const byte_t* bits, int bit_length);

#endif // _HUFFMAN_H
