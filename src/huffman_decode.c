#include "huffman.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

typedef struct _huffman_decoder_node {
    int symbol;
    struct _huffman_decoder_node* children[2];
} huffman_decoder_node_t;

typedef struct _huffman_decoder {
    huffman_decoder_node_t* root_node;
    huffman_decoder_node_t* current_node;
} huffman_decoder_t;

static huffman_decoder_node_t* huffman_decoder_add_code(huffman_decoder_node_t* node, int symbol,
                                                        const huffman_code_item_t* item, int depth) {
    if (!node) {
        node = malloc(sizeof(*node));
        node->symbol = HUFFMAN_NO_SYMBOL;
        node->children[0] = NULL;
        node->children[1] = NULL;
    }

    if (depth == item->bit_length) {
        node->symbol = symbol;
    } else {
        int byte_offset = depth / BYTE_NUM_BITS;
        int bit_offset = depth % BYTE_NUM_BITS;
        int bit_value = (item->code[byte_offset] >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
        node->children[bit_value] = huffman_decoder_add_code(node->children[bit_value], symbol,
                                                             item, depth + 1);
    }

    return node;
}

static huffman_decoder_node_t* huffman_decoder_free_trie(huffman_decoder_node_t* node) {
    if (!node) {
        return NULL;
    }
    huffman_decoder_free_trie(node->children[0]);
    huffman_decoder_free_trie(node->children[1]);
    free(node);

    return NULL;
}

huffman_decoder_t* huffman_decoder_create(const huffman_codebook_t* codebook) {
    // TODO: allocate nodes pool
    huffman_decoder_t* decoder = malloc(sizeof(huffman_decoder_t));
    for (int symbol_id = 0; symbol_id < codebook->alphabet_size; ++symbol_id) {
        decoder->root_node = huffman_decoder_add_code(decoder->root_node, symbol_id,
                                                      &codebook->items[symbol_id], 0);
    }
    decoder->current_node = decoder->root_node;
    return decoder;
}

huffman_decoder_t* huffman_decoder_destroy(huffman_decoder_t* decoder) {
    decoder->root_node = huffman_decoder_free_trie(decoder->root_node);
    decoder->current_node = NULL;
    free(decoder);
    return NULL;
}

void huffman_decoder_reset(huffman_decoder_t* decoder) {
    decoder->current_node = decoder->root_node;
}

int huffman_decoder_push_bit(huffman_decoder_t* decoder, int bit_value) {
    decoder->current_node = decoder->current_node->children[bit_value & 1];
    if (!decoder->current_node) {
        huffman_decoder_reset(decoder);
        return HUFFMAN_INVALID_SYMBOL;
    } else if (decoder->current_node->symbol != HUFFMAN_NO_SYMBOL) {
        int symbol = decoder->current_node->symbol;
        huffman_decoder_reset(decoder);
        return symbol;
    } else {
        return HUFFMAN_NO_SYMBOL;
    }
}

int huffman_decoder_push_bits(huffman_decoder_t* decoder, const byte_t* bits, int bit_length) {
    int push_result = HUFFMAN_NO_SYMBOL;
    int bit_index = 0;
    for (bit_index = 0;
         bit_index < bit_length && push_result == HUFFMAN_NO_SYMBOL;
         ++bit_index)
    {
        int byte_offset = bit_index / BYTE_NUM_BITS;
        int bit_offset = bit_index % BYTE_NUM_BITS;
        int bit_value = (bits[byte_offset] >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;

        push_result = huffman_decoder_push_bit(decoder, bit_value);
    }

    return (bit_index == bit_length) ? push_result : HUFFMAN_INVALID_SYMBOL;
}

#ifdef _HUFFMAN_DECODE_TEST

#define NUM_SYMBOLS 6
int main() {
    double counts[NUM_SYMBOLS] = {1.0, 4.0, 3.0, 8.0, 3.0, 8.0};
    huffman_codebook_t codebook;
    huffman_codebook_encode_init(&codebook, NUM_SYMBOLS, &counts[0]);
    huffman_decoder_t* decoder = huffman_decoder_create(&codebook);

    int seq[] = {
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0
    };
    char* decoded = malloc(32 * sizeof(*decoded));
    for (int i = 0; i < 31; ++i) {
        int symbol = seq[i];
        const huffman_code_item_t* item = &codebook.items[symbol];
        int result = huffman_decoder_push_bits(decoder, item->code, item->bit_length);
        assert(result >= 0 && "Cannot decode symbol");
        decoded[i] = 'a' + result;
    }
    decoded[31] = '\0';
    printf("Decoded: '%s'\n", decoded);
    assert(!strcmp(decoded, "abacabadabacabaeabacabadabacaba"));
    free(decoded);
    decoded = NULL;
    decoder = huffman_decoder_destroy(decoder);
    huffman_codebook_destroy(&codebook);
    return 0;
}
#undef NUM_SYMBOLS

#endif // _HUFFMAN_DECODE_TEST
