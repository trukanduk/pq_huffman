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
    int alphabet_size;
    int is_context;
    int noparent_symbol_length;
    int prev_symbol;
    int current_noparent_symbol_length;
    int current_noparent_symbol;
} huffman_decoder_t;

static void huffman_decoder_init_empty_node(huffman_decoder_node_t* node) {
    node->symbol = HUFFMAN_NO_SYMBOL;
    node->children[0] = NULL;
    node->children[1] = NULL;
}

static huffman_decoder_node_t* huffman_decoder_add_code(huffman_decoder_node_t* node, int symbol,
                                                        const huffman_code_item_t* item, int depth) {
    if (!node) {
        node = malloc(sizeof(*node));
        huffman_decoder_init_empty_node(node);
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

static huffman_decoder_node_t* huffman_decoder_free_trie(huffman_decoder_node_t* node,
                                                         int free_this) {
    if (!node) {
        return NULL;
    }
    huffman_decoder_free_trie(node->children[0], 1);
    huffman_decoder_free_trie(node->children[1], 1);
    if (free_this) {
        free(node);
    }

    return NULL;
}

huffman_decoder_t* huffman_decoder_create(const huffman_codebook_t* codebook) {
    // TODO: allocate nodes pool
    huffman_decoder_t* decoder = malloc(sizeof(huffman_decoder_t));
    decoder->alphabet_size = codebook->alphabet_size;
    decoder->is_context = codebook->is_context;
    decoder->prev_symbol = HUFFMAN_NO_SYMBOL;
    decoder->current_noparent_symbol_length = 0;
    decoder->current_noparent_symbol = 0;

    decoder->noparent_symbol_length = 1;
    while ((1 << decoder->noparent_symbol_length) < decoder->alphabet_size) {
        ++decoder->noparent_symbol_length;
    }

    if (decoder->is_context) {
        decoder->root_node = malloc(sizeof(*decoder->root_node) * decoder->alphabet_size);
        for (int src_symbol_id = 0; src_symbol_id < decoder->alphabet_size; ++src_symbol_id) {
            huffman_decoder_init_empty_node(decoder->root_node + src_symbol_id);
            for (int symbol_id = 0; symbol_id < codebook->alphabet_size; ++symbol_id) {
                huffman_decoder_add_code(
                        decoder->root_node + src_symbol_id, symbol_id,
                        &codebook->items[src_symbol_id * decoder->alphabet_size + symbol_id], 0);
            }
        }
        decoder->current_node = NULL;
    } else {
        decoder->root_node = NULL;
        for (int symbol_id = 0; symbol_id < codebook->alphabet_size; ++symbol_id) {
            decoder->root_node = huffman_decoder_add_code(decoder->root_node, symbol_id,
                                                          &codebook->items[symbol_id], 0);
        }
        decoder->current_node = decoder->root_node;
    }
    return decoder;
}

huffman_decoder_t* huffman_decoder_destroy(huffman_decoder_t* decoder) {
    if (decoder->is_context) {
        for (int symbol_id = 0; symbol_id < decoder->alphabet_size; ++symbol_id) {
            huffman_decoder_free_trie(decoder->root_node + symbol_id, 0);
        }
        free(decoder->root_node);
        decoder->root_node = NULL;
    } else {
        decoder->root_node = huffman_decoder_free_trie(decoder->root_node, 1);
    }
    decoder->current_node = NULL;
    free(decoder);
    return NULL;
}

void huffman_decoder_reset(huffman_decoder_t* decoder) {
    huffman_decoder_set_prev_symbol(decoder, HUFFMAN_NO_SYMBOL);
}

void huffman_decoder_set_prev_symbol(huffman_decoder_t* decoder, int prev_symbol) {
    if (!decoder->is_context) {
        decoder->current_node = decoder->root_node;
        return;
    }

    decoder->current_noparent_symbol_length = 0;
    decoder->current_noparent_symbol = 0;
    decoder->prev_symbol = prev_symbol;

    if (prev_symbol == HUFFMAN_NO_SYMBOL) {
        decoder->current_node = NULL;
    } else {
        assert(prev_symbol < decoder->alphabet_size);
        decoder->current_node = decoder->root_node + prev_symbol;
    }
}

int huffman_decoder_push_bit(huffman_decoder_t* decoder, int bit_value) {
    if (decoder->is_context && decoder->prev_symbol == HUFFMAN_NO_SYMBOL) {
        // NOTE: warming mode

        decoder->current_noparent_symbol = (decoder->current_noparent_symbol << 1) + (bit_value & 1);
        ++decoder->current_noparent_symbol_length;
        if (decoder->current_noparent_symbol_length >= decoder->noparent_symbol_length) {
            int current_symbol = decoder->current_noparent_symbol;
            assert(decoder->current_noparent_symbol < decoder->alphabet_size);

            huffman_decoder_set_prev_symbol(decoder, current_symbol);
            return current_symbol;
        } else {
            return HUFFMAN_NO_SYMBOL;
        }
    }

    decoder->current_node = decoder->current_node->children[bit_value & 1];
    if (!decoder->current_node) {
        huffman_decoder_reset(decoder);
        return HUFFMAN_INVALID_SYMBOL;
    } else if (decoder->current_node->symbol != HUFFMAN_NO_SYMBOL) {
        int symbol = decoder->current_node->symbol;
        huffman_decoder_set_prev_symbol(decoder, symbol);
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

int huffman_decoder_read_symbol(huffman_decoder_t* decoder, bit_stream_t* stream) {
    int symbol = HUFFMAN_NO_SYMBOL;
    while (symbol == HUFFMAN_NO_SYMBOL) {
        int bit_value = bit_stream_read_bit(stream);
        symbol = huffman_decoder_push_bit(decoder, bit_value);
    }
    return symbol;
}

#ifdef _HUFFMAN_DECODE_TEST

#define NUM_SYMBOLS 6
static void non_context_test() {
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
}

static void context_test() {
    double counts[NUM_SYMBOLS * NUM_SYMBOLS] = {
        1.0, 4.0, 3.0, 8.0, 3.0, 8.0,
        3.0, 9.0, 4.0, 5.0, 2.0, 4.0,
        9.0, 4.0, 3.0, 2.0, 8.0, 7.0,
        6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        5.0, 4.0, 2.0, 6.0, 9.0, 4.0,
        3.0, 7.0, 3.0, 6.0, 9.0, 3.0
    };

    huffman_codebook_t codebook;
    huffman_codebook_context_encode_init(&codebook, NUM_SYMBOLS, &counts[0]);
    huffman_decoder_t* decoder = huffman_decoder_create(&codebook);

    char decoded[NUM_SYMBOLS * 2 + 1];
    char expected[NUM_SYMBOLS * 2 + 1];
    decoded[NUM_SYMBOLS * 2] = 0;
    expected[NUM_SYMBOLS * 2] = 0;

    for (int starting_symbol_id = 0; starting_symbol_id < NUM_SYMBOLS; ++starting_symbol_id) {
        huffman_decoder_reset(decoder);
        int prev_symbol = HUFFMAN_NO_SYMBOL;
        for (int symbol_id = 0; symbol_id < NUM_SYMBOLS; ++symbol_id) {
            int got_symbol;
            if (prev_symbol == HUFFMAN_NO_SYMBOL) {
                byte_t symbol_byte = starting_symbol_id << (BYTE_NUM_BITS - 3);
                got_symbol = huffman_decoder_push_bits(decoder, &symbol_byte, 3);
            } else {
                const huffman_code_item_t* item = codebook.items
                        + prev_symbol * NUM_SYMBOLS + starting_symbol_id;
                got_symbol = huffman_decoder_push_bits(decoder, item->code, item->bit_length);
            }
            if (got_symbol < 0) {
                printf("Got %d symbol!\n", got_symbol);
            }
            decoded[symbol_id * 2] = 'a' + got_symbol;
            expected[symbol_id * 2] = 'a' + starting_symbol_id;

            const huffman_code_item_t* item = codebook.items
                        + starting_symbol_id * NUM_SYMBOLS + symbol_id;
            got_symbol = huffman_decoder_push_bits(decoder, item->code, item->bit_length);
            if (got_symbol < 0) {
                printf("Got %d symbol!\n", got_symbol);
            }
            decoded[symbol_id * 2 + 1] = 'a' + got_symbol;
            expected[symbol_id * 2 + 1] = 'a' + symbol_id;

            prev_symbol = symbol_id;
        }

        printf("Decoded %d: %s (expected %s)\n", starting_symbol_id, decoded, expected);
        assert(!strcmp(decoded, expected));
    }

    decoder = huffman_decoder_destroy(decoder);
    huffman_codebook_destroy(&codebook);
}

int main() {
    non_context_test();
    context_test();
    return 0;
}
#undef NUM_SYMBOLS

#endif // _HUFFMAN_DECODE_TEST
