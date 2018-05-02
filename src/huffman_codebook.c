#include "huffman.h"

#include <assert.h>
#include <stdlib.h>

static void huffman_codebook_write_bit_length(FILE* file, unsigned int bit_length) {
    unsigned int bit_length_rest = bit_length;
    int continous_flag = 1;
    for (int byte_index = 0; byte_index < sizeof(bit_length) && continous_flag; ++byte_index) {
        byte_t byte = bit_length_rest & 0x7f;
        continous_flag = (bit_length_rest != byte);
        if (continous_flag) {
            byte |= (1 << 7);
        }
        fwrite(&byte, 1, sizeof(byte), file);
        bit_length_rest >>= 7;
    }

    if (continous_flag) {
        fprintf(stderr, "Cannot write bit_length of code because it's too large: %u\n", bit_length);
    }
}

static unsigned int huffman_codebook_read_bit_length(FILE* file) {
    unsigned int bit_length = 0U;
    int continous_flag = 1;
    for (int byte_index = 0; byte_index < sizeof(bit_length) && continous_flag; ++byte_index) {
        byte_t byte = 0;
        fread(&byte, 1, sizeof(byte), file);
        bit_length |= ((unsigned int) (byte & 0x7f) << (7 * byte_index));

        continous_flag = (byte >> 7) & 1;
    }
    return bit_length;
}

void huffman_codebook_save(const huffman_codebook_t* codebook, FILE* file) {
    fwrite(&codebook->alphabet_size, 1, sizeof(codebook->alphabet_size), file);
    byte_t is_context_byte = codebook->is_context;
    fwrite(&is_context_byte, 1, sizeof(is_context_byte), file);
    for (const huffman_code_item_t* item = codebook->items;
         item != codebook->items + codebook->alphabet_size;
         ++item)
    {
        huffman_codebook_write_bit_length(file, item->bit_length);
    }

    bit_stream_t* stream = bit_stream_create_from_file(file);
    for (const huffman_code_item_t* item = codebook->items;
         item != codebook->items + codebook->alphabet_size;
         ++item)
    {
        bit_stream_write(stream, item->code, item->bit_length);
    }
    stream = bit_stream_destroy_file(stream, 0);
}

void huffman_codebook_load(huffman_codebook_t* codebook, FILE* file) {
    fread(&codebook->alphabet_size, 1, sizeof(codebook->alphabet_size), file);
    byte_t is_context_byte = 0;
    fread(&is_context_byte, 1, sizeof(is_context_byte), file);
    codebook->is_context = is_context_byte;
    codebook->items = malloc(sizeof(*codebook->items) * codebook->alphabet_size);
    int bitfield_num_bytes = 0;
    for (huffman_code_item_t* item = codebook->items;
         item != codebook->items + codebook->alphabet_size;
         ++item)
    {
        item->bit_length = huffman_codebook_read_bit_length(file);
        bitfield_num_bytes += (item->bit_length + BYTE_NUM_BITS - 1) / BYTE_NUM_BITS;
    }
    codebook->codefield = malloc(sizeof(byte_t) * bitfield_num_bytes);
    byte_t* next_code = codebook->codefield;
    bit_stream_t* stream = bit_stream_create_from_file(file);
    for (huffman_code_item_t* item = codebook->items;
         item != codebook->items + codebook->alphabet_size;
         ++item)
    {
        if (item->bit_length > 0) {
            bit_stream_read(stream, next_code, item->bit_length);
            item->code = next_code;
            next_code += (item->bit_length + BYTE_NUM_BITS - 1) / BYTE_NUM_BITS;
        } else {
            item->code = NULL;
        }
    }
    stream = bit_stream_destroy_file(stream, 0);
}

void huffman_codebook_destroy(huffman_codebook_t* codebook) {
    free(codebook->codefield);
    codebook->codefield = NULL;

    free(codebook->items);
    codebook->items = NULL;

    codebook->alphabet_size = 0;
    codebook->is_context = 0;
}

#ifdef _HUFFMAN_CODEBOOK_TEST

// NOTE: should be greater then 0x7f = 127 to test multi-byte bit_length compression
#define NUM_SYMBOLS (1 << 15)
int main() {
    double* counts = malloc(sizeof(*counts) * NUM_SYMBOLS);
    counts[0] = 0.0;
    counts[1] = 1.0;
    for (int i = 2; i < NUM_SYMBOLS; ++i) {
        counts[i] = counts[i - 1] + counts[i - 2];
    }
    huffman_codebook_t codebook;
    huffman_codebook_encode_init(&codebook, NUM_SYMBOLS, &counts[0]);
    printf("Encoded\n");
    printf("The longest code is %d (1<<14 = %d)\n", codebook.items[1].bit_length, 1 << 14);

    FILE* f = fopen("/tmp/codebook.bin", "wb");
    huffman_codebook_save(&codebook, f);
    fclose(f);
    printf("Saved\n");

    huffman_codebook_t codebook2;
    f = fopen("/tmp/codebook.bin", "rb");
    huffman_codebook_load(&codebook2, f);
    fclose(f);
    printf("Loaded\n");

    assert(codebook.alphabet_size == codebook2.alphabet_size);
    for (int i = 0; i < codebook.alphabet_size; ++i) {
        assert(codebook.items[i].bit_length == codebook2.items[i].bit_length);
        for (int bit_index = 0; bit_index < codebook.items[i].bit_length; ++bit_index) {
            int byte_offset = bit_index / BYTE_NUM_BITS;
            int bit_offset = bit_index % BYTE_NUM_BITS;

            byte_t byte1 = codebook.items[i].code[byte_offset];
            byte_t byte2 = codebook2.items[i].code[byte_offset];

            int bit1 = (byte1 >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
            int bit2 = (byte2 >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
            assert(bit1 == bit2);
        }
    }

    huffman_codebook_destroy(&codebook2);
    huffman_codebook_destroy(&codebook);
    free(counts);
    return 0;
}
#undef NUM_SYMBOLS

#endif // _HUFFMAN_CODEBOOK_TEST
