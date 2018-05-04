#include "huffman.h"

#include <assert.h>
#include <stdlib.h>

void huffman_dump_code(const huffman_code_item_t* item, FILE* f) {
    if (!item->code) {
        fprintf(f, "-");
        return;
    }

    for (int bit_index = 0; bit_index < item->bit_length; ++bit_index) {
        int byte_offset = bit_index / BYTE_NUM_BITS;
        int bit_offset = bit_index % BYTE_NUM_BITS;
        int bit_value = (item->code[byte_offset] >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
        fprintf(f, "%d", bit_value);
    }
    fprintf(f, " (%d)", item->bit_length);
}

void huffman_codebook_dump(const huffman_codebook_t* codebook, FILE* f) {
    for (int item_index = 0; item_index < codebook->num_items; ++item_index) {
        if (codebook->is_context) {
            int src_symbol = item_index / codebook->alphabet_size;
            int dst_symbol = item_index % codebook->alphabet_size;
            fprintf(f, "%d -> %d: ", src_symbol, dst_symbol);
        } else {
            fprintf(f, "%d: ", item_index);
        }
        huffman_dump_code(codebook->items + item_index, f);
        fprintf(f, "\n");
    }
}

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
         item != codebook->items + codebook->num_items;
         ++item)
    {
        huffman_codebook_write_bit_length(file, item->bit_length);
    }

    bit_stream_t* stream = bit_stream_create_from_file(file);
    for (const huffman_code_item_t* item = codebook->items;
         item != codebook->items + codebook->num_items;
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

    codebook->num_items = codebook->alphabet_size;
    if (codebook->is_context) {
        codebook->num_items *= codebook->alphabet_size;
    }

    codebook->items = malloc(sizeof(*codebook->items) * codebook->num_items);
    int bitfield_num_bytes = 0;
    for (huffman_code_item_t* item = codebook->items;
         item != codebook->items + codebook->num_items;
         ++item)
    {
        item->bit_length = huffman_codebook_read_bit_length(file);
        bitfield_num_bytes += (item->bit_length + BYTE_NUM_BITS - 1) / BYTE_NUM_BITS;
    }
    codebook->codefield = malloc(sizeof(byte_t) * bitfield_num_bytes);
    byte_t* next_code = codebook->codefield;
    bit_stream_t* stream = bit_stream_create_from_file(file);
    for (huffman_code_item_t* item = codebook->items;
         item != codebook->items + codebook->num_items;
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

// #define NUM_SYMBOLS (1 << 15)
void run(int context, int num_symbols) {
    printf("Starting %s test\n", context ? "context" : "non-context");
    int num_items = num_symbols;
    if (context) {
        num_items *= num_symbols;
    }
    double* counts = malloc(sizeof(*counts) * num_items);
    counts[0] = 0.0;
    counts[1] = 1.0;
    for (int i = 2; i < num_items; ++i) {
        counts[i] = counts[i - 1] + counts[i - 2];
    }
    huffman_codebook_t codebook;
    if (context) {
        huffman_codebook_context_encode_init(&codebook, num_symbols, counts);
    } else {
        huffman_codebook_encode_init(&codebook, num_symbols, counts);
    }
    // printf("The longest code is %d (1<<7 = %d)\n", codebook.items[1].bit_length, 1 << 7);

    FILE* f = fopen("/tmp/codebook.bin", "wb");
    int x = 0;
    fwrite(&x, sizeof(x), 1, f);
    huffman_codebook_save(&codebook, f);
    huffman_codebook_save(&codebook, f);
    fclose(f);

    huffman_codebook_t codebook2;
    huffman_codebook_t codebook3;
    f = fopen("/tmp/codebook.bin", "rb");
    fread(&x, sizeof(x), 1, f);
    huffman_codebook_load(&codebook2, f);
    huffman_codebook_load(&codebook3, f);
    fclose(f);

    assert(codebook.is_context == codebook2.is_context);
    assert(codebook.is_context == codebook2.is_context);

    assert(codebook.alphabet_size == codebook2.alphabet_size);
    assert(codebook.alphabet_size == codebook3.alphabet_size);
    for (int i = 0; i < num_items; ++i) {
        assert(codebook.items[i].bit_length == codebook2.items[i].bit_length);
        assert(codebook.items[i].bit_length == codebook3.items[i].bit_length);
        for (int bit_index = 0; bit_index < codebook.items[i].bit_length; ++bit_index) {
            int byte_offset = bit_index / BYTE_NUM_BITS;
            int bit_offset = bit_index % BYTE_NUM_BITS;

            byte_t byte1 = codebook.items[i].code[byte_offset];
            byte_t byte2 = codebook2.items[i].code[byte_offset];
            byte_t byte3 = codebook3.items[i].code[byte_offset];

            int bit1 = (byte1 >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
            int bit2 = (byte2 >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
            int bit3 = (byte3 >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
            assert(bit1 == bit2);
            assert(bit1 == bit3);
        }
    }

    huffman_codebook_destroy(&codebook3);
    huffman_codebook_destroy(&codebook2);
    huffman_codebook_destroy(&codebook);
    free(counts);
    printf("OK\n\n");
}

// NOTE: max length should be greater then 0x7f = 127 to test multi-byte bit_length compression
int main() {
    run(0, 1<<10);
    run(1, 1<<5);
    return 0;
}

#endif // _HUFFMAN_CODEBOOK_TEST
