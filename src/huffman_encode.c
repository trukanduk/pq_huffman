#include "huffman.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _huffman_encoding_tree_node {
    float sum_count;
    int symbol_id; // NOTE: -1 if is not leaf
    struct _huffman_encoding_tree_node* children[2];
} huffman_encoding_tree_node_t;

typedef struct _huffman_encoding_heap {
    huffman_encoding_tree_node_t** elements;
    int size;
    int capacity;
} huffman_encoding_heap_t;

static void huffman_heap_init(huffman_encoding_heap_t* heap, int capacity) {
    heap->capacity = capacity;
    heap->size = 0;
    heap->elements = calloc(sizeof(huffman_encoding_tree_node_t*), capacity);
}

static void huffman_heap_destroy(huffman_encoding_heap_t* heap) {
    free(heap->elements);
    heap->elements = NULL;
    heap->capacity = 0;
    heap->size = 0;
}

static void huffman_encoding_heap_push(huffman_encoding_heap_t* heap,
                                       huffman_encoding_tree_node_t* element) {
    int index = heap->size++;
    while (index > 0) {
        int parent_index = (index + 1) / 2 - 1;
        if (element->sum_count >= heap->elements[parent_index]->sum_count) {
            break;
        }

        heap->elements[index] = heap->elements[parent_index];
        index = parent_index;
    }
    heap->elements[index] = element;
}

static huffman_encoding_tree_node_t* huffman_encoding_heap_pop(huffman_encoding_heap_t* heap) {
    huffman_encoding_tree_node_t* result = heap->elements[0];
    huffman_encoding_tree_node_t* push_node = heap->elements[--heap->size];
    int index = 0;
    while (index < heap->size) {
        int left_child = (index + 1) * 2 - 1;
        int right_child = (index + 1) * 2;

        if (left_child < heap->size
            && push_node->sum_count > heap->elements[left_child]->sum_count
            && (right_child >= heap->size
                || heap->elements[left_child]->sum_count <= heap->elements[right_child]->sum_count))
        {
            heap->elements[index] = heap->elements[left_child];
            index = left_child;
        } else if (right_child < heap->size
                   && push_node->sum_count > heap->elements[right_child]->sum_count
                   && heap->elements[right_child]->sum_count
                        <= heap->elements[left_child]->sum_count)
        {
            heap->elements[index] = heap->elements[right_child];
            index = right_child;
        } else {
            break;
        }
    }
    heap->elements[index] = push_node;
    return result;
}

static void huffman_encoding_traverse_tree_collect_statistics(huffman_encoding_tree_node_t* node,
                                                              int current_depth,
                                                              int* sum_length_bytes,
                                                              int* max_length_bytes) {
    if (node->symbol_id != HUFFMAN_NO_SYMBOL) {
        int current_length_bytes = (current_depth + BYTE_NUM_BITS - 1) / BYTE_NUM_BITS;
        *sum_length_bytes += current_length_bytes;
        if (*max_length_bytes < current_length_bytes) {
            *max_length_bytes = current_length_bytes;
        }
    } else {
        huffman_encoding_traverse_tree_collect_statistics(node->children[0], current_depth + 1,
                                                          sum_length_bytes, max_length_bytes);
        huffman_encoding_traverse_tree_collect_statistics(node->children[1], current_depth + 1,
                                                          sum_length_bytes, max_length_bytes);
    }
}

static int huffman_encoding_traverse_tree_collect_codes(huffman_codebook_t* codebook,
                                                        int* code_byte_start,
                                                        huffman_encoding_tree_node_t* node,
                                                        int current_depth,
                                                        byte_t* current_code) {
    if (node->symbol_id != HUFFMAN_NO_SYMBOL) {
        int code_length_bytes = (current_depth + BYTE_NUM_BITS - 1) / BYTE_NUM_BITS;
        memcpy(codebook->codefield + *code_byte_start, current_code, code_length_bytes);
        codebook->items[node->symbol_id].code = codebook->codefield + *code_byte_start;
        codebook->items[node->symbol_id].bit_length = current_depth;
        *code_byte_start += code_length_bytes;
    } else {
        int code_byte = current_depth / BYTE_NUM_BITS;
        int code_bit = current_depth % BYTE_NUM_BITS;

        // NOTE: start with 1 to keep zero in current bit after done
        current_code[code_byte] |= (1 << (BYTE_NUM_BITS - code_bit - 1));
        huffman_encoding_traverse_tree_collect_codes(codebook, code_byte_start, node->children[1],
                                                     current_depth + 1, current_code);

        current_code[code_byte] &= ~(1 << (BYTE_NUM_BITS - code_bit - 1));
        huffman_encoding_traverse_tree_collect_codes(codebook, code_byte_start, node->children[0],
                                                     current_depth + 1, current_code);
    }
}

void huffman_codebook_encode_init(huffman_codebook_t* codebook, int alphabet_size,
                                  float* symbol_counts) {
    int num_nodes = 2 * alphabet_size - 1;
    huffman_encoding_tree_node_t* tree_nodes =
            malloc(sizeof(huffman_encoding_tree_node_t) * num_nodes);
    huffman_encoding_heap_t heap;
    huffman_heap_init(&heap, alphabet_size);
    for (int node_index = 0; node_index < num_nodes; ++node_index) {
        // NOTE: Create node for zero symbols but don't push them to queue
        tree_nodes[node_index].children[0] = NULL;
        tree_nodes[node_index].children[1] = NULL;

        if (node_index < alphabet_size) {
            tree_nodes[node_index].sum_count = symbol_counts[node_index];
            tree_nodes[node_index].symbol_id = node_index;

            if (symbol_counts[node_index] > 0.0f) {
                huffman_encoding_heap_push(&heap, &tree_nodes[node_index]);
            }
        } else {
            tree_nodes[node_index].sum_count = 0.0;
            tree_nodes[node_index].symbol_id = HUFFMAN_NO_SYMBOL;
        }
    }
    int used_nodes = alphabet_size;


    while (heap.size > 1) {
        huffman_encoding_tree_node_t* zero_node = huffman_encoding_heap_pop(&heap);
        huffman_encoding_tree_node_t* one_node = huffman_encoding_heap_pop(&heap);

        huffman_encoding_tree_node_t* new_node = &tree_nodes[used_nodes++];
        new_node->sum_count = zero_node->sum_count + one_node->sum_count;
        new_node->symbol_id = HUFFMAN_NO_SYMBOL;
        new_node->children[0] = zero_node;
        new_node->children[1] = one_node;
        huffman_encoding_heap_push(&heap, new_node);
    }

    huffman_encoding_tree_node_t* root_node = heap.elements[0];
    int bytes_to_allocate = 0;
    int longest_code_bytes = 0;
    huffman_encoding_traverse_tree_collect_statistics(root_node, 0, &bytes_to_allocate,
                                                      &longest_code_bytes);
    codebook->codefield = malloc(bytes_to_allocate * sizeof(byte_t));
    codebook->alphabet_size = alphabet_size;
    codebook->items = malloc(alphabet_size * sizeof(*codebook->items));
    for (int symbol_id = 0; symbol_id < alphabet_size; ++symbol_id) {
        codebook->items[symbol_id].bit_length = 0;
        codebook->items[symbol_id].code = NULL;
    }

    int code_byte_start = 0;
    byte_t* current_code = calloc(sizeof(*current_code), longest_code_bytes);
    huffman_encoding_traverse_tree_collect_codes(codebook, &code_byte_start, root_node, 0,
                                                 current_code);
    free(current_code);
    current_code = NULL;

    huffman_heap_destroy(&heap);
    free(tree_nodes);
    tree_nodes = NULL;
}

#ifdef _HUFFMAN_ENCODE_TEST

void print_code(const huffman_code_item_t* item) {
    if (!item->code) {
        printf("-");
        return;
    }

    for (int bit_index = 0; bit_index < item->bit_length; ++bit_index) {
        int byte_offset = bit_index / BYTE_NUM_BITS;
        int bit_offset = bit_index % BYTE_NUM_BITS;
        int bit_value = (item->code[byte_offset] >> (BYTE_NUM_BITS - bit_offset - 1)) & 1;
        printf("%d", bit_value);
    }
    printf(" (%d)", item->bit_length);
}

#define NUM_SYMBOLS 10
int main() {
    float counts[NUM_SYMBOLS] = {0.f, 21.f, 13.f, 8.f, 5.f, 3.f, 2.f, 1.f, 1.f, 0.f};
    huffman_codebook_t codebook;
    huffman_codebook_encode_init(&codebook, NUM_SYMBOLS, &counts[0]);
    for (int i = 0; i < NUM_SYMBOLS; ++i) {
        printf("%d: ", i);
        print_code(&codebook.items[i]);
        printf("\n");
    }

    huffman_codebook_destroy(&codebook);
    return 0;
}
#undef NUM_SYMBOLS

#endif // _HUFFMAN_ENCODE_TEST
