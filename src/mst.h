#ifndef _MST_H
#define _MST_H

#include <stdio.h>

#include "misc.h"
#include "huffman.h"
#include "stats.h"

// NOTE: Internal computing structure
typedef struct _mst_edge {
    vector_id_t source;
    vector_id_t target;
    float dist;
} mst_edge_t;

// NOTE: This way is not very optimal but pretty simple.
typedef struct _tree_edge {
    vector_id_t target;
    // float dist; // NOTE: We don't need dists info
} tree_edge_t;

typedef struct _tree_vertex {
    tree_edge_t* edges;
    long long num_edges;
} tree_vertex_t;

typedef struct _tree {
    long long num_vertices;
    tree_vertex_t* vertices;

    long long num_edges; // NOTE: Formally it is a forest, not tree
    tree_edge_t* edges;
} tree_t;

// TODO: FIXME: float pq_penalty, const byte_t* pq_indices
mst_edge_t* load_mst_edges_from_nn_filenames(long long num_vectors, int num_nn_to_load, int pq_m,
                                             float pq_penalty, const byte_t* pq_indices,
                                             const char* indices_filename,
                                             const char* dist_filename);
mst_edge_t* load_mst_edges_from_nn_files(long long num_vectors, int num_nn_to_load, int pq_m,
                                         float pq_penalty, const byte_t* pq_indices,
                                         FILE* indices_file, FILE* dist_file);

void minimum_spanning_tree(tree_t* tree, long long num_vertices, int num_nn, mst_edge_t* edges);
void tree_destroy(tree_t* tree);

void tree_save_filename(const tree_t* tree, const char* filename);
void tree_save_file(const tree_t* tree, FILE* file);

void tree_load_filename(tree_t* tree, const char* filename);
void tree_load_file(tree_t* tree, FILE* file);

// TODO: Move to separate file
// TODO: bfs
int tree_collect_vertices_dfs(const tree_t* tree, vector_id_t* vertices, int* num_children);

#define TRAVERSER_NO_PARENT_VECTOR ((vector_id_t)-1)

typedef struct _tree_traverser_item {
    vector_id_t vertex_id;
    int num_children;
} tree_traverser_item_t;

typedef struct _tree_traverser {
    long long num_vertices;
    long long stack_size;
    tree_traverser_item_t* stack;
} tree_traverser_t;

void tree_traverser_init(tree_traverser_t* traverser, long long num_vertices);
void tree_traverser_destroy(tree_traverser_t* traverser);

void tree_traverser_reset(tree_traverser_t* traverser);
vector_id_t tree_traverser_get_active_parent(const tree_traverser_t* traverser);
void tree_traverser_push_vertex(tree_traverser_t* traverser, vector_id_t vertex_id,
                                int num_children);

double* tree_collect_num_children_stats(long long num_vectors, const int* num_children,
                                        int* alphabet_size_out);
int tree_collect_indices_stats(long long num_vectors, int pq_m, const byte_t* pq_indices,
                               const vector_id_t* vertices, const int* num_children, double* stats);
void tree_estimate_huffman_encoding(huffman_stats_t* indices_stats, huffman_stats_t* children_stats,
                                    const tree_t* tree, int pq_m, const byte_t* pq_indices);

#endif // _MST_H
