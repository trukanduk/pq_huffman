#include "mst.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "dsu.h"
#include "vecs_io.h"

enum {
    NN_LOADER_BATCH_SIZE = 128 * 1024 // NOTE: vectors
};

static int mst_dist_comparator(const void* first, const void* second) {
    float first_dist = ((const mst_edge_t*) first)->dist;
    float second_dist = ((const mst_edge_t*) second)->dist;
    float diff = first_dist - second_dist;
    if (fabs(diff) < 1e-9) {
        return 0;
    } else if (diff < 0) {
        return -1;
    } else {
        return 1;
    }
}

static int mst_source_comparator(const void* first, const void* second) {
    long long first_dist = ((const mst_edge_t*) first)->source;
    long long second_dist = ((const mst_edge_t*) second)->source;
    return first_dist - second_dist;
}

static void swap_edges(mst_edge_t* first, mst_edge_t* second) {
    if (first == second) {
        return;
    }

    vector_id_t tmp_vec = first->source;
    first->source = second->source;
    second->source = tmp_vec;

    tmp_vec = first->target;
    first->target = second->target;
    second->target = tmp_vec;

    double tmp_dist = first->dist;
    first->dist = second->dist;
    second->dist = tmp_dist;
}

static void restore_tree_edges_pointers(tree_t* tree) {
    tree_edge_t* tree_edge_it = tree->edges;
    for (tree_vertex_t* vertex_it = tree->vertices;
         vertex_it != tree->vertices + tree->num_vertices;
         tree_edge_it += vertex_it->num_edges, ++vertex_it)
    {
        vertex_it->edges = tree_edge_it;
    }
    assert(tree_edge_it == tree->edges + tree->num_edges);
}

mst_edge_t* load_mst_edges_from_nn_filenames(long long num_vectors, int num_nn_to_load, int pq_m,
                                             float pq_penalty, const byte_t* pq_indices,
                                             const char* indices_filename,
                                             const char* dist_filename) {
    FILE* indices_file = fopen(indices_filename, "rb");
    FILE* dist_file = fopen(dist_filename, "rb");

    mst_edge_t* result = load_mst_edges_from_nn_files(num_vectors, num_nn_to_load, pq_m,
                                                      pq_penalty, pq_indices, indices_file,
                                                      dist_file);

    fclose(indices_file);
    fclose(dist_file);

    return result;
}

mst_edge_t* load_mst_edges_from_nn_files(long long num_vectors, int num_nn_to_load, int pq_m,
                                         float pq_penalty, const byte_t* pq_indices,
                                         FILE* indices_file, FILE* dist_file) {
    // TODO: FIXME: float pq_penalty, const byte_t* pq_indices

    long long num_vectors_ind, num_vectors_dist;
    int num_nn, num_nn_dist;
    load_vecs_light_meta_file(indices_file, &num_vectors_ind, &num_nn);
    load_vecs_light_meta_file(dist_file, &num_vectors_dist, &num_nn_dist);
    if (num_vectors_ind != num_vectors || num_vectors != num_vectors_dist || num_nn != num_nn_dist) {
        fprintf(stderr, "Does indices and dist files are of the same dataset?\n");
        fprintf(stderr, "\tN_expected=%lld, N_indices=%lld, N_dist=%lld,"
                "NNN_indices=%d, NNN_dist=%d\n", num_vectors, num_vectors_ind, num_vectors_dist,
                num_nn, num_nn_dist);
        assert(num_vectors != num_vectors_ind);
        assert(num_vectors != num_vectors_dist);
        assert(num_nn != num_nn_dist);
    }
    assert(num_nn_to_load <= num_nn);

    mst_edge_t* result = malloc(sizeof(*result) * num_vectors * num_nn_to_load);
    mst_edge_t* result_it = result;

    vector_id_t* indices_batch = malloc(sizeof(*indices_batch) * num_nn * NN_LOADER_BATCH_SIZE);
    long long got_vectors = 0;
    while (got_vectors < num_vectors) {
        long long current_batch_size = iminll(NN_LOADER_BATCH_SIZE, num_vectors - got_vectors);
        fread(indices_batch, sizeof(vector_id_t) * num_nn, current_batch_size, indices_file);
        for (long long vec_index = 0; vec_index < current_batch_size; ++vec_index) {
            for (int nn_index = 0; nn_index < num_nn_to_load; ++nn_index, ++result_it) {
                result_it->source = got_vectors + vec_index;
                result_it->target = indices_batch[num_nn * vec_index + nn_index];
            }
        }
        got_vectors += current_batch_size;
    }
    assert(result_it == result + num_vectors * num_nn_to_load);
    free(indices_batch);
    indices_batch = NULL;

    result_it = result;
    float* dist_batch = malloc(sizeof(*dist_batch) * num_nn * NN_LOADER_BATCH_SIZE);
    got_vectors = 0;
    while (got_vectors < num_vectors) {
        long long current_batch_size = iminll(NN_LOADER_BATCH_SIZE, num_vectors - got_vectors);
        fread(dist_batch, sizeof(vector_id_t) * num_nn, current_batch_size, dist_file);
        for (long long vec_index = 0; vec_index < current_batch_size; ++vec_index) {
            for (int nn_index = 0; nn_index < num_nn_to_load; ++nn_index, ++result_it) {
                result_it->dist = dist_batch[num_nn * vec_index + nn_index];
            }
        }
        got_vectors += current_batch_size;
    }
    assert(result_it == result + num_vectors * num_nn_to_load);
    free(dist_batch);
    dist_batch = NULL;

    return result;
}

void minimum_spanning_tree(tree_t* tree, long long num_vertices, int num_nn, mst_edge_t* edges) {
    tree->num_vertices = num_vertices;
    qsort(edges, num_vertices * num_nn, sizeof(*edges), mst_dist_comparator);
    tree->vertices = malloc(sizeof(*tree->vertices) * num_vertices);
    for (tree_vertex_t* vertex_it = tree->vertices;
         vertex_it != tree->vertices + num_vertices;
         ++vertex_it)
    {
        vertex_it->num_edges = 0;
        vertex_it->edges = NULL;
    }

    dsu_t dsu;
    dsu_init(&dsu, num_vertices);

    tree->num_edges = 0;
    mst_edge_t* edges_it = edges;
    mst_edge_t* good_edges_it = edges;
    for (; tree->num_edges + 1 < num_vertices && edges_it != edges + num_vertices * num_nn;
           ++edges_it) {
        if (dsu_is_one_set(&dsu, edges_it->source, edges_it->target)) {
            continue;
        }

        ++tree->vertices[edges_it->source].num_edges;
        dsu_union(&dsu, edges_it->source, edges_it->target);
        swap_edges(edges_it, good_edges_it++);
        ++tree->num_edges;
    }

    assert(tree->num_edges * 2 <= num_vertices * num_nn && "Need at least two nn-vectors");
    edges_it = edges;
    mst_edge_t* good_edges_end = good_edges_it;
    for (; edges_it != good_edges_end; ++edges_it, ++good_edges_it) {
        good_edges_it->source = edges_it->target;
        good_edges_it->target = edges_it->source;
        good_edges_it->dist = edges_it->dist;
        ++tree->vertices[edges_it->target].num_edges;
    }

    tree->num_edges *= 2;
    qsort(edges, tree->num_edges, sizeof(*edges), mst_source_comparator);
    tree->edges = malloc(sizeof(*tree->edges) * tree->num_edges);
    tree_edge_t* tree_edge_it = tree->edges;
    edges_it = edges;
    long long edge_num = 0;
    vector_id_t expected_source = 0;
    long long assigned_vertices = 0;
    for (; edges_it != edges + tree->num_edges;
           ++edges_it, ++tree_edge_it, ++edge_num, ++assigned_vertices)
    {
        while (edge_num >= tree->vertices[expected_source].num_edges) {
            edge_num -= tree->vertices[expected_source].num_edges;
            ++expected_source;
        }
        // printf("%lld %d %d %d\n", edge_num, tree->vertices[expected_source].num_edges, edges_it->source, expected_source);
        assert(edges_it->source == expected_source);
        assert(edges_it->target < num_vertices);
        tree_edge_it->target = edges_it->target;
    }
    assert(assigned_vertices == tree->num_edges);

    restore_tree_edges_pointers(tree);
}

void tree_destroy(tree_t* tree) {
    free(tree->edges);
    tree->edges = NULL;
    free(tree->vertices);
    tree->vertices = NULL;
    tree->num_vertices = 0;
    tree->num_edges = 0;
}

void tree_save_filename(const tree_t* tree, const char* filename) {
    FILE* f = fopen(filename, "wb");
    tree_save_file(tree, f);
    fclose(f);
}

void tree_save_file(const tree_t* tree, FILE* file) {
    fwrite(&tree->num_vertices, sizeof(tree->num_vertices), 1, file);
    fwrite(&tree->num_edges, sizeof(tree->num_edges), 1, file);
    // FIXME: Fix this if tree_edge will have more fields
    fwrite(tree->edges, sizeof(*tree->edges), tree->num_edges, file);

    int* children_counts = malloc(sizeof(*children_counts) * tree->num_vertices);
    for (long long i = 0; i < tree->num_vertices; ++i) {
        children_counts[i] = tree->vertices[i].num_edges;
    }
    fwrite(children_counts, sizeof(*children_counts), tree->num_vertices, file);
    free(children_counts);
}

void tree_load_filename(tree_t* tree, const char* filename) {
    FILE* f = fopen(filename, "rb");
    tree_load_file(tree, f);
    fclose(f);
}

void tree_load_file(tree_t* tree, FILE* file) {
    fread(&tree->num_vertices, sizeof(tree->num_vertices), 1, file);
    fread(&tree->num_edges, sizeof(tree->num_edges), 1, file);
    tree->edges = malloc(sizeof(*tree->edges) * tree->num_edges);
    // FIXME:
    fread(tree->edges, sizeof(*tree->edges), tree->num_edges, file);

    tree->vertices = malloc(sizeof(*tree->vertices) * tree->num_vertices);
    int* children_counts = malloc(sizeof(*children_counts) * tree->num_vertices);
    fread(children_counts, sizeof(*children_counts), tree->num_vertices, file);
    for (long long i = 0; i < tree->num_vertices; ++i) {
        tree->vertices[i].num_edges = children_counts[i];
    }
    restore_tree_edges_pointers(tree);
    free(children_counts);
}

int tree_collect_vertices_dfs(const tree_t* tree, vector_id_t* vertices, int* num_children) {
    vector_id_t* stack = malloc(sizeof(*stack) * tree->num_vertices);
    char* visited = calloc(sizeof(*visited), tree->num_vertices);

    // NOTE: may be null's
    vector_id_t* vertices_it = vertices;
    int* num_children_it = num_children;
    long long num_processed_vertices = 0;
    vector_id_t last_known_root_vector = 0;
    int num_roots = 0;

    long long num_edges_sum = 0;
    long long stack_size = 0;
    while (num_processed_vertices < tree->num_vertices) {
        if (stack_size == 0) {
            while (last_known_root_vector < tree->num_vertices && visited[last_known_root_vector]) {
                ++last_known_root_vector;
            }
            assert(last_known_root_vector < tree->num_vertices);
            stack[stack_size++] = last_known_root_vector;
            visited[last_known_root_vector] += 1;
            ++num_roots;
        }
        vector_id_t current_vertex_id = stack[--stack_size];
        assert (visited[current_vertex_id] == 1);
        // printf("Vertex %d, stack size %d\n", current_vertex_id, stack_size);
        // if (visited[current_vertex_id]) {
        //     continue;
        // }
        // visited[current_vertex_id] = 1;
        const tree_vertex_t* current_vertex = tree->vertices + current_vertex_id;

        if (vertices_it) {
            *vertices_it = current_vertex_id;
            ++vertices_it;
        }
        if (num_children) {
#if 1
            *num_children_it = 0;
            for (const tree_edge_t* edge = current_vertex->edges;
                 edge != current_vertex->edges + current_vertex->num_edges;
                 ++edge)
            {
                if (!visited[edge->target]) {
                    ++*num_children_it;
                }
            }
            num_edges_sum += *num_children_it;
#else
            *num_children_it = current_vertex->num_edges;
#endif
            ++num_children_it;
        }
        ++num_processed_vertices;

        for (const tree_edge_t* edges_it = current_vertex->edges;
             edges_it != current_vertex->edges + current_vertex->num_edges;
             ++edges_it)
        {
            // printf("%lld \n", current_vertex_id);
            // printf("Edge %lld, vis=%d\n", edges_it - tree->edges < tree->num_edges, visited[edges_it->target]);
            if (!visited[edges_it->target]) {
                // printf("Target %d\n", edges_it->target);
                visited[edges_it->target] += 1;
                stack[stack_size++] = edges_it->target;
            }
        }
    }

    assert(num_edges_sum == tree->num_vertices - num_roots);

    free(visited);
    free(stack);
    return num_roots;
}

void tree_traverser_init(tree_traverser_t* traverser, long long num_vertices) {
    traverser->num_vertices = num_vertices;
    traverser->stack_size = 0;
    traverser->stack = malloc(sizeof(*traverser->stack) * num_vertices);
}

void tree_traverser_destroy(tree_traverser_t* traverser) {
    traverser->num_vertices = 0;
    traverser->stack_size = 0;
    free(traverser->stack);
    traverser->stack = NULL;
}

void tree_traverser_reset(tree_traverser_t* traverser) {
    traverser->stack_size = 0;
}

vector_id_t tree_traverser_get_active_parent(const tree_traverser_t* traverser) {
    if (traverser->stack_size > 0) {
        return traverser->stack[traverser->stack_size - 1].vertex_id;
    } else {
        return TRAVERSER_NO_PARENT_VECTOR;
    }
}

void tree_traverser_push_vertex(tree_traverser_t* traverser, vector_id_t vertex_id,
                                int num_children) {
    if (traverser->stack_size > 0) {
        --traverser->stack[traverser->stack_size - 1].num_children;
        if (!traverser->stack[traverser->stack_size - 1].num_children) {
            --traverser->stack_size;
        }
    }

    if (num_children) {
        traverser->stack[traverser->stack_size].vertex_id = vertex_id;
        traverser->stack[traverser->stack_size].num_children = num_children;
        ++traverser->stack_size;
    }
}

double* tree_collect_num_children_stats(long long num_vectors, const int* num_children,
                                        int* alphabet_size_out) {
    int alphabet_size = 0;
    for (const int* num_children_it = num_children;
         num_children_it != num_children + num_vectors;
         ++num_children_it)
    {
        if (alphabet_size < *num_children_it) {
            alphabet_size = *num_children_it;
        }
    }
    ++alphabet_size;

    printf("Max num children: %d\n", alphabet_size);
    double* stats = malloc(sizeof(*stats) * alphabet_size);
    for (double* stats_it = stats; stats_it != stats + alphabet_size; ++stats_it) {
        *stats_it = 0.0;
    }

    for (const int* num_children_it = num_children;
         num_children_it != num_children + num_vectors;
         ++num_children_it)
    {
        stats[*num_children_it] += 1.0;
    }

    // huffman_codebook_encode_init(codebook, alphabet_size, stats);
    // free(stats);

    if (alphabet_size_out) {
        *alphabet_size_out = alphabet_size;
    }
    return stats;
}

#define K_STAR (1 << 8)
int tree_collect_indices_stats(long long num_vectors, int pq_m, const byte_t* pq_indices,
                               const vector_id_t* vertices, const int* num_children,
                               double* stats) {
    for (double* stats_it = stats; stats_it != stats + pq_m * K_STAR * K_STAR; ++stats_it) {
        *stats_it = 0.0;
    }
    tree_traverser_t traverser;
    if (vertices) {
        tree_traverser_init(&traverser, num_vectors);
    }

    const vector_id_t* vertices_it = vertices;
    const int* num_children_it = num_children;
    int num_components = 0;
    for (long long vec_index = 0; vec_index < num_vectors; ++vec_index) {
        vector_id_t current_vector_id = vertices_it ? *vertices_it : vec_index;
        vector_id_t prev_vector_id = tree_traverser_get_active_parent(&traverser);
        if (prev_vector_id != TRAVERSER_NO_PARENT_VECTOR) {
            const byte_t* prev_vector = pq_indices + 1LL * prev_vector_id * pq_m;
            const byte_t* current_vector = pq_indices + 1LL * current_vector_id * pq_m;
            for (int part_index = 0; part_index < pq_m; ++part_index) {
                double* stats_part = stats + part_index * K_STAR * K_STAR;
                stats_part[prev_vector[part_index] * K_STAR + current_vector[part_index]] += 1.0;
            }
        } else {
            ++num_components;
        }

        int current_num_children = num_children_it ? *num_children_it : 1;
        tree_traverser_push_vertex(&traverser, current_vector_id, current_num_children);

        if (vertices_it) {
            ++vertices_it;
        }
        if (num_children_it) {
            ++num_children_it;
        }
    }

#if 0
    for (int part_index = 0; part_index < pq_m; ++part_index) {
        double* stats_part = stats + part_index * K_STAR * K_STAR;
        huffman_codebook_context_encode_init(codebooks + part_index, K_STAR, stats_part);
    }
#endif

    if (vertices) {
        tree_traverser_destroy(&traverser);
    }

    return num_components;
}

void tree_estimate_huffman_encoding(huffman_stats_t* indices_stats, huffman_stats_t* children_stats,
                                    const tree_t* tree, int pq_m, const byte_t* pq_indices) {
    // TODO: Merge with huffman_encoder
    vector_id_t* vertices = malloc(sizeof(*vertices) * tree->num_vertices);
    int* num_children = malloc(sizeof(*num_children) * tree->num_vertices);

    double* indices_symbol_stats = malloc(sizeof(*indices_symbol_stats) * K_STAR * K_STAR * pq_m);
    tree_collect_vertices_dfs(tree, vertices, num_children);
    int num_components = tree_collect_indices_stats(tree->num_vertices, pq_m, pq_indices, vertices,
                                                    num_children, indices_symbol_stats);

    huffman_codebook_t codebook;
    huffman_codebook_context_encode_init(&codebook, K_STAR, indices_symbol_stats);
    huffman_stats_init(indices_stats, tree->num_vertices, pq_m, K_STAR);
    indices_stats->num_roots = num_components;
    for (int part_index = 0; part_index < pq_m; ++part_index) {
        double estimated = huffman_estimate_size(&codebook, indices_symbol_stats + part_index * K_STAR * K_STAR);
        huffman_stats_push(indices_stats, part_index, estimated);
    }
    huffman_codebook_destroy(&codebook);
    int num_children_alphabet_size = 0;
    double* num_children_symbol_stats =
            tree_collect_num_children_stats(tree->num_vertices, num_children,
                                            &num_children_alphabet_size); // malloc inside
    huffman_codebook_encode_init(&codebook, num_children_alphabet_size, num_children_symbol_stats);
    double num_children_estimation = huffman_estimate_size(&codebook, num_children_symbol_stats);
    huffman_stats_init(children_stats, tree->num_vertices, 1, num_children_alphabet_size);
    huffman_stats_push(children_stats, 0, num_children_estimation);
    huffman_codebook_destroy(&codebook);

    free(num_children_symbol_stats);
    free(indices_symbol_stats);
    free(vertices);
    free(num_children);
}
