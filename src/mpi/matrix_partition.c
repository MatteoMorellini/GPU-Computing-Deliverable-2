#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metis.h>

#include "matrix_partition.h"

typedef struct {
    int row;
    int col;
} Edge;

typedef struct {
    int vertex;
    int degree;
} VertexOrder;

static int compare_edges(const void *left, const void *right) {
    const Edge *a = (const Edge *)left;
    const Edge *b = (const Edge *)right;
    if (a->row != b->row) return a->row < b->row ? -1 : 1;
    if (a->col != b->col) return a->col < b->col ? -1 : 1;
    return 0;
}

static int compare_vertex_order(const void *left, const void *right) {
    const VertexOrder *a = (const VertexOrder *)left;
    const VertexOrder *b = (const VertexOrder *)right;
    if (a->degree != b->degree) return a->degree > b->degree ? -1 : 1;
    if (a->vertex != b->vertex) return a->vertex < b->vertex ? -1 : 1;
    return 0;
}

static unsigned long long splitmix64(unsigned long long value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

const char *matrix_partition_mode_name(MatrixPartitionMode mode) {
    switch (mode) {
        case MATRIX_PARTITION_REPLICATED: return "replicated";
        case MATRIX_PARTITION_1D_BLOCK: return "1d-block";
        case MATRIX_PARTITION_1D_RANDOM: return "1d-random";
        case MATRIX_PARTITION_1D_GP: return "1d-gp";
        case MATRIX_PARTITION_1D_HP: return "1d-hp";
        case MATRIX_PARTITION_1D_LRA: return "1d-lra";
        case MATRIX_PARTITION_2D_BLOCK: return "2d-block";
        case MATRIX_PARTITION_2D_RANDOM: return "2d-random";
        case MATRIX_PARTITION_2D_GP: return "2d-gp";
        case MATRIX_PARTITION_2D_HP: return "2d-hp";
        default: return "cyclic";
    }
}

int parse_matrix_partition_mode(const char *name, MatrixPartitionMode *mode) {
    if (!strcmp(name, "cyclic") || !strcmp(name, "distributed") ||
        !strcmp(name, "dist") || !strcmp(name, "1d-cyclic")) {
        *mode = MATRIX_PARTITION_CYCLIC;
    } else if (!strcmp(name, "replicated") || !strcmp(name, "repl")) {
        *mode = MATRIX_PARTITION_REPLICATED;
    } else if (!strcmp(name, "block") || !strcmp(name, "1d-block")) {
        *mode = MATRIX_PARTITION_1D_BLOCK;
    } else if (!strcmp(name, "random") || !strcmp(name, "1d-random")) {
        *mode = MATRIX_PARTITION_1D_RANDOM;
    } else if (!strcmp(name, "gp") || !strcmp(name, "1d-gp")) {
        *mode = MATRIX_PARTITION_1D_GP;
    } else if (!strcmp(name, "hp") || !strcmp(name, "1d-hp")) {
        *mode = MATRIX_PARTITION_1D_HP;
    } else if (!strcmp(name, "lra") || !strcmp(name, "1d-lra") ||
               !strcmp(name, "long-row-aware")) {
        *mode = MATRIX_PARTITION_1D_LRA;
    } else if (!strcmp(name, "2d-block")) {
        *mode = MATRIX_PARTITION_2D_BLOCK;
    } else if (!strcmp(name, "2d-random")) {
        *mode = MATRIX_PARTITION_2D_RANDOM;
    } else if (!strcmp(name, "2d-gp")) {
        *mode = MATRIX_PARTITION_2D_GP;
    } else if (!strcmp(name, "2d-hp")) {
        *mode = MATRIX_PARTITION_2D_HP;
    } else {
        return 1;
    }
    return 0;
}

int matrix_partition_is_2d(MatrixPartitionMode mode) {
    return mode >= MATRIX_PARTITION_2D_BLOCK;
}

int matrix_partition_is_gp_or_hp(MatrixPartitionMode mode) {
    return mode == MATRIX_PARTITION_1D_GP || mode == MATRIX_PARTITION_1D_HP ||
           mode == MATRIX_PARTITION_2D_GP || mode == MATRIX_PARTITION_2D_HP;
}

int matrix_partition_uses_explicit_map(MatrixPartitionMode mode) {
    return matrix_partition_is_gp_or_hp(mode) ||
           mode == MATRIX_PARTITION_1D_LRA;
}

int matrix_partition_uses_distributed_vector(MatrixPartitionMode mode) {
    return mode != MATRIX_PARTITION_REPLICATED;
}

int matrix_partition_choose_grid(int processes,
                                 int requested_rows,
                                 int requested_cols,
                                 int *process_rows,
                                 int *process_cols) {
    if (processes <= 0) return 1;
    if (requested_rows > 0 || requested_cols > 0) {
        if (requested_rows <= 0 || requested_cols <= 0 ||
            requested_rows > INT_MAX / requested_cols ||
            requested_rows * requested_cols != processes) {
            return 1;
        }
        *process_rows = requested_rows;
        *process_cols = requested_cols;
        return 0;
    }

    int rows = (int)sqrt((double)processes);
    while (rows > 1 && processes % rows != 0) rows--;
    *process_rows = rows;
    *process_cols = processes / rows;
    return 0;
}

static int balanced_block_owner(int vertex, int vertices, int processes) {
    const int base = vertices / processes;
    const int remainder = vertices % processes;
    const int long_vertices = (base + 1) * remainder;
    if (vertex < long_vertices) return vertex / (base + 1);
    return remainder + (vertex - long_vertices) / base;
}

static int formula_owner(MatrixPartitionMode mode,
                         int vertex,
                         int vertices,
                         int processes,
                         unsigned long long seed) {
    if (mode == MATRIX_PARTITION_CYCLIC ||
        mode == MATRIX_PARTITION_REPLICATED) {
        return vertex % processes;
    }
    if (mode == MATRIX_PARTITION_1D_BLOCK ||
        mode == MATRIX_PARTITION_2D_BLOCK) {
        return balanced_block_owner(vertex, vertices, processes);
    }
    return (int)(splitmix64((unsigned long long)vertex ^ seed) %
                 (unsigned long long)processes);
}

static int read_partition_file(const char *path, int vertices, int parts,
                               int *assignment) {
    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Cannot open partition file %s\n", path);
        return 1;
    }

    char line[256];
    int next_vertex = 0;
    while (fgets(line, sizeof(line), file)) {
        char *cursor = line;
        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (*cursor == '\0' || *cursor == '\n' || *cursor == '#') continue;

        int vertex = -1;
        int part = -1;
        const int fields = sscanf(cursor, "%d %d", &vertex, &part);
        if (fields == 1) {
            part = vertex;
            vertex = next_vertex;
        } else if (fields != 2) {
            fclose(file);
            return 1;
        }
        if (vertex != next_vertex || part < 0 || part >= parts) {
            fprintf(stderr,
                    "Invalid partition entry at vertex %d (part %d) in %s\n",
                    vertex, part, path);
            fclose(file);
            return 1;
        }
        assignment[next_vertex++] = part;
    }
    fclose(file);
    if (next_vertex != vertices) {
        fprintf(stderr, "Partition file %s has %d entries; expected %d\n",
                path, next_vertex, vertices);
        return 1;
    }
    return 0;
}

static int build_graph_partition(const COO_Matrix *matrix,
                                 int parts,
                                 int *assignment) {
    const int vertices = matrix->rows;
    size_t capacity = (size_t)matrix->nnz * 2;
    if (capacity > (size_t)INT_MAX) {
        fprintf(stderr, "METIS adjacency exceeds the 32-bit build limit\n");
        return 1;
    }
    Edge *edges = capacity ? (Edge *)malloc(capacity * sizeof(Edge)) : NULL;
    if (capacity && !edges) return 1;

    int count = 0;
    for (int i = 0; i < matrix->nnz; i++) {
        const int row = matrix->row[i];
        const int col = matrix->col[i];
        if (row == col) continue;
        edges[count++] = (Edge){row, col};
        edges[count++] = (Edge){col, row};
    }
    qsort(edges, (size_t)count, sizeof(Edge), compare_edges);
    int unique = 0;
    for (int i = 0; i < count; i++) {
        if (unique == 0 || compare_edges(&edges[i], &edges[unique - 1])) {
            edges[unique++] = edges[i];
        }
    }

    idx_t *xadj = (idx_t *)calloc((size_t)vertices + 1, sizeof(idx_t));
    idx_t *adjncy = unique ? (idx_t *)malloc((size_t)unique * sizeof(idx_t)) : NULL;
    idx_t *vwgt = vertices ? (idx_t *)calloc((size_t)vertices, sizeof(idx_t)) : NULL;
    idx_t *part = vertices ? (idx_t *)malloc((size_t)vertices * sizeof(idx_t)) : NULL;
    if (!xadj || (unique && !adjncy) || (vertices && (!vwgt || !part))) {
        free(edges); free(xadj); free(adjncy); free(vwgt); free(part);
        return 1;
    }
    for (int i = 0; i < unique; i++) xadj[edges[i].row + 1]++;
    for (int v = 0; v < vertices; v++) xadj[v + 1] += xadj[v];
    for (int i = 0; i < unique; i++) adjncy[i] = (idx_t)edges[i].col;
    for (int i = 0; i < matrix->nnz; i++) vwgt[matrix->row[i]]++;
    for (int v = 0; v < vertices; v++) if (vwgt[v] == 0) vwgt[v] = 1;

    idx_t nvtxs = (idx_t)vertices;
    idx_t ncon = 1;
    idx_t nparts = (idx_t)parts;
    idx_t edgecut = 0;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    const int metis_status = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj, adjncy, vwgt, NULL, NULL, &nparts,
        NULL, NULL, options, &edgecut, part);
    if (metis_status == METIS_OK) {
        for (int v = 0; v < vertices; v++) assignment[v] = (int)part[v];
    }
    free(edges); free(xadj); free(adjncy); free(vwgt); free(part);
    if (metis_status != METIS_OK) {
        fprintf(stderr, "METIS_PartGraphKway failed with status %d\n",
                metis_status);
        return 1;
    }
    return 0;
}

/*
 * Deterministic row-net hypergraph fallback. Vertices are matrix rows and
 * columns are nets. Heavy rows are placed first; an eligible part receives a
 * row when it already owns the largest number of that row's claimed nets.
 * The 3% capacity slack balances nonzeros rather than only vertex count.
 */
static int build_hypergraph_partition(const COO_Matrix *matrix,
                                      int parts,
                                      int *assignment) {
    const int vertices = matrix->rows;
    int *degree = (int *)calloc((size_t)vertices, sizeof(int));
    VertexOrder *order = vertices
                             ? (VertexOrder *)malloc((size_t)vertices * sizeof(VertexOrder))
                             : NULL;
    long long *load = (long long *)calloc((size_t)parts, sizeof(long long));
    int *net_part = matrix->cols ? (int *)malloc((size_t)matrix->cols * sizeof(int)) : NULL;
    int *row_ptr = (int *)calloc((size_t)vertices + 1, sizeof(int));
    int *row_cols = matrix->nnz ? (int *)malloc((size_t)matrix->nnz * sizeof(int)) : NULL;
    if ((vertices && (!degree || !order)) || !load ||
        (matrix->cols && !net_part) || !row_ptr ||
        (matrix->nnz && !row_cols)) {
        free(degree); free(order); free(load); free(net_part);
        free(row_ptr); free(row_cols);
        return 1;
    }
    for (int c = 0; c < matrix->cols; c++) net_part[c] = -1;
    for (int i = 0; i < matrix->nnz; i++) {
        degree[matrix->row[i]]++;
        row_ptr[matrix->row[i] + 1]++;
    }
    for (int v = 0; v < vertices; v++) row_ptr[v + 1] += row_ptr[v];
    int *next = vertices ? (int *)malloc((size_t)vertices * sizeof(int)) : NULL;
    if (vertices && !next) {
        free(degree); free(order); free(load); free(net_part);
        free(row_ptr); free(row_cols);
        return 1;
    }
    if (vertices) memcpy(next, row_ptr, (size_t)vertices * sizeof(int));
    for (int i = 0; i < matrix->nnz; i++)
        row_cols[next[matrix->row[i]]++] = matrix->col[i];
    free(next);
    for (int v = 0; v < vertices; v++)
        order[v] = (VertexOrder){v, degree[v]};
    qsort(order, (size_t)vertices, sizeof(VertexOrder), compare_vertex_order);
    const long long total_weight = matrix->nnz + vertices;
    const long long capacity =
        (long long)ceil(1.03 * (double)total_weight / (double)parts);
    int *affinity = (int *)calloc((size_t)parts, sizeof(int));
    if (!affinity) {
        free(degree); free(order); free(load); free(net_part);
        free(row_ptr); free(row_cols);
        return 1;
    }
    for (int oi = 0; oi < vertices; oi++) {
        const int vertex = order[oi].vertex;
        memset(affinity, 0, (size_t)parts * sizeof(int));
        for (int j = row_ptr[vertex]; j < row_ptr[vertex + 1]; j++) {
            const int claimed = net_part[row_cols[j]];
            if (claimed >= 0) affinity[claimed]++;
        }
        const long long weight = (long long)degree[vertex] + 1;
        int best = -1;
        for (int part = 0; part < parts; part++) {
            if (best < 0 ||
                ((load[part] + weight <= capacity) &&
                 (load[best] + weight > capacity)) ||
                ((load[part] + weight <= capacity) ==
                     (load[best] + weight <= capacity) &&
                 (affinity[part] > affinity[best] ||
                  (affinity[part] == affinity[best] && load[part] < load[best])))) {
                best = part;
            }
        }
        assignment[vertex] = best;
        load[best] += weight;
        for (int j = row_ptr[vertex]; j < row_ptr[vertex + 1]; j++) {
            if (net_part[row_cols[j]] < 0) net_part[row_cols[j]] = best;
        }
    }
    free(degree); free(order); free(load); free(net_part);
    free(row_ptr); free(row_cols); free(affinity);
    return 0;
}

/*
 * Split one consecutive row range into consecutive, approximately NNZ-equal
 * sub-blocks. Boundaries are placed at the row prefix closest to each ideal
 * cumulative-NNZ target. Empty sub-blocks are allowed when there are fewer
 * rows than parts or one indivisible row dominates the range.
 */
static void assign_nnz_balanced_range(const int *row_nnz,
                                      int begin,
                                      int end,
                                      int parts,
                                      int *assignment) {
    long long total = 0;
    for (int row = begin; row < end; row++) total += row_nnz[row];

    if (total == 0) {
        const int rows = end - begin;
        for (int row = begin; row < end; row++) {
            assignment[row] =
                balanced_block_owner(row - begin, rows, parts);
        }
        return;
    }

    int previous = begin;
    int cursor = begin;
    long long cumulative = 0;
    for (int part = 0; part < parts - 1; part++) {
        const long long target =
            (long long)(((long double)total * (part + 1)) / parts);
        while (cursor < end &&
               cumulative + (long long)row_nnz[cursor] < target) {
            cumulative += row_nnz[cursor++];
        }
        if (cursor < end && cumulative <= target) {
            const long long before_distance = target - cumulative;
            const long long after =
                cumulative + (long long)row_nnz[cursor];
            const long long after_distance = after - target;
            if (after_distance <= before_distance) {
                cumulative = after;
                cursor++;
            }
        }
        for (int row = previous; row < cursor; row++)
            assignment[row] = part;
        previous = cursor;
    }
    for (int row = previous; row < end; row++)
        assignment[row] = parts - 1;
}

/*
 * Gao, Ji, and Wang, "Optimization of Large-Scale Sparse Matrix-Vector
 * Multiplication on Multi-GPU Systems", ACM TACO 21(4), 2024, Section 3.3.
 * https://doi.org/10.1145/3676847
 *
 * The longest row selects whether the floor(alpha*n)-row long-row block is
 * taken from the beginning or the end of the matrix. For tiny nonempty
 * matrices the block is clamped to at least one row. The long- and short-row
 * regions are then independently partitioned by cumulative NNZ so every rank
 * receives one consecutive sub-block from each region.
 */
static int build_long_row_aware_partition(const COO_Matrix *matrix,
                                          int parts,
                                          double long_row_fraction,
                                          int *assignment) {
    const int rows = matrix->rows;
    if (!(long_row_fraction > 0.0 && long_row_fraction < 1.0)) {
        fprintf(stderr,
                "Long-row fraction must be strictly between 0 and 1\n");
        return 1;
    }
    if (rows == 0) return 0;

    int *row_nnz = (int *)calloc((size_t)rows, sizeof(int));
    if (!row_nnz) return 1;
    for (int entry = 0; entry < matrix->nnz; entry++) {
        const int row = matrix->row[entry];
        if (row < 0 || row >= rows || row_nnz[row] == INT_MAX) {
            free(row_nnz);
            return 1;
        }
        row_nnz[row]++;
    }

    int longest_row = 0;
    for (int row = 1; row < rows; row++) {
        if (row_nnz[row] > row_nnz[longest_row]) longest_row = row;
    }

    int long_rows = (int)floor(long_row_fraction * (double)rows);
    if (long_rows < 1) long_rows = 1;
    if (rows > 1 && long_rows >= rows) long_rows = rows - 1;

    if (longest_row < long_rows) {
        assign_nnz_balanced_range(row_nnz, 0, long_rows, parts,
                                  assignment);
        assign_nnz_balanced_range(row_nnz, long_rows, rows, parts,
                                  assignment);
    } else {
        const int long_begin = rows - long_rows;
        assign_nnz_balanced_range(row_nnz, 0, long_begin, parts,
                                  assignment);
        assign_nnz_balanced_range(row_nnz, long_begin, rows, parts,
                                  assignment);
    }
    free(row_nnz);
    return 0;
}

int matrix_partition_prepare_with_long_row_fraction(
    MatrixPartition *partition,
    MatrixPartitionMode mode,
    int vertices,
    int processes,
    int requested_rows,
    int requested_cols,
    unsigned long long seed,
    double long_row_fraction,
    const char *partition_file,
    const COO_Matrix *root_matrix,
    int root,
    MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    memset(partition, 0, sizeof(*partition));
    if (vertices < 0 || processes <= 0) return 1;
    partition->mode = mode;
    partition->vertices = vertices;
    partition->processes = processes;
    partition->seed = seed;
    partition->long_row_fraction = long_row_fraction;
    if (matrix_partition_choose_grid(processes, requested_rows, requested_cols,
                                     &partition->process_rows,
                                     &partition->process_cols)) return 1;
    if (!matrix_partition_uses_explicit_map(mode)) {
        if (mode == MATRIX_PARTITION_1D_RANDOM ||
            mode == MATRIX_PARTITION_2D_RANDOM) {
            partition->local_indices =
                vertices ? (int *)malloc((size_t)vertices * sizeof(int)) : NULL;
            int local_error = vertices > 0 && !partition->local_indices;
            int any_error = 0;
            MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
            if (any_error) {
                free_matrix_partition(partition);
                return 1;
            }
            int *counts = (int *)calloc((size_t)processes, sizeof(int));
            local_error = !counts;
            MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
            if (any_error) {
                free(counts);
                free_matrix_partition(partition);
                return 1;
            }
            for (int vertex = 0; vertex < vertices; vertex++) {
                const int owner = formula_owner(mode, vertex, vertices,
                                                processes, seed);
                partition->local_indices[vertex] = counts[owner]++;
            }
            free(counts);
        }
        return 0;
    }

    partition->parts = vertices ? (int *)malloc((size_t)vertices * sizeof(int)) : NULL;
    int local_error = vertices > 0 && !partition->parts;
    if (rank == root && !local_error) {
        if (processes == 1) {
            for (int vertex = 0; vertex < vertices; vertex++)
                partition->parts[vertex] = 0;
        } else if (partition_file) {
            local_error = read_partition_file(partition_file, vertices,
                                              processes, partition->parts);
        } else if (!root_matrix || root_matrix->rows != vertices ||
                   root_matrix->cols != vertices) {
            fprintf(stderr,
                    "GP/HP/LRA partitioning requires a square root matrix\n");
            local_error = 1;
        } else if (mode == MATRIX_PARTITION_1D_GP ||
                   mode == MATRIX_PARTITION_2D_GP) {
            local_error = build_graph_partition(root_matrix, processes,
                                                partition->parts);
        } else if (mode == MATRIX_PARTITION_1D_HP ||
                   mode == MATRIX_PARTITION_2D_HP) {
            local_error = build_hypergraph_partition(root_matrix, processes,
                                                     partition->parts);
        } else {
            local_error = build_long_row_aware_partition(
                root_matrix, processes, long_row_fraction,
                partition->parts);
        }
    }
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free_matrix_partition(partition);
        return 1;
    }
    MPI_Bcast(partition->parts, vertices, MPI_INT, root, comm);
    partition->local_indices =
        vertices ? (int *)malloc((size_t)vertices * sizeof(int)) : NULL;
    local_error = vertices > 0 && !partition->local_indices;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free_matrix_partition(partition);
        return 1;
    }
    int *counts = (int *)calloc((size_t)processes, sizeof(int));
    local_error = !counts;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free(counts);
        free_matrix_partition(partition);
        return 1;
    }
    for (int vertex = 0; vertex < vertices; vertex++) {
        const int owner = partition->parts[vertex];
        partition->local_indices[vertex] = counts[owner]++;
    }
    free(counts);
    return 0;
}

int matrix_partition_prepare(MatrixPartition *partition,
                             MatrixPartitionMode mode,
                             int vertices,
                             int processes,
                             int requested_rows,
                             int requested_cols,
                             unsigned long long seed,
                             const char *partition_file,
                             const COO_Matrix *root_matrix,
                             int root,
                             MPI_Comm comm) {
    return matrix_partition_prepare_with_long_row_fraction(
        partition, mode, vertices, processes, requested_rows,
        requested_cols, seed,
        MATRIX_PARTITION_DEFAULT_LONG_ROW_FRACTION, partition_file,
        root_matrix, root, comm);
}

int matrix_partition_vertex_owner(const MatrixPartition *partition,
                                  int vertex) {
    if (!partition || vertex < 0 || vertex >= partition->vertices) return -1;
    if (partition->parts) return partition->parts[vertex];
    return formula_owner(partition->mode, vertex, partition->vertices,
                         partition->processes, partition->seed);
}

int matrix_partition_entry_owner(const MatrixPartition *partition,
                                 int row,
                                 int col) {
    const int row_part = matrix_partition_vertex_owner(partition, row);
    if (!matrix_partition_is_2d(partition->mode)) return row_part;
    const int col_part = matrix_partition_vertex_owner(partition, col);
    const int process_row = row_part % partition->process_rows;
    const int process_col = col_part / partition->process_rows;
    return process_row + process_col * partition->process_rows;
}

int matrix_partition_owned_count(const MatrixPartition *partition, int rank) {
    int count = 0;
    for (int vertex = 0; vertex < partition->vertices; vertex++)
        count += matrix_partition_vertex_owner(partition, vertex) == rank;
    return count;
}

int matrix_partition_local_index(const MatrixPartition *partition,
                                 int vertex) {
    if (partition->local_indices) return partition->local_indices[vertex];
    const int owner = matrix_partition_vertex_owner(partition, vertex);
    if (partition->mode == MATRIX_PARTITION_CYCLIC ||
        partition->mode == MATRIX_PARTITION_REPLICATED) {
        return vertex / partition->processes;
    }
    const int base = partition->vertices / partition->processes;
    const int remainder = partition->vertices % partition->processes;
    const int first = owner * base + (owner < remainder ? owner : remainder);
    return vertex - first;
}

void matrix_partition_build_counts(const MatrixPartition *partition,
                                   int *counts,
                                   int *displs) {
    memset(counts, 0, (size_t)partition->processes * sizeof(int));
    for (int vertex = 0; vertex < partition->vertices; vertex++)
        counts[matrix_partition_vertex_owner(partition, vertex)]++;
    int offset = 0;
    for (int rank = 0; rank < partition->processes; rank++) {
        displs[rank] = offset;
        offset += counts[rank];
    }
}

void free_matrix_partition(MatrixPartition *partition) {
    if (!partition) return;
    free(partition->parts);
    free(partition->local_indices);
    memset(partition, 0, sizeof(*partition));
}
