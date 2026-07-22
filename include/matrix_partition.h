#ifndef MATRIX_PARTITION_H
#define MATRIX_PARTITION_H

#include <mpi.h>

#include "mtx_reader.h"

typedef enum {
    MATRIX_PARTITION_CYCLIC = 0,
    MATRIX_PARTITION_REPLICATED,
    MATRIX_PARTITION_1D_BLOCK,
    MATRIX_PARTITION_1D_RANDOM,
    MATRIX_PARTITION_1D_GP,
    MATRIX_PARTITION_1D_HP,
    MATRIX_PARTITION_2D_BLOCK,
    MATRIX_PARTITION_2D_RANDOM,
    MATRIX_PARTITION_2D_GP,
    MATRIX_PARTITION_2D_HP
} MatrixPartitionMode;

typedef struct {
    MatrixPartitionMode mode;
    int vertices;
    int processes;
    int process_rows;
    int process_cols;
    unsigned long long seed;

    /* Present for GP/HP modes. Every rank receives the same vertex map. */
    int *parts;
    /* O(1) global-to-owner-local lookup for random and GP/HP layouts. */
    int *local_indices;
} MatrixPartition;

#ifdef __cplusplus
extern "C" {
#endif

const char *matrix_partition_mode_name(MatrixPartitionMode mode);
int parse_matrix_partition_mode(const char *name, MatrixPartitionMode *mode);
int matrix_partition_is_2d(MatrixPartitionMode mode);
int matrix_partition_is_gp_or_hp(MatrixPartitionMode mode);
int matrix_partition_uses_distributed_vector(MatrixPartitionMode mode);

int matrix_partition_choose_grid(int processes,
                                 int requested_rows,
                                 int requested_cols,
                                 int *process_rows,
                                 int *process_cols);

/* Build/broadcast GP or HP rpart. Other modes require no stored map. */
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
                             MPI_Comm comm);

int matrix_partition_vertex_owner(const MatrixPartition *partition,
                                  int vertex);
int matrix_partition_entry_owner(const MatrixPartition *partition,
                                 int row,
                                 int col);
int matrix_partition_owned_count(const MatrixPartition *partition, int rank);
int matrix_partition_local_index(const MatrixPartition *partition,
                                 int vertex);
void matrix_partition_build_counts(const MatrixPartition *partition,
                                   int *counts,
                                   int *displs);
void free_matrix_partition(MatrixPartition *partition);

#ifdef __cplusplus
}
#endif

#endif
