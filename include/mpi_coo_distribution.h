#ifndef MPI_COO_DISTRIBUTION_H
#define MPI_COO_DISTRIBUTION_H

#include <mpi.h>
#include "mtx_reader.h"

typedef struct {
    /* Global matrix dimensions, known by every rank after MPI_Bcast. */
    int rows;
    int cols;

    /* Total number of nonzeros in the full matrix. */
    int global_nnz;

    /* Number of COO entries owned by this rank. */
    int local_nnz;

    /* First owned row for block distribution; -1 for cyclic distribution. */
    int global_offset;

    /* Local COO triples for rows owned by this rank. Indices remain global. */
    int *row;
    int *col;
    double *data;
} LocalCOO_Matrix;

typedef enum {
    COO_ROW_PARTITION_CYCLIC = 0,
    COO_ROW_PARTITION_BLOCK
} COORowPartition;

#ifdef __cplusplus
extern "C" {
#endif

int coo_partition_owned_count(int length,
                              int rank,
                              int size,
                              COORowPartition partition);
int coo_partition_first_index(int length,
                              int rank,
                              int size,
                              COORowPartition partition);
int coo_partition_owner(int index,
                        int length,
                        int size,
                        COORowPartition partition);

int distribute_coo_entries(const COO_Matrix *global_matrix,
                           LocalCOO_Matrix *local_matrix,
                           int root,
                           MPI_Comm comm);
int distribute_coo_entries_partitioned(const COO_Matrix *global_matrix,
                                       LocalCOO_Matrix *local_matrix,
                                       COORowPartition partition,
                                       int root,
                                       MPI_Comm comm);
/*
 * Every rank reads a disjoint byte chunk of the Matrix Market file, then COO
 * triples are exchanged directly with their cyclic or block row owner.
 */
int read_distributed_coo_entries(const char *filename,
                                 LocalCOO_Matrix *local_matrix,
                                 COORowPartition partition,
                                 MPI_Comm comm);
/* Reconstruct a complete COO matrix on root without re-reading the file. */
int gather_coo_entries(const LocalCOO_Matrix *local_matrix,
                       COO_Matrix *global_matrix,
                       int root,
                       MPI_Comm comm);
void free_local_coo(LocalCOO_Matrix *mat);

#ifdef __cplusplus
}
#endif

#endif
