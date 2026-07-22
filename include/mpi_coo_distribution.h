#ifndef MPI_COO_DISTRIBUTION_H
#define MPI_COO_DISTRIBUTION_H

#include <mpi.h>
#include "matrix_partition.h"
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

typedef struct {
    /* Complete path until every rank owns its local COO rows. */
    double total_s;

    /* Raw reader time before entry-count validation and row redistribution. */
    double read_parse_s;

    /* MPI-IO-specific split; zero for readers that cannot separate the two. */
    double file_io_s;
    double parse_s;

    /* Collective entry-count checks before redistribution. */
    double validation_s;

    /* Complete redistribution and its local-preparation/MPI-exchange split. */
    double redistribution_s;
    double pack_s;
    double exchange_s;

    /* Expanded COO entries initially parsed and sent to another rank. */
    long long source_nnz;
    long long remote_nnz;
} MatrixInputMetrics;

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
/*
 * Reduce timings to root while keeping subcomponents from the rank that had
 * the slowest enclosing phase. Entry traffic is summed across ranks.
 */
void reduce_matrix_input_metrics(const MatrixInputMetrics *local,
                                 MatrixInputMetrics *global,
                                 int root,
                                 MPI_Comm comm);

int distribute_coo_entries(const COO_Matrix *global_matrix,
                           LocalCOO_Matrix *local_matrix,
                           int root,
                           MPI_Comm comm);
int distribute_coo_entries_partitioned(const COO_Matrix *global_matrix,
                                       LocalCOO_Matrix *local_matrix,
                                       COORowPartition partition,
                                       int root,
                                       MPI_Comm comm);
int distribute_coo_entries_partitioned_timed(
    const COO_Matrix *global_matrix,
    LocalCOO_Matrix *local_matrix,
    COORowPartition partition,
    int root,
    MatrixInputMetrics *metrics,
    MPI_Comm comm);
int distribute_coo_entries_layout_timed(
    const COO_Matrix *global_matrix,
    LocalCOO_Matrix *local_matrix,
    const MatrixPartition *partition,
    int root,
    MatrixInputMetrics *metrics,
    MPI_Comm comm);
/*
 * Every rank reads a disjoint byte chunk of the Matrix Market file, then COO
 * triples are exchanged directly with their cyclic or block row owner.
 */
int read_distributed_coo_entries(const char *filename,
                                 LocalCOO_Matrix *local_matrix,
                                 COORowPartition partition,
                                 MPI_Comm comm);
int read_distributed_coo_entries_timed(const char *filename,
                                       LocalCOO_Matrix *local_matrix,
                                       COORowPartition partition,
                                       MatrixInputMetrics *metrics,
                                       MPI_Comm comm);
int read_distributed_coo_entries_layout_timed(
    const char *filename,
    LocalCOO_Matrix *local_matrix,
    const MatrixPartition *partition,
    MatrixInputMetrics *metrics,
    MPI_Comm comm);
/*
 * Collectively read line-aligned byte chunks with MPI_File_read_at_all, then
 * exchange COO triples directly with their cyclic or block row owner.
 */
int read_mpi_io_coo_entries(const char *filename,
                            LocalCOO_Matrix *local_matrix,
                            COORowPartition partition,
                            MPI_Comm comm);
int read_mpi_io_coo_entries_timed(const char *filename,
                                  LocalCOO_Matrix *local_matrix,
                                  COORowPartition partition,
                                  MatrixInputMetrics *metrics,
                                  MPI_Comm comm);
int read_mpi_io_coo_entries_layout_timed(
    const char *filename,
    LocalCOO_Matrix *local_matrix,
    const MatrixPartition *partition,
    MatrixInputMetrics *metrics,
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
