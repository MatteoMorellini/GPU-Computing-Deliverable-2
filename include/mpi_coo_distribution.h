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

    /* Unused for cyclic row distribution. */
    int global_offset;

    /* Local COO triples for rows where row % number_of_processes == rank. */
    int *row;
    int *col;
    double *data;
} LocalCOO_Matrix;

#ifdef __cplusplus
extern "C" {
#endif

int distribute_coo_entries(const COO_Matrix *global_matrix,
                           LocalCOO_Matrix *local_matrix,
                           int root,
                           MPI_Comm comm);
void free_local_coo(LocalCOO_Matrix *mat);

#ifdef __cplusplus
}
#endif

#endif
