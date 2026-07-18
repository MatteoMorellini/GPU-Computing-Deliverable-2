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

    /* First global COO entry index owned by this rank. */
    int global_offset;

    /* Local slices of the global COO row, column, and value arrays. */
    int *row;
    int *col;
    double *data;
} LocalCOO_Matrix;

int distribute_coo_entries(const COO_Matrix *global_matrix,
                           LocalCOO_Matrix *local_matrix,
                           int root,
                           MPI_Comm comm);
void free_local_coo(LocalCOO_Matrix *mat);

#endif
