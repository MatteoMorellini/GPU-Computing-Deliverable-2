#include <stdio.h>
#include <stdlib.h>
#include "mpi_coo_distribution.h"

static void compute_displacements(int size, const int *counts, int *displs) {
    int offset = 0;

    for (int rank = 0; rank < size; rank++) {
        displs[rank] = offset;
        offset += counts[rank];
    }
}

static int build_cyclic_send_buffers(const COO_Matrix *global_matrix,
                                     int size,
                                     int **counts_out,
                                     int **displs_out,
                                     int **rows_out,
                                     int **cols_out,
                                     double **data_out) {
    int *counts = calloc((size_t)size, sizeof(int));
    int *displs = malloc((size_t)size * sizeof(int));
    int *next = NULL;
    int *rows = NULL;
    int *cols = NULL;
    double *data = NULL;

    if (!counts || !displs) {
        goto error;
    }

    for (int i = 0; i < global_matrix->nnz; i++) {
        int owner = global_matrix->row[i] % size;
        counts[owner]++;
    }

    compute_displacements(size, counts, displs);

    if (global_matrix->nnz > 0) {
        rows = malloc((size_t)global_matrix->nnz * sizeof(int));
        cols = malloc((size_t)global_matrix->nnz * sizeof(int));
        data = malloc((size_t)global_matrix->nnz * sizeof(double));
        next = malloc((size_t)size * sizeof(int));
        if (!rows || !cols || !data || !next) {
            goto error;
        }
    }

    if (next) {
        for (int rank = 0; rank < size; rank++) {
            next[rank] = displs[rank];
        }
    }

    for (int i = 0; i < global_matrix->nnz; i++) {
        int owner = global_matrix->row[i] % size;
        int dest = next[owner]++;

        rows[dest] = global_matrix->row[i];
        cols[dest] = global_matrix->col[i];
        data[dest] = global_matrix->data[i];
    }

    free(next);
    *counts_out = counts;
    *displs_out = displs;
    *rows_out = rows;
    *cols_out = cols;
    *data_out = data;
    return 0;

error:
    free(counts);
    free(displs);
    free(next);
    free(rows);
    free(cols);
    free(data);
    return 1;
}

static int allocate_local_entries(LocalCOO_Matrix *mat) {
    if (mat->local_nnz == 0) {
        mat->row = NULL;
        mat->col = NULL;
        mat->data = NULL;
        return 0;
    }

    mat->row = malloc((size_t)mat->local_nnz * sizeof(int));
    mat->col = malloc((size_t)mat->local_nnz * sizeof(int));
    mat->data = malloc((size_t)mat->local_nnz * sizeof(double));

    if (!mat->row || !mat->col || !mat->data) {
        free_local_coo(mat);
        return 1;
    }

    return 0;
}

int distribute_coo_entries(const COO_Matrix *global_matrix,
                           LocalCOO_Matrix *local_matrix,
                           int root,
                           MPI_Comm comm) {
    int rank;
    int size;
    int metadata[3] = {0, 0, 0};

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_error = 0;
    if (rank == root) {
        if (!global_matrix || !global_matrix->row || !global_matrix->col ||
            !global_matrix->data) {
            fprintf(stderr, "Root rank has no complete COO matrix to distribute\n");
            local_error = 1;
        } else {
            metadata[0] = global_matrix->rows;
            metadata[1] = global_matrix->cols;
            metadata[2] = global_matrix->nnz;
        }
    }

    /*
     * MPI collectives must be reached consistently by every rank.
     *
     * If one rank detects an error and returns while the others continue to
     * MPI_Bcast or MPI_Scatterv, the remaining ranks can hang forever waiting
     * for a process that left the collective sequence.
     *
     * MPI_Allreduce with MPI_MAX turns all local_error flags into one shared
     * any_error flag. If any rank reports 1, every rank sees any_error == 1
     * and returns together before entering the next collective.
     */
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        return 1;
    }

    /*
     * Rank 0 is the only process that reads the Matrix Market file.
     *
     * The other ranks still need the global shape and nnz:
     *   - rows and cols describe the global matrix that each local row-cyclic
     *     slice belongs to;
     *   - nnz is used to verify the final distributed entry count;
     *   - later distributed operations, such as SpMV, need the global
     *     dimensions to allocate input/output vectors correctly.
     *
     * Even if a rank receives only a small local slice, those entries still
     * belong to the full global matrix.
     */
    MPI_Bcast(metadata, 3, MPI_INT, root, comm);

    local_matrix->rows = metadata[0];
    local_matrix->cols = metadata[1];
    local_matrix->global_nnz = metadata[2];
    local_matrix->row = NULL;
    local_matrix->col = NULL;
    local_matrix->data = NULL;

    int *counts = NULL;
    int *displs = NULL;
    int *send_rows_buffer = NULL;
    int *send_cols_buffer = NULL;
    double *send_data_buffer = NULL;

    if (rank == root) {
        local_error = build_cyclic_send_buffers(global_matrix, size, &counts,
                                                &displs, &send_rows_buffer,
                                                &send_cols_buffer,
                                                &send_data_buffer);
    }

    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free(counts);
        free(displs);
        free(send_rows_buffer);
        free(send_cols_buffer);
        free(send_data_buffer);
        return 1;
    }

    MPI_Scatter(counts, 1, MPI_INT, &local_matrix->local_nnz, 1, MPI_INT,
                root, comm);
    local_matrix->global_offset = -1;

    // allocate this rank's local COO arrays
    local_error = allocate_local_entries(local_matrix);

    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free(counts);
        free(displs);
        free(send_rows_buffer);
        free(send_cols_buffer);
        free(send_data_buffer);
        return 1;
    }

    const int *send_rows = rank == root ? send_rows_buffer : NULL;
    const int *send_cols = rank == root ? send_cols_buffer : NULL;
    const double *send_data = rank == root ? send_data_buffer : NULL;

    /*
     * Use 1D modulo row ownership:
     *
     *   owner(row) = row % number_of_processes
     *
     * Rank 0 first groups COO triples by owner, preserving the original global
     * row index inside each triple. MPI_Scatterv then sends each rank all
     * nonzeros for its owned rows.
     */

    MPI_Scatterv(send_rows, counts, displs, MPI_INT,
                 local_matrix->row, local_matrix->local_nnz, MPI_INT,
                 root, comm);
    MPI_Scatterv(send_cols, counts, displs, MPI_INT,
                 local_matrix->col, local_matrix->local_nnz, MPI_INT,
                 root, comm);
    MPI_Scatterv(send_data, counts, displs, MPI_DOUBLE,
                 local_matrix->data, local_matrix->local_nnz, MPI_DOUBLE,
                 root, comm);

    free(counts);
    free(displs);
    free(send_rows_buffer);
    free(send_cols_buffer);
    free(send_data_buffer);
    return 0;
}

void free_local_coo(LocalCOO_Matrix *mat) {
    free(mat->row);
    free(mat->col);
    free(mat->data);
    mat->row = NULL;
    mat->col = NULL;
    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
    mat->global_nnz = 0;
    mat->local_nnz = 0;
    mat->global_offset = 0;
}
