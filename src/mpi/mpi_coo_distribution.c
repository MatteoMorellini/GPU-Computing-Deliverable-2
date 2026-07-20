#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi_coo_distribution.h"

static int allocate_local_entries(LocalCOO_Matrix *mat);

int coo_partition_owned_count(int length,
                              int rank,
                              int size,
                              COORowPartition partition) {
    if (length <= 0 || rank < 0 || size <= 0 || rank >= size) {
        return 0;
    }
    if (partition == COO_ROW_PARTITION_BLOCK) {
        return length / size + (rank < length % size);
    }
    if (rank >= length) {
        return 0;
    }
    return ((length - 1 - rank) / size) + 1;
}

int coo_partition_first_index(int length,
                              int rank,
                              int size,
                              COORowPartition partition) {
    if (length < 0 || rank < 0 || size <= 0 || rank >= size) {
        return -1;
    }
    if (partition == COO_ROW_PARTITION_BLOCK) {
        const int base = length / size;
        const int remainder = length % size;
        return rank * base + (rank < remainder ? rank : remainder);
    }
    return rank;
}

int coo_partition_owner(int index,
                        int length,
                        int size,
                        COORowPartition partition) {
    if (index < 0 || index >= length || size <= 0) {
        return -1;
    }
    if (partition == COO_ROW_PARTITION_CYCLIC) {
        return index % size;
    }

    const int base = length / size;
    const int remainder = length % size;
    const int long_rows = (base + 1) * remainder;
    if (index < long_rows) {
        return index / (base + 1);
    }
    /* base is zero only when every valid index is in the first branch. */
    return remainder + (index - long_rows) / base;
}

static void compute_displacements(int size, const int *counts, int *displs) {
    int offset = 0;

    for (int rank = 0; rank < size; rank++) {
        displs[rank] = offset;
        offset += counts[rank];
    }
}

static int redistribute_chunk_by_row(const COO_Matrix *chunk,
                                     int global_nnz,
                                     LocalCOO_Matrix *local_matrix,
                                     COORowPartition partition,
                                     MPI_Comm comm) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int *send_counts = (int *)calloc((size_t)size, sizeof(int));
    int *send_displs = (int *)malloc((size_t)size * sizeof(int));
    int *recv_counts = (int *)malloc((size_t)size * sizeof(int));
    int *recv_displs = (int *)malloc((size_t)size * sizeof(int));
    int *next = (int *)malloc((size_t)size * sizeof(int));
    int *send_rows = NULL;
    int *send_cols = NULL;
    double *send_data = NULL;
    int local_error =
        !send_counts || !send_displs || !recv_counts || !recv_displs || !next;

    if (!local_error) {
        for (int i = 0; i < chunk->nnz; i++) {
            const int owner = coo_partition_owner(
                chunk->row[i], chunk->rows, size, partition);
            send_counts[owner]++;
        }
        compute_displacements(size, send_counts, send_displs);
        for (int r = 0; r < size; r++) {
            next[r] = send_displs[r];
        }
    }

    if (!local_error && chunk->nnz > 0) {
        send_rows = (int *)malloc((size_t)chunk->nnz * sizeof(int));
        send_cols = (int *)malloc((size_t)chunk->nnz * sizeof(int));
        send_data = (double *)malloc((size_t)chunk->nnz * sizeof(double));
        local_error = !send_rows || !send_cols || !send_data;
    }

    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        goto cleanup;
    }

    for (int i = 0; i < chunk->nnz; i++) {
        const int owner =
            coo_partition_owner(chunk->row[i], chunk->rows, size, partition);
        const int dest = next[owner]++;
        send_rows[dest] = chunk->row[i];
        send_cols[dest] = chunk->col[i];
        send_data[dest] = chunk->data[i];
    }

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
    compute_displacements(size, recv_counts, recv_displs);

    local_matrix->rows = chunk->rows;
    local_matrix->cols = chunk->cols;
    local_matrix->global_nnz = global_nnz;
    local_matrix->local_nnz = 0;
    for (int r = 0; r < size; r++) {
        local_matrix->local_nnz += recv_counts[r];
    }
    local_matrix->global_offset =
        partition == COO_ROW_PARTITION_BLOCK
            ? coo_partition_first_index(chunk->rows, rank, size, partition)
            : -1;
    local_matrix->row = NULL;
    local_matrix->col = NULL;
    local_matrix->data = NULL;

    local_error = allocate_local_entries(local_matrix);
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        goto cleanup;
    }

    MPI_Alltoallv(send_rows, send_counts, send_displs, MPI_INT,
                  local_matrix->row, recv_counts, recv_displs, MPI_INT, comm);
    MPI_Alltoallv(send_cols, send_counts, send_displs, MPI_INT,
                  local_matrix->col, recv_counts, recv_displs, MPI_INT, comm);
    MPI_Alltoallv(send_data, send_counts, send_displs, MPI_DOUBLE,
                  local_matrix->data, recv_counts, recv_displs, MPI_DOUBLE,
                  comm);

cleanup:
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    free(next);
    free(send_rows);
    free(send_cols);
    free(send_data);
    return any_error;
}

static int build_partitioned_send_buffers(const COO_Matrix *global_matrix,
                                          int size,
                                          COORowPartition partition,
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
        int owner = coo_partition_owner(global_matrix->row[i],
                                        global_matrix->rows, size, partition);
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
        int owner = coo_partition_owner(global_matrix->row[i],
                                        global_matrix->rows, size, partition);
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

int distribute_coo_entries_partitioned(const COO_Matrix *global_matrix,
                                       LocalCOO_Matrix *local_matrix,
                                       COORowPartition partition,
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
        local_error = build_partitioned_send_buffers(
            global_matrix, size, partition, &counts, &displs,
            &send_rows_buffer, &send_cols_buffer, &send_data_buffer);
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
    local_matrix->global_offset =
        partition == COO_ROW_PARTITION_BLOCK
            ? coo_partition_first_index(local_matrix->rows, rank, size,
                                        partition)
            : -1;

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
     * Root groups COO triples by the selected row owner, preserving global row
     * indices. MPI_Scatterv then sends each rank all nonzeros for its rows.
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

int distribute_coo_entries(const COO_Matrix *global_matrix,
                           LocalCOO_Matrix *local_matrix,
                           int root,
                           MPI_Comm comm) {
    return distribute_coo_entries_partitioned(
        global_matrix, local_matrix, COO_ROW_PARTITION_CYCLIC, root, comm);
}

int read_distributed_coo_entries(const char *filename,
                                 LocalCOO_Matrix *local_matrix,
                                 COORowPartition partition,
                                 MPI_Comm comm) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    memset(local_matrix, 0, sizeof(*local_matrix));
    COO_Matrix chunk = {0};
    int declared_stored_nnz = 0;
    int local_stored_nnz = 0;
    int local_error =
        read_mtx_chunk(filename, &chunk, &declared_stored_nnz,
                       &local_stored_nnz, rank, size);
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free_coo(&chunk);
        return 1;
    }

    int total_stored_nnz = 0;
    MPI_Allreduce(&local_stored_nnz, &total_stored_nnz, 1, MPI_INT,
                  MPI_SUM, comm);
    if (total_stored_nnz != declared_stored_nnz) {
        if (rank == 0) {
            fprintf(stderr,
                    "Matrix Market entry count mismatch in %s: declared %d, parsed %d\n",
                    filename, declared_stored_nnz, total_stored_nnz);
        }
        free_coo(&chunk);
        return 1;
    }

    int global_nnz = 0;
    MPI_Allreduce(&chunk.nnz, &global_nnz, 1, MPI_INT, MPI_SUM, comm);
    int status = redistribute_chunk_by_row(&chunk, global_nnz, local_matrix,
                                           partition, comm);
    free_coo(&chunk);
    if (status != 0) {
        free_local_coo(local_matrix);
    }
    return status;
}

int gather_coo_entries(const LocalCOO_Matrix *local_matrix,
                       COO_Matrix *global_matrix,
                       int root,
                       MPI_Comm comm) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        memset(global_matrix, 0, sizeof(*global_matrix));
        global_matrix->rows = local_matrix->rows;
        global_matrix->cols = local_matrix->cols;
        global_matrix->nnz = local_matrix->global_nnz;
    }

    int *counts = rank == root
                      ? (int *)malloc((size_t)size * sizeof(int))
                      : NULL;
    int *displs = rank == root
                      ? (int *)malloc((size_t)size * sizeof(int))
                      : NULL;
    int local_error = rank == root && (!counts || !displs);
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free(counts);
        free(displs);
        return 1;
    }

    MPI_Gather(&local_matrix->local_nnz, 1, MPI_INT,
               counts, 1, MPI_INT, root, comm);
    if (rank == root) {
        compute_displacements(size, counts, displs);
        int gathered_nnz = 0;
        for (int r = 0; r < size; r++) {
            gathered_nnz += counts[r];
        }
        if (gathered_nnz != global_matrix->nnz) {
            local_error = 1;
        } else if (gathered_nnz > 0) {
            global_matrix->row =
                (int *)malloc((size_t)gathered_nnz * sizeof(int));
            global_matrix->col =
                (int *)malloc((size_t)gathered_nnz * sizeof(int));
            global_matrix->data =
                (double *)malloc((size_t)gathered_nnz * sizeof(double));
            local_error = !global_matrix->row || !global_matrix->col ||
                          !global_matrix->data;
        }
    }

    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free(counts);
        free(displs);
        if (rank == root) {
            free_coo(global_matrix);
        }
        return 1;
    }

    MPI_Gatherv(local_matrix->row, local_matrix->local_nnz, MPI_INT,
                rank == root ? global_matrix->row : NULL, counts, displs,
                MPI_INT, root, comm);
    MPI_Gatherv(local_matrix->col, local_matrix->local_nnz, MPI_INT,
                rank == root ? global_matrix->col : NULL, counts, displs,
                MPI_INT, root, comm);
    MPI_Gatherv(local_matrix->data, local_matrix->local_nnz, MPI_DOUBLE,
                rank == root ? global_matrix->data : NULL, counts, displs,
                MPI_DOUBLE, root, comm);

    free(counts);
    free(displs);
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
