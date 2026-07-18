#include <stdio.h>
#include <stdlib.h>
#include "mpi_coo_distribution.h"

static void compute_counts_and_displacements(int nnz, int size,
                                             int *counts, int *displs) {
    int base = nnz / size;
    int remainder = nnz % size;
    int offset = 0;

    /*
     * Split the global COO entries as evenly as possible.
     *
     * Example: nnz = 10, size = 4
     *   counts = [3, 3, 2, 2]
     *   displs = [0, 3, 6, 8]
     *
     * counts[rank] tells MPI how many entries that rank receives.
     * displs[rank] tells MPI where that rank's slice starts in the root array.
     */
    for (int rank = 0; rank < size; rank++) {
        counts[rank] = base + (rank < remainder ? 1 : 0);
        displs[rank] = offset;
        offset += counts[rank];
    }
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
     *   - nnz is needed to compute the balanced entry partition;
     *   - rows and cols describe the global matrix that each local slice
     *     belongs to;
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

    int *counts = malloc((size_t)size * sizeof(int));
    /* counts[i] = number of entries for rank i */
    int *displs = malloc((size_t)size * sizeof(int));
    /* displs[i] = starting index of entries for rank i */
    /* eg global entries: 0 1 2 3 4 5 6 7 8 9 10
       rank 0 gets: 0 1 2  rank 1 gets: 3 4 5 ...
       counts = [3, 3, 2, 2] displs = [0, 3, 6, 8]
    */

    // check whether either allocation failed
    local_error = (!counts || !displs);

    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free(counts);
        free(displs);
        return 1;
    }
    
    /*
        Every rank computes the full counts and displs array
        Only the root rank actually uses counts and displs to decide 
        what to send to each process. But the function signature is
        collective, so every rank calls it with those parameters
    */
    compute_counts_and_displacements(local_matrix->global_nnz, size, counts, displs);
    local_matrix->local_nnz = counts[rank];
    local_matrix->global_offset = displs[rank];

    // allocate this rank's local COO arrays
    local_error = allocate_local_entries(local_matrix);

    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free(counts);
        free(displs);
        return 1;
    }

    // if this process is the root, prepare the send buffers for MPI_Scatterv, else NULL
    const int *send_rows = rank == root ? global_matrix->row : NULL;
    const int *send_cols = rank == root ? global_matrix->col : NULL;
    const double *send_data = rank == root ? global_matrix->data : NULL;

    /*
     * Use MPI_Scatterv instead of MPI_Scatter because nnz is not guaranteed to
     * be divisible by the number of processes. Different ranks may receive
     * different local_nnz values.
     *
     * COO stores one matrix entry across three parallel arrays:
     *   row[i], col[i], data[i]
     *
     * Scattering the same interval from all three arrays preserves valid
     * (row, col, value) triples on each rank.
     * 
     * Each rank receives its local_nnz entries from the global arrays, starting
     * at the global_offset index, into local_matrix->row ...
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
