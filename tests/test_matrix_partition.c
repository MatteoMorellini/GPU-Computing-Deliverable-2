#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_partition.h"
#include "mpi_coo_distribution.h"

static int build_test_matrix(COO_Matrix *matrix) {
    const int vertices = 12;
    const int nnz = vertices * 3;
    memset(matrix, 0, sizeof(*matrix));
    matrix->rows = vertices;
    matrix->cols = vertices;
    matrix->nnz = nnz;
    matrix->row = (int *)malloc((size_t)nnz * sizeof(int));
    matrix->col = (int *)malloc((size_t)nnz * sizeof(int));
    matrix->data = (double *)malloc((size_t)nnz * sizeof(double));
    if (!matrix->row || !matrix->col || !matrix->data) return 1;
    int at = 0;
    for (int row = 0; row < vertices; row++) {
        matrix->row[at] = row;
        matrix->col[at] = row;
        matrix->data[at++] = 2.0;
        matrix->row[at] = row;
        matrix->col[at] = (row + 1) % vertices;
        matrix->data[at++] = -1.0;
        matrix->row[at] = row;
        matrix->col[at] = (row + vertices - 1) % vertices;
        matrix->data[at++] = -1.0;
    }
    return 0;
}

static int build_lra_test_matrix(COO_Matrix *matrix) {
    static const int row_nnz[12] = {
        1, 1, 1, 1, 1, 20, 1, 1, 1, 3, 3, 3,
    };
    int nnz = 0;
    for (int row = 0; row < 12; row++) nnz += row_nnz[row];

    memset(matrix, 0, sizeof(*matrix));
    matrix->rows = 12;
    matrix->cols = 12;
    matrix->nnz = nnz;
    matrix->row = (int *)malloc((size_t)nnz * sizeof(int));
    matrix->col = (int *)malloc((size_t)nnz * sizeof(int));
    matrix->data = (double *)malloc((size_t)nnz * sizeof(double));
    if (!matrix->row || !matrix->col || !matrix->data) return 1;

    int at = 0;
    for (int row = 0; row < 12; row++) {
        for (int entry = 0; entry < row_nnz[row]; entry++) {
            matrix->row[at] = row;
            matrix->col[at] = entry % 12;
            matrix->data[at++] = 1.0;
        }
    }
    return 0;
}

static int test_mode(MatrixPartitionMode mode, int rank, int size) {
    COO_Matrix global = {0};
    if (rank == 0 && build_test_matrix(&global)) return 1;
    MatrixPartition partition = {0};
    if (matrix_partition_prepare(&partition, mode, 12, size, 0, 0,
                                 0x12345678ULL, NULL,
                                 rank == 0 ? &global : NULL, 0,
                                 MPI_COMM_WORLD)) {
        if (rank == 0) free_coo(&global);
        return 1;
    }

    int local_error = 0;
    int sum_counts = 0;
    for (int owner = 0; owner < size; owner++)
        sum_counts += matrix_partition_owned_count(&partition, owner);
    if (sum_counts != 12) local_error = 1;
    for (int vertex = 0; vertex < 12; vertex++) {
        const int owner = matrix_partition_vertex_owner(&partition, vertex);
        const int local = matrix_partition_local_index(&partition, vertex);
        if (owner < 0 || owner >= size || local < 0) local_error = 1;
    }

    LocalCOO_Matrix local = {0};
    if (distribute_coo_entries_layout_timed(
            &global, &local, &partition, 0, NULL, MPI_COMM_WORLD)) {
        local_error = 1;
    } else {
        for (int i = 0; i < local.local_nnz; i++) {
            if (matrix_partition_entry_owner(
                    &partition, local.row[i], local.col[i]) != rank) {
                local_error = 1;
            }
        }
        int global_nnz = 0;
        MPI_Allreduce(&local.local_nnz, &global_nnz, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        if (global_nnz != 36) local_error = 1;
    }

    free_local_coo(&local);
    free_matrix_partition(&partition);
    if (rank == 0) free_coo(&global);
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    return any_error;
}

static int test_lra_definition(int rank, int size) {
    if (size != 2) return 0;

    COO_Matrix global = {0};
    int local_error = rank == 0 && build_lra_test_matrix(&global);
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        if (rank == 0) free_coo(&global);
        return 1;
    }

    MatrixPartition partition = {0};
    if (matrix_partition_prepare_with_long_row_fraction(
            &partition, MATRIX_PARTITION_1D_LRA, 12, size, 0, 0,
            0, 0.25, NULL, rank == 0 ? &global : NULL, 0,
            MPI_COMM_WORLD)) {
        if (rank == 0) free_coo(&global);
        return 1;
    }

    /*
     * floor(0.25*12)=3. The longest row is row 5, outside the first
     * three rows, so Section 3.3 selects rows 9..11 as the long-row
     * region. Its [3,3,3] NNZ distribution splits after row 9.
     * The complementary region splits immediately before row 5.
     */
    if (matrix_partition_vertex_owner(&partition, 4) != 0 ||
        matrix_partition_vertex_owner(&partition, 5) != 1 ||
        matrix_partition_vertex_owner(&partition, 8) != 1 ||
        matrix_partition_vertex_owner(&partition, 9) != 0 ||
        matrix_partition_vertex_owner(&partition, 10) != 1 ||
        matrix_partition_vertex_owner(&partition, 11) != 1 ||
        fabs(partition.long_row_fraction - 0.25) > 1e-12) {
        local_error = 1;
    }

    free_matrix_partition(&partition);
    if (rank == 0) free_coo(&global);
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    return any_error;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const MatrixPartitionMode modes[] = {
        MATRIX_PARTITION_1D_BLOCK, MATRIX_PARTITION_1D_RANDOM,
        MATRIX_PARTITION_1D_GP, MATRIX_PARTITION_1D_HP,
        MATRIX_PARTITION_1D_LRA,
        MATRIX_PARTITION_2D_BLOCK, MATRIX_PARTITION_2D_RANDOM,
        MATRIX_PARTITION_2D_GP, MATRIX_PARTITION_2D_HP,
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); i++) {
        const int mode_failed = test_mode(modes[i], rank, size);
        if (rank == 0)
            printf("%s: %s\n", matrix_partition_mode_name(modes[i]),
                   mode_failed ? "FAIL" : "PASS");
        failed |= mode_failed;
    }
    const int lra_failed = test_lra_definition(rank, size);
    if (rank == 0 && size == 2)
        printf("1d-lra-paper-definition: %s\n",
               lra_failed ? "FAIL" : "PASS");
    failed |= lra_failed;

    MatrixPartitionMode parsed = MATRIX_PARTITION_CYCLIC;
    int parse_failed =
        parse_matrix_partition_mode("lra", &parsed) ||
        parsed != MATRIX_PARTITION_1D_LRA ||
        parse_matrix_partition_mode("long-row-aware", &parsed) ||
        parsed != MATRIX_PARTITION_1D_LRA;
    int any_parse_failed = 0;
    MPI_Allreduce(&parse_failed, &any_parse_failed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    failed |= any_parse_failed;
    MPI_Finalize();
    return failed;
}
