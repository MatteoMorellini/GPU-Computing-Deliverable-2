#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#include "mpi_coo_distribution.h"
#include "mtx_reader.h"

typedef struct {
    int row;
    int col;
    double value;
} Entry;

static int compare_entries(const void *left, const void *right) {
    const Entry *a = (const Entry *)left;
    const Entry *b = (const Entry *)right;
    if (a->row != b->row) {
        return a->row < b->row ? -1 : 1;
    }
    if (a->col != b->col) {
        return a->col < b->col ? -1 : 1;
    }
    if (a->value == b->value) {
        return 0;
    }
    return a->value < b->value ? -1 : 1;
}

static Entry *sorted_entries(const LocalCOO_Matrix *matrix) {
    if (matrix->local_nnz == 0) {
        return NULL;
    }

    Entry *entries =
        (Entry *)malloc((size_t)matrix->local_nnz * sizeof(Entry));
    if (!entries) {
        return NULL;
    }
    for (int i = 0; i < matrix->local_nnz; i++) {
        entries[i].row = matrix->row[i];
        entries[i].col = matrix->col[i];
        entries[i].value = matrix->data[i];
    }
    qsort(entries, (size_t)matrix->local_nnz, sizeof(Entry), compare_entries);
    return entries;
}

static int compare_local_matrices(const LocalCOO_Matrix *expected,
                                  const LocalCOO_Matrix *actual) {
    if (expected->rows != actual->rows ||
        expected->cols != actual->cols ||
        expected->global_nnz != actual->global_nnz ||
        expected->local_nnz != actual->local_nnz ||
        expected->global_offset != actual->global_offset) {
        return 1;
    }

    Entry *expected_entries = sorted_entries(expected);
    Entry *actual_entries = sorted_entries(actual);
    if (expected->local_nnz > 0 &&
        (!expected_entries || !actual_entries)) {
        free(expected_entries);
        free(actual_entries);
        return 1;
    }

    int mismatch = 0;
    for (int i = 0; i < expected->local_nnz; i++) {
        if (compare_entries(&expected_entries[i], &actual_entries[i]) != 0) {
            mismatch = 1;
            break;
        }
    }
    free(expected_entries);
    free(actual_entries);
    return mismatch;
}

static int test_file(const char *filename,
                     COORowPartition partition,
                     int rank,
                     MPI_Comm comm) {
    COO_Matrix global = {0};
    if (rank == 0) {
        read_mtx(filename, &global);
    }

    LocalCOO_Matrix expected = {0};
    int status = distribute_coo_entries_partitioned(
        &global, &expected, partition, 0, comm);
    if (rank == 0) {
        free_coo(&global);
    }
    if (status != 0) {
        free_local_coo(&expected);
        return 1;
    }

    LocalCOO_Matrix actual = {0};
    MatrixInputMetrics metrics = {0};
    status = read_mpi_io_coo_entries_timed(
        filename, &actual, partition, &metrics, comm);
    long long local_traffic[] = {metrics.source_nnz, metrics.remote_nnz};
    long long global_traffic[2] = {0, 0};
    MPI_Allreduce(local_traffic, global_traffic, 2, MPI_LONG_LONG, MPI_SUM,
                  comm);
    int invalid_metrics =
        metrics.total_s < 0.0 || metrics.read_parse_s < 0.0 ||
        metrics.file_io_s < 0.0 || metrics.parse_s < 0.0 ||
        metrics.validation_s < 0.0 || metrics.redistribution_s < 0.0 ||
        metrics.pack_s < 0.0 || metrics.exchange_s < 0.0 ||
        metrics.remote_nnz < 0 || metrics.remote_nnz > metrics.source_nnz ||
        global_traffic[0] != actual.global_nnz ||
        global_traffic[1] > global_traffic[0];
    int local_mismatch =
        status != 0 || invalid_metrics ||
        compare_local_matrices(&expected, &actual);
    int any_mismatch = 0;
    MPI_Allreduce(&local_mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, comm);

    free_local_coo(&expected);
    free_local_coo(&actual);
    return any_mismatch;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s matrix.mtx [matrix.mtx ...]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int failed = 0;
    for (int i = 1; i < argc; i++) {
        for (int partition = COO_ROW_PARTITION_CYCLIC;
             partition <= COO_ROW_PARTITION_BLOCK; partition++) {
            if (test_file(argv[i], (COORowPartition)partition, rank,
                          MPI_COMM_WORLD) != 0) {
                failed = 1;
                if (rank == 0) {
                    fprintf(stderr,
                            "FAIL: MPI-IO differs from serial input for %s "
                            "(P=%d, partition=%s)\n",
                            argv[i], size,
                            partition == COO_ROW_PARTITION_BLOCK
                                ? "block"
                                : "cyclic");
                }
            }
        }
    }

    if (rank == 0 && !failed) {
        printf("PASS: MPI-IO matches serial input for %d file(s) at P=%d\n",
               argc - 1, size);
    }
    MPI_Finalize();
    return failed;
}
