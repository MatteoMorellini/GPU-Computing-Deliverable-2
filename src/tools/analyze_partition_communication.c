#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_partition.h"
#include "mtx_reader.h"

#define DEFAULT_PARTITION_SEED 20260722ULL
#define PROCESS_COUNT_COUNT 2
#define PARTITION_MODE_COUNT 6
#define COMMUNICATED_VALUE_BYTES ((long long)sizeof(float))

typedef struct {
    long long rank_nnz;
    long long expand_send;
    long long expand_recv;
    long long fold_send;
    long long fold_recv;
} RankCommunication;

static const int process_counts[PROCESS_COUNT_COUNT] = {2, 4};

static const MatrixPartitionMode partition_modes[PARTITION_MODE_COUNT] = {
    MATRIX_PARTITION_CYCLIC,
    MATRIX_PARTITION_1D_BLOCK,
    MATRIX_PARTITION_1D_GP,
    MATRIX_PARTITION_1D_LRA,
    MATRIX_PARTITION_2D_BLOCK,
    MATRIX_PARTITION_2D_GP,
};

static const char *analysis_mode_name(MatrixPartitionMode mode) {
    return mode == MATRIX_PARTITION_CYCLIC
               ? "1d-cyclic"
               : matrix_partition_mode_name(mode);
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage: %s [--output PATH] [--force] "
            "[--partition-seed N] [--long-row-fraction F] "
            "MATRIX [MATRIX ...]\n",
            program);
}

static int parse_seed(const char *text, unsigned long long *seed) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || !text[0] || !end || *end != '\0') return 1;
    *seed = value;
    return 0;
}

static int parse_fraction(const char *text, double *fraction) {
    char *end = NULL;
    errno = 0;
    const double value = strtod(text, &end);
    if (errno || !text[0] || !end || *end != '\0' ||
        !isfinite(value) || value <= 0.0 || value >= 1.0) {
        return 1;
    }
    *fraction = value;
    return 0;
}

static void write_csv_string(FILE *file, const char *value) {
    int quote = 0;
    for (const char *cursor = value; *cursor; cursor++) {
        if (*cursor == ',' || *cursor == '"' || *cursor == '\n' ||
            *cursor == '\r') {
            quote = 1;
            break;
        }
    }
    if (!quote) {
        fputs(value, file);
        return;
    }
    fputc('"', file);
    for (const char *cursor = value; *cursor; cursor++) {
        if (*cursor == '"') fputc('"', file);
        fputc(*cursor, file);
    }
    fputc('"', file);
}

static void write_csv_header(FILE *file) {
    fputs("matrix,rows,cols,total_nnz,processes,partition,"
          "long_row_fraction,process_rows,process_cols,rank,rank_nnz,"
          "expand_send_values,expand_recv_values,"
          "fold_send_values,fold_recv_values,"
          "total_send_values,total_recv_values,total_volume_values,"
          "total_send_bytes,total_recv_bytes,total_volume_bytes,"
          "min_rank_volume_values,avg_rank_volume_values,"
          "max_rank_volume_values\n",
          file);
}

static void write_csv_row(FILE *file,
                          const char *matrix_path,
                          const COO_Matrix *matrix,
                          const MatrixPartition *partition,
                          int rank,
                          const RankCommunication *communication,
                          long long min_volume,
                          double avg_volume,
                          long long max_volume) {
    const long long total_send =
        communication->expand_send + communication->fold_send;
    const long long total_recv =
        communication->expand_recv + communication->fold_recv;
    const long long total_volume = total_send + total_recv;

    write_csv_string(file, matrix_path);
    fprintf(file, ",%d,%d,%d,%d,%s",
            matrix->rows, matrix->cols, matrix->nnz,
            partition->processes, analysis_mode_name(partition->mode));
    if (partition->mode == MATRIX_PARTITION_1D_LRA) {
        fprintf(file, ",%.9f", partition->long_row_fraction);
    } else {
        fputc(',', file);
    }
    fprintf(file,
            ",%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
            "%lld,%lld,%lld,%lld,%.6f,%lld\n",
            partition->process_rows, partition->process_cols, rank,
            communication->rank_nnz,
            communication->expand_send, communication->expand_recv,
            communication->fold_send, communication->fold_recv,
            total_send, total_recv, total_volume,
            total_send * COMMUNICATED_VALUE_BYTES,
            total_recv * COMMUNICATED_VALUE_BYTES,
            total_volume * COMMUNICATED_VALUE_BYTES,
            min_volume, avg_volume, max_volume);
}

static int analyze_layout(const char *matrix_path,
                          const COO_Matrix *matrix,
                          MatrixPartition *partition,
                          MatrixPartitionMode mode,
                          FILE *csv_file) {
    partition->mode = mode;
    const int processes = partition->processes;
    RankCommunication *communication = (RankCommunication *)calloc(
        (size_t)processes, sizeof(RankCommunication));
    uint8_t *column_consumers =
        matrix->cols > 0
            ? (uint8_t *)calloc((size_t)matrix->cols, sizeof(uint8_t))
            : NULL;
    uint8_t *row_producers =
        matrix_partition_is_2d(mode) && matrix->rows > 0
            ? (uint8_t *)calloc((size_t)matrix->rows, sizeof(uint8_t))
            : NULL;
    if (!communication || (matrix->cols > 0 && !column_consumers) ||
        (matrix_partition_is_2d(mode) && matrix->rows > 0 &&
         !row_producers)) {
        fprintf(stderr,
                "Could not allocate communication counters for P=%d, %s\n",
                processes, analysis_mode_name(mode));
        free(communication);
        free(column_consumers);
        free(row_producers);
        return 1;
    }

    for (int entry = 0; entry < matrix->nnz; entry++) {
        const int row = matrix->row[entry];
        const int col = matrix->col[entry];
        const int matrix_owner =
            matrix_partition_entry_owner(partition, row, col);
        const int vector_owner =
            matrix_partition_vertex_owner(partition, col);
        if (matrix_owner < 0 || matrix_owner >= processes ||
            vector_owner < 0 || vector_owner >= processes) {
            fprintf(stderr,
                    "Invalid owner for entry %d under partition %s\n",
                    entry, analysis_mode_name(mode));
            free(communication);
            free(column_consumers);
            free(row_producers);
            return 1;
        }
        communication[matrix_owner].rank_nnz++;
        if (vector_owner != matrix_owner) {
            column_consumers[col] |= (uint8_t)(1U << matrix_owner);
        }
        if (row_producers) {
            row_producers[row] |= (uint8_t)(1U << matrix_owner);
        }
    }

    for (int col = 0; col < matrix->cols; col++) {
        const int owner = matrix_partition_vertex_owner(partition, col);
        const uint8_t consumers = column_consumers[col];
        for (int rank = 0; rank < processes; rank++) {
            if (consumers & (uint8_t)(1U << rank)) {
                communication[rank].expand_recv++;
                communication[owner].expand_send++;
            }
        }
    }

    if (row_producers) {
        for (int row = 0; row < matrix->rows; row++) {
            const int owner = matrix_partition_vertex_owner(partition, row);
            const uint8_t producers = row_producers[row];
            for (int rank = 0; rank < processes; rank++) {
                if (rank != owner &&
                    (producers & (uint8_t)(1U << rank))) {
                    communication[rank].fold_send++;
                    communication[owner].fold_recv++;
                }
            }
        }
    }

    long long expand_send_sum = 0;
    long long expand_recv_sum = 0;
    long long fold_send_sum = 0;
    long long fold_recv_sum = 0;
    long long nnz_sum = 0;
    long long min_volume = LLONG_MAX;
    long long max_volume = 0;
    long long volume_sum = 0;
    for (int rank = 0; rank < processes; rank++) {
        const long long total_send =
            communication[rank].expand_send + communication[rank].fold_send;
        const long long total_recv =
            communication[rank].expand_recv + communication[rank].fold_recv;
        const long long total_volume = total_send + total_recv;
        expand_send_sum += communication[rank].expand_send;
        expand_recv_sum += communication[rank].expand_recv;
        fold_send_sum += communication[rank].fold_send;
        fold_recv_sum += communication[rank].fold_recv;
        nnz_sum += communication[rank].rank_nnz;
        if (total_volume < min_volume) min_volume = total_volume;
        if (total_volume > max_volume) max_volume = total_volume;
        volume_sum += total_volume;
    }
    if (expand_send_sum != expand_recv_sum ||
        fold_send_sum != fold_recv_sum || nnz_sum != matrix->nnz) {
        fprintf(stderr,
                "Communication conservation failed for P=%d, %s\n",
                processes, analysis_mode_name(mode));
        free(communication);
        free(column_consumers);
        free(row_producers);
        return 1;
    }

    const double avg_volume = (double)volume_sum / (double)processes;
    printf("  P=%d  %-9s grid=%dx%d  "
           "rank volumes (send/recv/total values):\n",
           processes, analysis_mode_name(mode),
           partition->process_rows, partition->process_cols);
    for (int rank = 0; rank < processes; rank++) {
        const long long total_send =
            communication[rank].expand_send + communication[rank].fold_send;
        const long long total_recv =
            communication[rank].expand_recv + communication[rank].fold_recv;
        const long long total_volume = total_send + total_recv;
        printf("    rank %d: expand=%lld/%lld fold=%lld/%lld "
               "total=%lld/%lld/%lld (%lld bytes)\n",
               rank,
               communication[rank].expand_send,
               communication[rank].expand_recv,
               communication[rank].fold_send,
               communication[rank].fold_recv,
               total_send, total_recv, total_volume,
               total_volume * COMMUNICATED_VALUE_BYTES);
        if (csv_file) {
            write_csv_row(csv_file, matrix_path, matrix, partition, rank,
                          &communication[rank], min_volume, avg_volume,
                          max_volume);
        }
    }
    printf("    volume min/avg/max: %lld / %.2f / %lld values\n",
           min_volume, avg_volume, max_volume);

    free(communication);
    free(column_consumers);
    free(row_producers);
    return 0;
}

static int analyze_matrix(const char *matrix_path,
                          unsigned long long seed,
                          double long_row_fraction,
                          FILE *csv_file) {
    COO_Matrix matrix = {0};
    printf("\n=== %s ===\n", matrix_path);
    read_mtx(matrix_path, &matrix);
    printf("Shape: %d x %d, expanded NNZ: %d\n",
           matrix.rows, matrix.cols, matrix.nnz);

    if (matrix.rows != matrix.cols) {
        fprintf(stderr,
                "The selected GP/LRA/2D layouts require a square matrix; "
                "skipping %s (%d x %d)\n",
                matrix_path, matrix.rows, matrix.cols);
        free_coo(&matrix);
        return 1;
    }

    int failed = 0;
    for (int process_index = 0;
         process_index < PROCESS_COUNT_COUNT && !failed;
         process_index++) {
        const int processes = process_counts[process_index];
        MatrixPartition gp_partition = {0};
        int gp_partition_ready = 0;
        for (int mode_index = 0;
             mode_index < PARTITION_MODE_COUNT && !failed;
             mode_index++) {
            const MatrixPartitionMode mode = partition_modes[mode_index];
            if (mode == MATRIX_PARTITION_2D_GP) {
                if (!gp_partition_ready) {
                    fprintf(stderr,
                            "Internal error: 2d-gp has no prepared GP map\n");
                    failed = 1;
                    break;
                }
                failed = analyze_layout(
                    matrix_path, &matrix, &gp_partition, mode, csv_file);
                break;
            }

            MatrixPartition partition = {0};
            if (matrix_partition_prepare_with_long_row_fraction(
                    &partition, mode, matrix.rows, processes,
                    0, 0, seed, long_row_fraction, NULL,
                    &matrix, 0, MPI_COMM_SELF)) {
                fprintf(stderr,
                        "Could not prepare %s partition for P=%d and %s\n",
                        analysis_mode_name(mode), processes, matrix_path);
                failed = 1;
                break;
            }
            failed = analyze_layout(
                matrix_path, &matrix, &partition, mode, csv_file);
            if (mode == MATRIX_PARTITION_1D_GP && !failed) {
                gp_partition = partition;
                gp_partition_ready = 1;
            } else {
                free_matrix_partition(&partition);
            }
        }
        if (gp_partition_ready) free_matrix_partition(&gp_partition);
    }

    free_coo(&matrix);
    return failed;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 1) {
        if (world_rank == 0) {
            fprintf(stderr,
                    "This tool simulates P=2 and P=4 communication from one "
                    "matrix load; run it as a single process.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *output_path = NULL;
    unsigned long long seed = DEFAULT_PARTITION_SEED;
    double long_row_fraction =
        MATRIX_PARTITION_DEFAULT_LONG_ROW_FRACTION;
    int force = 0;
    int first_matrix = argc;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (!strncmp(argv[i], "--output=", 9)) {
            output_path = argv[i] + 9;
        } else if (!strcmp(argv[i], "--partition-seed") && i + 1 < argc) {
            if (parse_seed(argv[++i], &seed)) {
                print_usage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        } else if (!strncmp(argv[i], "--partition-seed=", 17)) {
            if (parse_seed(argv[i] + 17, &seed)) {
                print_usage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        } else if (!strcmp(argv[i], "--long-row-fraction") &&
                   i + 1 < argc) {
            if (parse_fraction(argv[++i], &long_row_fraction)) {
                print_usage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        } else if (!strncmp(argv[i], "--long-row-fraction=", 20)) {
            if (parse_fraction(argv[i] + 20, &long_row_fraction)) {
                print_usage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        } else if (!strcmp(argv[i], "--force")) {
            force = 1;
        } else if (argv[i][0] == '-') {
            print_usage(argv[0]);
            MPI_Finalize();
            return 1;
        } else {
            first_matrix = i;
            break;
        }
    }

    if (first_matrix == argc) {
        print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE *csv_file = NULL;
    if (output_path) {
        csv_file = fopen(output_path, force ? "w" : "wx");
        if (!csv_file) {
            fprintf(stderr, "Could not create %s: %s%s\n",
                    output_path, strerror(errno),
                    force ? "" : " (pass --force to replace it)");
            MPI_Finalize();
            return 1;
        }
        write_csv_header(csv_file);
    }

    printf("Per-SpMV remote communication analysis; P={2,4}, seed=%llu, "
           "long-row-fraction=%.6f\n",
           seed, long_row_fraction);
    printf("Volume = expand x traffic + 2D fold y traffic; "
           "self-copies, setup, input, and final gather are excluded.\n");
    int failed = 0;
    for (int i = first_matrix; i < argc; i++) {
        failed |= analyze_matrix(
            argv[i], seed, long_row_fraction, csv_file);
    }

    if (csv_file) {
        if (fclose(csv_file)) {
            fprintf(stderr, "Error closing %s\n", output_path);
            failed = 1;
        } else {
            printf("\nWrote %s\n", output_path);
        }
    }

    MPI_Finalize();
    return failed;
}
