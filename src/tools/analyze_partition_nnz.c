#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_partition.h"
#include "mtx_reader.h"

#define DEFAULT_PARTITION_SEED 20260722ULL
#define PROCESS_COUNT_COUNT 2
#define PARTITION_MODE_COUNT 6

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
          "long_row_fraction,process_rows,process_cols,"
          "rank_0_nnz,rank_1_nnz,rank_2_nnz,rank_3_nnz,"
          "min_nnz,avg_nnz,max_nnz,"
          "min_over_avg,max_over_avg,spread_over_avg\n",
          file);
}

static void write_csv_row(FILE *file,
                          const char *matrix_path,
                          const COO_Matrix *matrix,
                          const MatrixPartition *partition,
                          const long long *counts,
                          long long min_nnz,
                          double avg_nnz,
                          long long max_nnz) {
    write_csv_string(file, matrix_path);
    fprintf(file, ",%d,%d,%d,%d,%s",
            matrix->rows, matrix->cols, matrix->nnz,
            partition->processes,
            analysis_mode_name(partition->mode));
    if (partition->mode == MATRIX_PARTITION_1D_LRA) {
        fprintf(file, ",%.9f", partition->long_row_fraction);
    } else {
        fputc(',', file);
    }
    fprintf(file, ",%d,%d",
            partition->process_rows, partition->process_cols);
    for (int rank = 0; rank < 4; rank++) {
        if (rank < partition->processes) {
            fprintf(file, ",%lld", counts[rank]);
        } else {
            fputc(',', file);
        }
    }
    const double min_over_avg = avg_nnz > 0.0 ? min_nnz / avg_nnz : 0.0;
    const double max_over_avg = avg_nnz > 0.0 ? max_nnz / avg_nnz : 0.0;
    const double spread_over_avg =
        avg_nnz > 0.0 ? (max_nnz - min_nnz) / avg_nnz : 0.0;
    fprintf(file, ",%lld,%.6f,%lld,%.9f,%.9f,%.9f\n",
            min_nnz, avg_nnz, max_nnz,
            min_over_avg, max_over_avg, spread_over_avg);
}

static int analyze_layout(const char *matrix_path,
                          const COO_Matrix *matrix,
                          MatrixPartition *partition,
                          MatrixPartitionMode mode,
                          FILE *csv_file) {
    partition->mode = mode;
    long long *counts =
        (long long *)calloc((size_t)partition->processes, sizeof(long long));
    if (!counts) {
        fprintf(stderr, "Could not allocate NNZ counters for P=%d\n",
                partition->processes);
        return 1;
    }

    for (int entry = 0; entry < matrix->nnz; entry++) {
        const int owner = matrix_partition_entry_owner(
            partition, matrix->row[entry], matrix->col[entry]);
        if (owner < 0 || owner >= partition->processes) {
            fprintf(stderr,
                    "Invalid owner %d for entry %d under partition %s\n",
                    owner, entry, analysis_mode_name(mode));
            free(counts);
            return 1;
        }
        counts[owner]++;
    }

    long long min_nnz = LLONG_MAX;
    long long max_nnz = 0;
    long long sum_nnz = 0;
    for (int rank = 0; rank < partition->processes; rank++) {
        if (counts[rank] < min_nnz) min_nnz = counts[rank];
        if (counts[rank] > max_nnz) max_nnz = counts[rank];
        sum_nnz += counts[rank];
    }
    if (sum_nnz != matrix->nnz) {
        fprintf(stderr,
                "Partition %s assigned %lld entries; expected %d\n",
                analysis_mode_name(mode), sum_nnz, matrix->nnz);
        free(counts);
        return 1;
    }

    const double avg_nnz =
        (double)matrix->nnz / (double)partition->processes;
    const double max_over_avg = avg_nnz > 0.0 ? max_nnz / avg_nnz : 0.0;
    const double spread_over_avg =
        avg_nnz > 0.0 ? (max_nnz - min_nnz) / avg_nnz : 0.0;

    printf("  P=%d  %-9s grid=%dx%d  ranks=[",
           partition->processes, analysis_mode_name(mode),
           partition->process_rows, partition->process_cols);
    for (int rank = 0; rank < partition->processes; rank++) {
        printf("%s%lld", rank ? ", " : "", counts[rank]);
    }
    printf("]  min=%lld avg=%.2f max=%lld  max/avg=%.6f "
           "spread/avg=%.6f\n",
           min_nnz, avg_nnz, max_nnz, max_over_avg, spread_over_avg);

    if (csv_file) {
        write_csv_row(csv_file, matrix_path, matrix, partition, counts,
                      min_nnz, avg_nnz, max_nnz);
    }
    free(counts);
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
                "All analyzed 1D/2D paper layouts require a square matrix; "
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
                        analysis_mode_name(mode),
                        processes, matrix_path);
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
                    "This tool simulates P=2 and P=4 ownership from one "
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

    printf("NNZ ownership analysis only (no SpMV); P={2,4}, seed=%llu, "
           "long-row-fraction=%.6f\n",
           seed, long_row_fraction);
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
