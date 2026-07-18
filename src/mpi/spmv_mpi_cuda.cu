#include <cuda_runtime.h>
#include <mpi.h>

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coo_to_csr.h"
#include "generate_dense.h"
#include "mpi_coo_distribution.h"
#include "mtx_reader.h"
#include "perf_stats.h"
#include "spmv_kernel_runner.cuh"

#ifndef BENCH_REPS
#define BENCH_REPS 100
#endif

#ifndef BENCH_WARMUP
#define BENCH_WARMUP 5
#endif

#ifndef BENCH_MATRICES_DIR
#define BENCH_MATRICES_DIR "./matrices/"
#endif

static void check_cuda(cudaError_t status,
                       const char *call,
                       const char *file,
                       int line,
                       int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d CUDA error at %s:%d in %s: %s\n",
                rank, file, line, call, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CHECK_CUDA(call, rank) \
    check_cuda((call), #call, __FILE__, __LINE__, (rank))

static void abort_all(const char *message, int rank) {
    if (rank == 0) {
        fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static void reduce_max_time(double local_seconds,
                            double *global_seconds,
                            MPI_Comm comm) {
    MPI_Reduce(&local_seconds, global_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
}

static void csr_to_device(const CSR_Matrix *host,
                          CSR_Matrix *device,
                          int rank) {
    device->rows = host->rows;
    device->cols = host->cols;
    device->nnz = host->nnz;
    device->row_ptr = NULL;
    device->col_idx = NULL;
    device->values = NULL;

    CHECK_CUDA(cudaMalloc((void **)&device->row_ptr,
                          (size_t)(host->rows + 1) * sizeof(int)),
               rank);
    CHECK_CUDA(cudaMemcpy(device->row_ptr, host->row_ptr,
                          (size_t)(host->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice),
               rank);

    if (host->nnz > 0) {
        CHECK_CUDA(cudaMalloc((void **)&device->col_idx,
                              (size_t)host->nnz * sizeof(int)),
                   rank);
        CHECK_CUDA(cudaMalloc((void **)&device->values,
                              (size_t)host->nnz * sizeof(float)),
                   rank);
        CHECK_CUDA(cudaMemcpy(device->col_idx, host->col_idx,
                              (size_t)host->nnz * sizeof(int),
                              cudaMemcpyHostToDevice),
                   rank);
        CHECK_CUDA(cudaMemcpy(device->values, host->values,
                              (size_t)host->nnz * sizeof(float),
                              cudaMemcpyHostToDevice),
                   rank);
    }
}

static void free_device_csr(CSR_Matrix *device) {
    cudaFree(device->row_ptr);
    cudaFree(device->col_idx);
    cudaFree(device->values);
    device->row_ptr = NULL;
    device->col_idx = NULL;
    device->values = NULL;
}

static double checksum(const float *y, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += y[i];
    }
    return sum;
}

static const char *matrix_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static int has_mtx_extension(const char *name) {
    const char *ext = strrchr(name, '.');
    return ext && strcmp(ext, ".mtx") == 0;
}

static void reduce_kernel_times(const double *times,
                                int n,
                                int nnz,
                                PerfStats *stats) {
    double total = 0.0;
    for (int r = 0; r < n; r++) {
        total += times[r];
    }
    double avg = total / (double)n;

    double variance = 0.0;
    for (int r = 0; r < n; r++) {
        double diff = times[r] - avg;
        variance += diff * diff;
    }
    variance /= (double)n;

    stats->avg_time_s = avg;
    stats->std_time_s = sqrt(variance);
    stats->gflops = avg > 0.0 ? (2.0 * (double)nnz) / (avg * 1e9) : 0.0;
}

static void validate_vs_reference(const CSR_Matrix *csr,
                                  const float *x,
                                  const float *gpu_y,
                                  PerfStats *stats) {
    double max_abs_error = 0.0;
    int mismatches = 0;
    const double tol = 1e-3;

    for (int row = 0; row < csr->rows; row++) {
        double expected = 0.0;
        for (int j = csr->row_ptr[row]; j < csr->row_ptr[row + 1]; j++) {
            expected += (double)csr->values[j] * (double)x[csr->col_idx[j]];
        }

        double actual = (double)gpu_y[row];
        double abs_error = fabs(actual - expected);
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
        if (abs_error > tol * fmax(1.0, fabs(expected))) {
            if (mismatches < 5) {
                printf("  MISMATCH row %d: gpu=%g cpu=%g abs_err=%g\n",
                       row, actual, expected, abs_error);
            }
            mismatches++;
        }
    }

    stats->valid = mismatches == 0;
    stats->max_abs_error = max_abs_error;
    if (mismatches) {
        printf("  CORRECTNESS: %d mismatches (max_abs_err=%g)\n",
               mismatches, max_abs_error);
    } else {
        printf("  CORRECTNESS: OK\n");
    }
}

static void print_usage(const char *program, FILE *stream) {
    fprintf(stream,
            "Usage: %s [--kernel <name>] [--reps N] [--warmup N] "
            "[--output path]\n"
            "Runs every .mtx file in %s.\n",
            program, BENCH_MATRICES_DIR);
    print_spmv_kernel_choices(stream);
}

static int parse_args(int argc,
                      char **argv,
                      const char **kernel_name,
                      int *reps,
                      int *warmup,
                      const char **csv_path) {
    *kernel_name = "scalar";
    *reps = BENCH_REPS;
    *warmup = BENCH_WARMUP;
    *csv_path = "results/mpi_spmv.csv";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kernel") == 0) {
            if (i + 1 >= argc) {
                return 1;
            }
            *kernel_name = argv[++i];
        } else if (strncmp(argv[i], "--kernel=", 9) == 0) {
            *kernel_name = argv[i] + 9;
        } else if (strcmp(argv[i], "--reps") == 0) {
            if (i + 1 >= argc) {
                return 1;
            }
            *reps = atoi(argv[++i]);
        } else if (strncmp(argv[i], "--reps=", 7) == 0) {
            *reps = atoi(argv[i] + 7);
        } else if (strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 >= argc) {
                return 1;
            }
            *warmup = atoi(argv[++i]);
        } else if (strncmp(argv[i], "--warmup=", 9) == 0) {
            *warmup = atoi(argv[i] + 9);
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                return 1;
            }
            *csv_path = argv[++i];
        } else if (strncmp(argv[i], "--output=", 9) == 0) {
            *csv_path = argv[i] + 9;
        } else {
            return 1;
        }
    }

    return *reps > 0 && *warmup >= 0 ? 0 : 1;
}

static int owned_row_count(int rows, int rank, int size) {
    if (rank >= rows) {
        return 0;
    }
    return ((rows - 1 - rank) / size) + 1;
}

static void build_owned_row_counts(int rows,
                                   int size,
                                   int *counts,
                                   int *displs) {
    int offset = 0;
    for (int rank = 0; rank < size; rank++) {
        counts[rank] = owned_row_count(rows, rank, size);
        displs[rank] = offset;
        offset += counts[rank];
    }
}

static int create_local_csr_coo(const LocalCOO_Matrix *local,
                                int rank,
                                int size,
                                int local_rows,
                                COO_Matrix *local_as_coo) {
    local_as_coo->rows = local_rows;
    local_as_coo->cols = local->cols;
    local_as_coo->nnz = local->local_nnz;
    local_as_coo->row = NULL;
    local_as_coo->col = local->col;
    local_as_coo->data = local->data;

    if (local->local_nnz == 0) {
        return 0;
    }

    local_as_coo->row = (int *)malloc((size_t)local->local_nnz * sizeof(int));
    if (!local_as_coo->row) {
        return 1;
    }

    for (int i = 0; i < local->local_nnz; i++) {
        int global_row = local->row[i];
        if (global_row % size != rank) {
            fprintf(stderr,
                    "Rank %d received row %d, owned by rank %d\n",
                    rank, global_row, global_row % size);
            free(local_as_coo->row);
            local_as_coo->row = NULL;
            return 1;
        }
        local_as_coo->row[i] = global_row / size;
    }

    return 0;
}

static void reconstruct_global_y(const float *gathered_y,
                                 float *y,
                                 const int *row_counts,
                                 const int *row_displs,
                                 int size) {
    for (int owner = 0; owner < size; owner++) {
        for (int local_row = 0; local_row < row_counts[owner]; local_row++) {
            int global_row = owner + local_row * size;
            y[global_row] = gathered_y[row_displs[owner] + local_row];
        }
    }
}

static void execute_kernel(SpmvKernelKind kind,
                           const CSR_Matrix *csr,
                           const CSR_Matrix *d_csr,
                           const CSR_Matrix *reference_csr,
                           float *d_x,
                           float *d_y,
                           float *y_local,
                           float *gathered_y,
                           float *y,
                           const float *x,
                           int local_rows,
                           int global_rows,
                           int global_cols,
                           int global_nnz,
                           int rank,
                           int size,
                           const int *row_counts,
                           const int *row_displs,
                           const char *matrix_name,
                           int reps,
                           int warmup,
                           double scatter_seconds,
                           double convert_seconds,
                           double h2d_seconds,
                           double file_parse_seconds,
                           FILE *csv,
                           MPI_Comm comm) {
    const char *kernel_name = spmv_kernel_kind_name(kind);
    double prep_seconds = 0.0;
    double merge_seconds = 0.0;
    double *kernel_times = rank == 0 ? (double *)malloc((size_t)reps * sizeof(double)) : NULL;
    if (rank == 0 && !kernel_times) {
        abort_all("Error allocating kernel timing array", rank);
    }

    SpmvKernelRunner *runner = NULL;
    if (local_rows > 0) {
        runner = create_spmv_kernel_runner(kind, rank);
        if (!runner) {
            abort_all("Error creating SpMV kernel runner", rank);
        }

        prep_seconds = runner->prep(*csr, *d_csr, d_x, d_y);

        for (int r = 0; r < warmup; r++) {
            runner->launch(*d_csr, d_x, d_y);
        }
        CHECK_CUDA(cudaGetLastError(), rank);
        CHECK_CUDA(cudaDeviceSynchronize(), rank);
    }

    cudaEvent_t event_start;
    cudaEvent_t event_stop;
    if (local_rows > 0) {
        CHECK_CUDA(cudaEventCreate(&event_start), rank);
        CHECK_CUDA(cudaEventCreate(&event_stop), rank);
    }

    for (int r = 0; r < reps; r++) {
        double local_kernel_seconds = 0.0;
        if (local_rows > 0) {
            CHECK_CUDA(cudaEventRecord(event_start), rank);
            runner->launch(*d_csr, d_x, d_y);
            CHECK_CUDA(cudaEventRecord(event_stop), rank);
            CHECK_CUDA(cudaEventSynchronize(event_stop), rank);
            float milliseconds = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, event_start, event_stop),
                       rank);
            local_kernel_seconds = (double)milliseconds / 1000.0;
        }

        double global_kernel_seconds = 0.0;
        MPI_Reduce(&local_kernel_seconds, &global_kernel_seconds, 1,
                   MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            kernel_times[r] = global_kernel_seconds;
        }
    }

    if (local_rows > 0) {
        runner->launch(*d_csr, d_x, d_y);
        CHECK_CUDA(cudaGetLastError(), rank);
        CHECK_CUDA(cudaDeviceSynchronize(), rank);
        CHECK_CUDA(cudaMemcpy(y_local, d_y, (size_t)local_rows * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   rank);
        CHECK_CUDA(cudaEventDestroy(event_start), rank);
        CHECK_CUDA(cudaEventDestroy(event_stop), rank);
    }

    double merge_start = MPI_Wtime();
    MPI_Gatherv(y_local, local_rows, MPI_FLOAT,
                gathered_y, row_counts, row_displs, MPI_FLOAT,
                0, comm);

    if (rank == 0) {
        reconstruct_global_y(gathered_y, y, row_counts, row_displs, size);
    }
    merge_seconds = MPI_Wtime() - merge_start;

    double max_scatter_seconds = 0.0;
    double max_convert_seconds = 0.0;
    double max_prep_seconds = 0.0;
    double max_h2d_seconds = 0.0;
    double max_merge_seconds = 0.0;

    reduce_max_time(scatter_seconds, &max_scatter_seconds, comm);
    reduce_max_time(convert_seconds, &max_convert_seconds, comm);
    reduce_max_time(prep_seconds, &max_prep_seconds, comm);
    reduce_max_time(h2d_seconds, &max_h2d_seconds, comm);
    reduce_max_time(merge_seconds, &max_merge_seconds, comm);

    if (rank == 0) {
        PerfStats stats;
        memset(&stats, 0, sizeof(stats));
        snprintf(stats.name, sizeof(stats.name), "%s", matrix_name);
        snprintf(stats.format, sizeof(stats.format), "CSR");
        snprintf(stats.implementation, sizeof(stats.implementation), "MPI %s",
                 kernel_name);
        stats.rows = global_rows;
        stats.cols = global_cols;
        stats.nnz = global_nnz;
        stats.processes = size;
        stats.file_parse_s = file_parse_seconds;
        stats.format_conv_s = max_convert_seconds + max_prep_seconds;
        stats.h2d_transfer_s = max_h2d_seconds;

        reduce_kernel_times(kernel_times, reps, global_nnz, &stats);
        validate_vs_reference(reference_csr, x, y, &stats);

        perf_stats_write_csv_row(csv, &stats);

        printf("[%s] avg=%.9f s | GFLOP/s=%.6f | std=%.9f s\n",
               kernel_name, stats.avg_time_s, stats.gflops, stats.std_time_s);
        printf("[%s] setup: parse %.6f s, scatter %.6f s, local COO->CSR+prep %.6f s, H2D %.6f s, merge %.6f s\n",
               kernel_name, stats.file_parse_s, max_scatter_seconds,
               stats.format_conv_s, stats.h2d_transfer_s, max_merge_seconds);
        printf("[%s] result checksum: %.8e\n", kernel_name,
               checksum(y, global_rows));

        int sample_count = global_rows < 8 ? global_rows : 8;
        printf("[%s] first %d y values:", kernel_name, sample_count);
        for (int i = 0; i < sample_count; i++) {
            printf(" %.6e", y[i]);
        }
        printf("\n");
    }

    if (runner) {
        runner->teardown();
        destroy_spmv_kernel_runner(runner);
    }
    free(kernel_times);
}

static int run_matrix(const char *matrix_path,
                      const SpmvKernelKind *kernel_kinds,
                      int kernel_count,
                      const char *kernel_name,
                      int reps,
                      int warmup,
                      FILE *csv,
                      const char *csv_path,
                      int rank,
                      int size) {
    COO_Matrix global = {0};
    CSR_Matrix reference_csr = {0};
    double read_seconds = 0.0;
    int any_error = 0;

    if (rank == 0) {
        double read_start = MPI_Wtime();
        read_mtx(matrix_path, &global);
        read_seconds = MPI_Wtime() - read_start;

        printf("Rank 0 read %s: %d rows, %d cols, %d non-zeros in %.6f s\n",
               matrix_path, global.rows, global.cols, global.nnz, read_seconds);
        coo_to_csr(&global, &reference_csr);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_start = MPI_Wtime();

    LocalCOO_Matrix local = {0};
    if (distribute_coo_entries(&global, &local, 0, MPI_COMM_WORLD) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error distributing COO entries\n");
        }
        if (rank == 0) {
            free_coo(&global);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double scatter_seconds = MPI_Wtime() - scatter_start;

    int total_distributed = 0;
    MPI_Reduce(&local.local_nnz, &total_distributed, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    int local_rows = owned_row_count(local.rows, rank, size);
    float *x = (float *)malloc((size_t)local.cols * sizeof(float));
    float *y_local = local_rows > 0 ? (float *)malloc((size_t)local_rows * sizeof(float)) : NULL;
    float *y = rank == 0 ? (float *)malloc((size_t)local.rows * sizeof(float)) : NULL;
    float *gathered_y = rank == 0 ? (float *)malloc((size_t)local.rows * sizeof(float)) : NULL;
    int allocation_error = (!x || (local_rows > 0 && !y_local) ||
                            (rank == 0 && (!y || !gathered_y)));
    MPI_Allreduce(&allocation_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        free(x);
        free(y_local);
        free(y);
        free(gathered_y);
        free_local_coo(&local);
        if (rank == 0) {
            free_coo(&global);
        }
        abort_all("Error allocating dense vectors", rank);
    }

    if (rank == 0) {
        srand(0);
        fill_dense(x, (size_t)local.cols);
    }
    MPI_Bcast(x, local.cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double convert_start = MPI_Wtime();
    COO_Matrix local_as_coo;
    int remap_error = create_local_csr_coo(&local, rank, size, local_rows,
                                           &local_as_coo);
    MPI_Allreduce(&remap_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        free(local_as_coo.row);
        free(x);
        free(y_local);
        free(y);
        free(gathered_y);
        free_local_coo(&local);
        if (rank == 0) {
            free_coo(&global);
        }
        abort_all("Error remapping cyclic rows to local CSR rows", rank);
    }

    CSR_Matrix csr = {0};
    coo_to_csr(&local_as_coo, &csr);
    free(local_as_coo.row);
    double convert_seconds = MPI_Wtime() - convert_start;

    CSR_Matrix d_csr = {0};
    float *d_x = NULL;
    float *d_y = NULL;

    double h2d_start = MPI_Wtime();
    csr_to_device(&csr, &d_csr, rank);
    CHECK_CUDA(cudaDeviceSynchronize(), rank);
    double h2d_seconds = MPI_Wtime() - h2d_start;

    CHECK_CUDA(cudaMalloc((void **)&d_x, (size_t)local.cols * sizeof(float)),
               rank);
    CHECK_CUDA(cudaMemcpy(d_x, x, (size_t)local.cols * sizeof(float),
                          cudaMemcpyHostToDevice),
               rank);
    if (local_rows > 0) {
        CHECK_CUDA(cudaMalloc((void **)&d_y, (size_t)local_rows * sizeof(float)),
                   rank);
    }

    int *row_counts = NULL;
    int *row_displs = NULL;
    if (rank == 0) {
        row_counts = (int *)malloc((size_t)size * sizeof(int));
        row_displs = (int *)malloc((size_t)size * sizeof(int));
        if (!row_counts || !row_displs) {
            free(row_counts);
            free(row_displs);
            abort_all("Error allocating row gather metadata", rank);
        }
        build_owned_row_counts(local.rows, size, row_counts, row_displs);
    }

    if (rank == 0) {
        printf("Distributed %d/%d entries across %d ranks\n",
               total_distributed, global.nnz, size);
        printf("Selected kernel: %s\n", kernel_name);
        printf("Warmup repetitions: %d, measured repetitions: %d\n",
               warmup, reps);
        printf("Writing CSV metrics to %s\n", perf_stats_resolve_path(csv_path));
    }

    for (int i = 0; i < kernel_count; i++) {
        execute_kernel(kernel_kinds[i], &csr, &d_csr, &reference_csr, d_x, d_y,
                       y_local, gathered_y, y, x, local_rows, local.rows,
                       local.cols, local.global_nnz, rank, size,
                       row_counts, row_displs, matrix_basename(matrix_path),
                       reps, warmup, scatter_seconds, convert_seconds,
                       h2d_seconds, read_seconds, csv, MPI_COMM_WORLD);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    free_device_csr(&d_csr);
    free_csr(&csr);
    free(row_counts);
    free(row_displs);
    free(x);
    free(y_local);
    free(y);
    free(gathered_y);
    free_local_coo(&local);
    if (rank == 0) {
        free_csr(&reference_csr);
        free_coo(&global);
    }

    return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *kernel_name = NULL;
    const char *csv_path = NULL;
    int reps = 0;
    int warmup = 0;
    if (parse_args(argc, argv, &kernel_name, &reps, &warmup, &csv_path) != 0) {
        if (rank == 0) {
            print_usage(argv[0], stderr);
        }
        MPI_Finalize();
        return 1;
    }

    SpmvKernelKind kernel_kinds[SPMV_KERNEL_CUSPARSE + 1];
    int kernel_count = 0;
    if (strcmp(kernel_name, "all") == 0) {
        for (int i = 0; i <= SPMV_KERNEL_CUSPARSE; i++) {
            kernel_kinds[kernel_count++] = (SpmvKernelKind)i;
        }
    } else {
        if (parse_spmv_kernel_kind(kernel_name, &kernel_kinds[0]) != 0) {
            if (rank == 0) {
                fprintf(stderr, "Unknown kernel: %s\n", kernel_name);
                print_spmv_kernel_choices(stderr);
            }
            MPI_Finalize();
            return 1;
        }
        kernel_count = 1;
    }

    FILE *csv = NULL;
    int csv_error = 0;
    if (rank == 0) {
        csv = perf_stats_open_csv(perf_stats_resolve_path(csv_path));
        csv_error = csv == NULL;
    }
    int any_error = 0;
    MPI_Allreduce(&csv_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        MPI_Finalize();
        return 1;
    }

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count), rank);
    if (device_count <= 0) {
        abort_all("No CUDA devices available", rank);
    }
    CHECK_CUDA(cudaSetDevice(rank % device_count), rank);

    DIR *dir = NULL;
    int dir_error = 0;
    if (rank == 0) {
        dir = opendir(BENCH_MATRICES_DIR);
        if (!dir) {
            fprintf(stderr, "Error opening directory %s\n", BENCH_MATRICES_DIR);
            dir_error = 1;
        } else {
            printf("Files in %s:\n", BENCH_MATRICES_DIR);
        }
    }
    MPI_Allreduce(&dir_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        if (rank == 0) {
            fclose(csv);
        }
        MPI_Finalize();
        return 1;
    }

    int found_matrix = 0;
    int status = 0;
    while (1) {
        int has_matrix = 0;
        char matrix_path[1024] = {0};

        if (rank == 0) {
            struct dirent *entry = NULL;
            while ((entry = readdir(dir)) != NULL) {
                if (entry->d_name[0] == '.') {
                    continue;
                }
                if (!has_mtx_extension(entry->d_name)) {
                    continue;
                }

                snprintf(matrix_path, sizeof(matrix_path), "%s%s",
                         BENCH_MATRICES_DIR, entry->d_name);
                has_matrix = 1;
                found_matrix = 1;
                break;
            }
        }

        MPI_Bcast(&has_matrix, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!has_matrix) {
            break;
        }

        MPI_Bcast(matrix_path, sizeof(matrix_path), MPI_CHAR, 0,
                  MPI_COMM_WORLD);
        if (run_matrix(matrix_path, kernel_kinds, kernel_count, kernel_name,
                       reps, warmup, csv, csv_path, rank, size) != 0) {
            status = 1;
            break;
        }
    }

    if (rank == 0) {
        closedir(dir);
        if (!found_matrix) {
            fprintf(stderr, "No .mtx files found in %s\n", BENCH_MATRICES_DIR);
            status = 1;
        }
        fclose(csv);
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
