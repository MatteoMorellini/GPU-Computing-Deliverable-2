#include <cuda_runtime.h>
#include <mpi.h>

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "coo_to_csr.h"
#include "generate_dense.h"
#include "mpi_coo_distribution.h"
#include "mtx_reader.h"
#include "perf_stats.h"
#include "spmv_kernel_runner.cuh"

typedef enum {
    X_MODE_DISTRIBUTED = 0,
    X_MODE_REPLICATED
} XVectorMode;

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

static const char *x_vector_mode_name(XVectorMode mode) {
    return mode == X_MODE_REPLICATED ? "replicated" : "distributed";
}

static const char *x_vector_mode_short_name(XVectorMode mode) {
    return mode == X_MODE_REPLICATED ? "repl" : "dist";
}

static int parse_x_vector_mode(const char *name, XVectorMode *mode) {
    if (strcmp(name, "distributed") == 0 || strcmp(name, "dist") == 0) {
        *mode = X_MODE_DISTRIBUTED;
        return 0;
    }
    if (strcmp(name, "replicated") == 0 || strcmp(name, "repl") == 0) {
        *mode = X_MODE_REPLICATED;
        return 0;
    }
    return 1;
}

static void compute_time_stats(const double *times,
                               int n,
                               double *avg_out,
                               double *std_out) {
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

    *avg_out = avg;
    *std_out = sqrt(variance);
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
            "[--output path] [--x-mode distributed|replicated] "
            "[--cuda-aware-mpi]\n"
            "Runs every .mtx file in %s.\n",
            program, BENCH_MATRICES_DIR);
    print_spmv_kernel_choices(stream);
}

static int parse_args(int argc,
                      char **argv,
                      const char **kernel_name,
                      int *reps,
                      int *warmup,
                      const char **csv_path,
                      XVectorMode *x_mode,
                      int *cuda_aware_mpi) {
    *kernel_name = "scalar";
    *reps = BENCH_REPS;
    *warmup = BENCH_WARMUP;
    *csv_path = "results/mpi_spmv.csv";
    *x_mode = X_MODE_DISTRIBUTED;
    *cuda_aware_mpi = 0;

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
        } else if (strcmp(argv[i], "--x-mode") == 0) {
            if (i + 1 >= argc ||
                parse_x_vector_mode(argv[++i], x_mode) != 0) {
                return 1;
            }
        } else if (strncmp(argv[i], "--x-mode=", 9) == 0) {
            if (parse_x_vector_mode(argv[i] + 9, x_mode) != 0) {
                return 1;
            }
        } else if (strcmp(argv[i], "--cuda-aware-mpi") == 0) {
            *cuda_aware_mpi = 1;
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

static int owned_col_count(int cols, int rank, int size) {
    return owned_row_count(cols, rank, size);
}

static int owner_of_vector_entry(int global_col, int size) {
    return global_col % size;
}

static int owned_vector_local_index(int global_col, int size) {
    return global_col / size;
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

typedef struct {
    /*
     * Vector ownership:
     *   owner(j) = j % number_of_processes.
     *
     * Each rank stores its owned x values first. Remote columns referenced by
     * local rows are stored after that owned segment as ghost entries.
     */
    int owned_count;
    int ghost_count;
    int compact_count;

    /*
     * Ghost-index construction:
     * recv_global_cols is grouped by owning rank.  recv_counts[owner] tells how
     * many x_j values this rank needs from that owner, and recv_displs[owner]
     * points to their positions inside the ghost segment.
     */
    int *recv_counts;
    int *recv_displs;
    int *recv_global_cols;

    /*
     * For communication of ghost values, send_global_cols is the transposed
     * request list: columns owned by this rank that remote ranks need.
     */
    int *send_counts;
    int *send_displs;
    int *send_global_cols;

    float *send_values;
    float *ghost_values;
    int *d_send_local_indices;
    float *d_send_values;
    MPI_Request *requests;
} DistributedXPlan;

static void free_distributed_x_plan(DistributedXPlan *plan) {
    free(plan->recv_counts);
    free(plan->recv_displs);
    free(plan->recv_global_cols);
    free(plan->send_counts);
    free(plan->send_displs);
    free(plan->send_global_cols);
    free(plan->send_values);
    free(plan->ghost_values);
    cudaFree(plan->d_send_local_indices);
    cudaFree(plan->d_send_values);
    free(plan->requests);
    memset(plan, 0, sizeof(*plan));
}

static void prefix_displacements(const int *counts, int n, int *displs) {
    int offset = 0;
    for (int i = 0; i < n; i++) {
        displs[i] = offset;
        offset += counts[i];
    }
}

static int find_ghost_compact_index(const DistributedXPlan *plan,
                                    int global_col,
                                    int owner) {
    const int first = plan->recv_displs[owner];
    const int last = first + plan->recv_counts[owner];
    const int *begin = plan->recv_global_cols + first;
    const int *end = plan->recv_global_cols + last;
    const int *found = std::lower_bound(begin, end, global_col);

    if (found == end || *found != global_col) {
        return -1;
    }

    /*
     * Global-to-local column-index mapping:
     *   owned x_j  -> j / P
     *   ghost x_j  -> owned_count + position in grouped ghost list
     */
    return plan->owned_count + (int)(found - plan->recv_global_cols);
}

static int build_distributed_x_plan(const CSR_Matrix *csr,
                                    int global_cols,
                                    int rank,
                                    int size,
                                    DistributedXPlan *plan,
                                    MPI_Comm comm) {
    memset(plan, 0, sizeof(*plan));
    plan->owned_count = owned_col_count(global_cols, rank, size);

    std::vector<int> ghost_cols;
    ghost_cols.reserve((size_t)csr->nnz);
    for (int i = 0; i < csr->nnz; i++) {
        const int global_col = csr->col_idx[i];
        const int owner = owner_of_vector_entry(global_col, size);
        if (owner != rank) {
            ghost_cols.push_back(global_col);
        }
    }

    std::sort(ghost_cols.begin(), ghost_cols.end(),
              [size](int a, int b) {
                  const int owner_a = owner_of_vector_entry(a, size);
                  const int owner_b = owner_of_vector_entry(b, size);
                  if (owner_a != owner_b) {
                      return owner_a < owner_b;
                  }
                  return a < b;
              });
    ghost_cols.erase(std::unique(ghost_cols.begin(), ghost_cols.end()),
                     ghost_cols.end());

    plan->ghost_count = (int)ghost_cols.size();
    plan->compact_count = plan->owned_count + plan->ghost_count;
    plan->recv_counts = (int *)calloc((size_t)size, sizeof(int));
    plan->recv_displs = (int *)malloc((size_t)size * sizeof(int));
    plan->send_counts = (int *)calloc((size_t)size, sizeof(int));
    plan->send_displs = (int *)malloc((size_t)size * sizeof(int));

    int local_error = 0;
    if (!plan->recv_counts || !plan->recv_displs || !plan->send_counts ||
        !plan->send_displs) {
        local_error = 1;
    }
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free_distributed_x_plan(plan);
        return 1;
    }

    if (plan->ghost_count > 0) {
        plan->recv_global_cols =
            (int *)malloc((size_t)plan->ghost_count * sizeof(int));
        if (!plan->recv_global_cols) {
            local_error = 1;
        }
    }
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free_distributed_x_plan(plan);
        return 1;
    }

    if (plan->ghost_count > 0) {
        for (int i = 0; i < plan->ghost_count; i++) {
            const int global_col = ghost_cols[(size_t)i];
            plan->recv_global_cols[i] = global_col;
            plan->recv_counts[owner_of_vector_entry(global_col, size)]++;
        }
    }
    prefix_displacements(plan->recv_counts, size, plan->recv_displs);

    /*
     * Each rank knows the x_j values it must receive, because it inspected its
     * own CSR column indices. The owner ranks do not know that yet.
     *
     * MPI_Alltoall transposes only the counts:
     *   recv_counts[owner] = how many x_j values I need from owner
     *   send_counts[rank]  = how many owned x_j values that rank needs from me
     */
    MPI_Alltoall(plan->recv_counts, 1, MPI_INT,
                 plan->send_counts, 1, MPI_INT, comm);
    prefix_displacements(plan->send_counts, size, plan->send_displs);

    int total_send = 0;
    for (int i = 0; i < size; i++) {
        total_send += plan->send_counts[i];
    }

    if (total_send > 0) {
        plan->send_global_cols = (int *)malloc((size_t)total_send * sizeof(int));
        plan->send_values = (float *)malloc((size_t)total_send * sizeof(float));
        if (!plan->send_global_cols || !plan->send_values) {
            local_error = 1;
        }
    }
    if (plan->ghost_count > 0) {
        plan->ghost_values = (float *)malloc((size_t)plan->ghost_count * sizeof(float));
        if (!plan->ghost_values) {
            local_error = 1;
        }
    }
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        free_distributed_x_plan(plan);
        return 1;
    }

    /*
     * MPI_Alltoallv transposes the actual requested global column IDs.
     *
     * After this one-time metadata exchange, every owner rank has a permanent
     * send list:
     *   send_global_cols[send_displs[dest] ...] are the owned x_j values that
     *   dest needs on every SpMV iteration.
     *
     * The matrix structure is fixed, so these column ID lists do not change.
     * Only the floating-point x_j values are refreshed each iteration.
     */
    MPI_Alltoallv(plan->recv_global_cols, plan->recv_counts, plan->recv_displs,
                  MPI_INT, plan->send_global_cols, plan->send_counts,
                  plan->send_displs, MPI_INT, comm);

    int request_capacity = 0;
    for (int r = 0; r < size; r++) {
        request_capacity += (plan->recv_counts[r] > 0);
        request_capacity += (plan->send_counts[r] > 0);
    }
    if (request_capacity > 0) {
        plan->requests =
            (MPI_Request *)malloc((size_t)request_capacity * sizeof(MPI_Request));
        if (!plan->requests) {
            free_distributed_x_plan(plan);
            return 1;
        }
    }

    return 0;
}

static int remap_csr_columns_to_distributed_x(CSR_Matrix *csr,
                                              const DistributedXPlan *plan,
                                              int rank,
                                              int size) {
    for (int i = 0; i < csr->nnz; i++) {
        const int global_col = csr->col_idx[i];
        const int owner = owner_of_vector_entry(global_col, size);
        if (owner == rank) {
            csr->col_idx[i] = owned_vector_local_index(global_col, size);
        } else {
            const int compact_col =
                find_ghost_compact_index(plan, global_col, owner);
            if (compact_col < 0) {
                return 1;
            }
            csr->col_idx[i] = compact_col;
        }
    }
    csr->cols = plan->compact_count;
    return 0;
}

static void fill_send_ghost_values(const DistributedXPlan *plan,
                                   const float *x_owned,
                                   int size) {
    int total_send = 0;
    for (int r = 0; r < size; r++) {
        total_send += plan->send_counts[r];
    }

    for (int i = 0; i < total_send; i++) {
        const int global_col = plan->send_global_cols[i];
        plan->send_values[i] =
            x_owned[owned_vector_local_index(global_col, size)];
    }
}

static int distributed_x_total_send(const DistributedXPlan *plan, int size) {
    int total_send = 0;
    for (int r = 0; r < size; r++) {
        total_send += plan->send_counts[r];
    }
    return total_send;
}

static void prepare_cuda_aware_x_plan(DistributedXPlan *plan,
                                      int size,
                                      int rank) {
    const int total_send = distributed_x_total_send(plan, size);
    if (total_send == 0) {
        return;
    }

    std::vector<int> send_local_indices((size_t)total_send);
    for (int i = 0; i < total_send; i++) {
        send_local_indices[(size_t)i] =
            owned_vector_local_index(plan->send_global_cols[i], size);
    }

    CHECK_CUDA(cudaMalloc((void **)&plan->d_send_local_indices,
                          (size_t)total_send * sizeof(int)),
               rank);
    CHECK_CUDA(cudaMalloc((void **)&plan->d_send_values,
                          (size_t)total_send * sizeof(float)),
               rank);
    CHECK_CUDA(cudaMemcpy(plan->d_send_local_indices,
                          send_local_indices.data(),
                          (size_t)total_send * sizeof(int),
                          cudaMemcpyHostToDevice),
               rank);
}

static __global__ void pack_ghost_send_values(const float *x_owned,
                                               const int *send_local_indices,
                                               float *send_values,
                                               int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        send_values[i] = x_owned[send_local_indices[i]];
    }
}

static double exchange_ghost_values(DistributedXPlan *plan,
                                    const float *x_owned,
                                    float *d_x,
                                    int size,
                                    int cuda_aware_mpi,
                                    int rank,
                                    MPI_Comm comm) {
    /*
     * Reuse the precomputed send_global_cols list to pack the current owned
     * values requested by other ranks. No rank re-inspects the matrix here.
     */
    if (cuda_aware_mpi) {
        const int total_send = distributed_x_total_send(plan, size);
        if (total_send > 0) {
            const int threads = 256;
            const int blocks = (total_send + threads - 1) / threads;
            pack_ghost_send_values<<<blocks, threads>>>(
                d_x, plan->d_send_local_indices, plan->d_send_values,
                total_send);
            CHECK_CUDA(cudaGetLastError(), rank);
            CHECK_CUDA(cudaDeviceSynchronize(), rank);
        }
    } else {
        fill_send_ghost_values(plan, x_owned, size);
    }

    const double start = MPI_Wtime();
    int request_count = 0;

    for (int owner = 0; owner < size; owner++) {
        if (plan->recv_counts[owner] > 0) {
            float *recv_buffer =
                cuda_aware_mpi
                    ? d_x + plan->owned_count + plan->recv_displs[owner]
                    : plan->ghost_values + plan->recv_displs[owner];
            MPI_Irecv(recv_buffer,
                      plan->recv_counts[owner], MPI_FLOAT, owner, 4200, comm,
                      &plan->requests[request_count++]);
        }
    }

    for (int dest = 0; dest < size; dest++) {
        if (plan->send_counts[dest] > 0) {
            float *send_buffer =
                cuda_aware_mpi
                    ? plan->d_send_values + plan->send_displs[dest]
                    : plan->send_values + plan->send_displs[dest];
            MPI_Isend(send_buffer,
                      plan->send_counts[dest], MPI_FLOAT, dest, 4200, comm,
                      &plan->requests[request_count++]);
        }
    }

    if (request_count > 0) {
        MPI_Waitall(request_count, plan->requests, MPI_STATUSES_IGNORE);
    }

    return MPI_Wtime() - start;
}

static int scatter_owned_x(const float *x_reference,
                           float *x_owned,
                           int cols,
                           int rank,
                           int size,
                           MPI_Comm comm) {
    int *counts = NULL;
    int *displs = NULL;
    float *send_buffer = NULL;
    int local_error = 0;

    if (rank == 0) {
        counts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        send_buffer = cols > 0 ? (float *)malloc((size_t)cols * sizeof(float)) : NULL;
        if (!counts || !displs || (cols > 0 && !send_buffer)) {
            local_error = 1;
        } else {
            build_owned_row_counts(cols, size, counts, displs);
            for (int owner = 0; owner < size; owner++) {
                for (int local_col = 0; local_col < counts[owner]; local_col++) {
                    const int global_col = owner + local_col * size;
                    send_buffer[displs[owner] + local_col] =
                        x_reference[global_col];
                }
            }
        }
    }

    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (!any_error) {
        MPI_Scatterv(send_buffer, counts, displs, MPI_FLOAT,
                     x_owned, owned_col_count(cols, rank, size), MPI_FLOAT,
                     0, comm);
    }

    free(counts);
    free(displs);
    free(send_buffer);
    return any_error;
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
    /*
     * MPI_Gatherv stores local y chunks grouped by owner rank:
     *   [rank 0 rows][rank 1 rows][rank 2 rows]...
     *
     * Rows are cyclically owned, so rank owner local row k is global row
     * owner + k * P. Rank 0 uses this mapping only after the timed SpMV work to
     * rebuild global y for validation/checksum printing.
     */
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
                           XVectorMode x_mode,
                           DistributedXPlan *x_plan,
                           const float *x_owned,
                           float *d_x,
                           float *d_y,
                           float *d_gathered_y,
                           float *y_local,
                           float *gathered_y,
                           float *y,
                           const float *x_reference,
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
                           int cuda_aware_mpi,
                           double scatter_seconds,
                           double convert_seconds,
                           double h2d_seconds,
                           double file_parse_seconds,
                           FILE *csv,
                           MPI_Comm comm) {
    const char *kernel_name = spmv_kernel_kind_name(kind);
    double prep_seconds = 0.0;
    double merge_seconds = 0.0;
    double *total_times = rank == 0 ? (double *)malloc((size_t)reps * sizeof(double)) : NULL;
    double *comm_times = rank == 0 ? (double *)malloc((size_t)reps * sizeof(double)) : NULL;
    double *compute_times = rank == 0 ? (double *)malloc((size_t)reps * sizeof(double)) : NULL;
    if (rank == 0 && (!total_times || !comm_times || !compute_times)) {
        free(total_times);
        free(comm_times);
        free(compute_times);
        abort_all("Error allocating SpMV timing arrays", rank);
    }

    SpmvKernelRunner *runner = NULL;
    if (local_rows > 0) {
        runner = create_spmv_kernel_runner(kind, rank);
        if (!runner) {
            abort_all("Error creating SpMV kernel runner", rank);
        }

        prep_seconds = runner->prep(*csr, *d_csr, d_x, d_y);
    }

    for (int r = 0; r < warmup; r++) {
        if (x_mode == X_MODE_DISTRIBUTED) {
            exchange_ghost_values(x_plan, x_owned, d_x, size,
                                  cuda_aware_mpi, rank, comm);
            if (!cuda_aware_mpi && x_plan->ghost_count > 0) {
                CHECK_CUDA(cudaMemcpy(d_x + x_plan->owned_count,
                                      x_plan->ghost_values,
                                      (size_t)x_plan->ghost_count *
                                          sizeof(float),
                                      cudaMemcpyHostToDevice),
                           rank);
            }
        }
        if (local_rows > 0) {
            runner->launch(*d_csr, d_x, d_y);
        }
    }
    if (local_rows > 0) {
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
        double local_comm_seconds = 0.0;
        double local_compute_seconds = 0.0;
        const double local_total_start = MPI_Wtime();

        if (x_mode == X_MODE_DISTRIBUTED) {
            /*
             * Communication of ghost values:
             * the structure-derived send/receive lists are reused every
             * iteration; only the requested x_j values are exchanged.
             */
            local_comm_seconds =
                exchange_ghost_values(x_plan, x_owned, d_x, size,
                                      cuda_aware_mpi, rank, comm);

            if (!cuda_aware_mpi && x_plan->ghost_count > 0) {
                CHECK_CUDA(cudaMemcpy(d_x + x_plan->owned_count,
                                      x_plan->ghost_values,
                                      (size_t)x_plan->ghost_count *
                                          sizeof(float),
                                      cudaMemcpyHostToDevice),
                           rank);
            }
        }

        if (local_rows > 0) {
            CHECK_CUDA(cudaEventRecord(event_start), rank);
            runner->launch(*d_csr, d_x, d_y);
            CHECK_CUDA(cudaEventRecord(event_stop), rank);
            CHECK_CUDA(cudaEventSynchronize(event_stop), rank);
            float milliseconds = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&milliseconds, event_start, event_stop),
                       rank);
            local_compute_seconds = (double)milliseconds / 1000.0;
        }
        const double local_total_seconds = MPI_Wtime() - local_total_start;

        double global_total_seconds = 0.0;
        double global_comm_seconds = 0.0;
        double global_compute_seconds = 0.0;
        MPI_Reduce(&local_total_seconds, &global_total_seconds, 1,
                   MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&local_comm_seconds, &global_comm_seconds, 1,
                   MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&local_compute_seconds, &global_compute_seconds, 1,
                   MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            total_times[r] = global_total_seconds;
            comm_times[r] = global_comm_seconds;
            compute_times[r] = global_compute_seconds;
        }
    }

    if (x_mode == X_MODE_DISTRIBUTED) {
        exchange_ghost_values(x_plan, x_owned, d_x, size,
                              cuda_aware_mpi, rank, comm);
        if (!cuda_aware_mpi && x_plan->ghost_count > 0) {
            CHECK_CUDA(cudaMemcpy(d_x + x_plan->owned_count,
                                  x_plan->ghost_values,
                                  (size_t)x_plan->ghost_count * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       rank);
        }
    }
    if (local_rows > 0) {
        runner->launch(*d_csr, d_x, d_y);
        CHECK_CUDA(cudaGetLastError(), rank);
        CHECK_CUDA(cudaDeviceSynchronize(), rank);
        if (!cuda_aware_mpi) {
            CHECK_CUDA(cudaMemcpy(y_local, d_y,
                                  (size_t)local_rows * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       rank);
        }
        CHECK_CUDA(cudaEventDestroy(event_start), rank);
        CHECK_CUDA(cudaEventDestroy(event_stop), rank);
    }

    double merge_start = MPI_Wtime();
    MPI_Gatherv(cuda_aware_mpi ? d_y : y_local, local_rows, MPI_FLOAT,
                cuda_aware_mpi ? d_gathered_y : gathered_y,
                row_counts, row_displs, MPI_FLOAT,
                0, comm);

    if (rank == 0) {
        if (cuda_aware_mpi && global_rows > 0) {
            CHECK_CUDA(cudaMemcpy(gathered_y, d_gathered_y,
                                  (size_t)global_rows * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       rank);
        }
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
        snprintf(stats.implementation, sizeof(stats.implementation),
                 "MPI %s-x %s%s", x_vector_mode_short_name(x_mode),
                 kernel_name, cuda_aware_mpi ? " CUDA-aware" : "");
        stats.rows = global_rows;
        stats.cols = global_cols;
        stats.nnz = global_nnz;
        stats.processes = size;
        stats.file_parse_s = file_parse_seconds;
        stats.format_conv_s = max_convert_seconds + max_prep_seconds;
        stats.h2d_transfer_s = max_h2d_seconds;

        compute_time_stats(total_times, reps, &stats.avg_time_s,
                           &stats.std_time_s);
        double unused_std = 0.0;
        compute_time_stats(comm_times, reps, &stats.comm_time_s, &unused_std);
        compute_time_stats(compute_times, reps, &stats.compute_time_s,
                           &unused_std);
        stats.gflops = stats.avg_time_s > 0.0
                           ? (2.0 * (double)global_nnz) /
                                 (stats.avg_time_s * 1e9)
                           : 0.0;
        validate_vs_reference(reference_csr, x_reference, y, &stats);

        perf_stats_write_csv_row(csv, &stats);

        printf("[%s] avg=%.9f s | comm=%.9f s | compute=%.9f s | GFLOP/s=%.6f | std=%.9f s\n",
               kernel_name, stats.avg_time_s, stats.comm_time_s,
               stats.compute_time_s, stats.gflops, stats.std_time_s);
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
    free(total_times);
    free(comm_times);
    free(compute_times);
}

static int run_matrix(const char *matrix_path,
                      const SpmvKernelKind *kernel_kinds,
                      int kernel_count,
                      const char *kernel_name,
                      XVectorMode x_mode,
                      int reps,
                      int warmup,
                      int cuda_aware_mpi,
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
    int local_x_count = owned_col_count(local.cols, rank, size);
    float *x_reference =
        rank == 0 && local.cols > 0
            ? (float *)malloc((size_t)local.cols * sizeof(float))
            : NULL;
    float *x_owned =
        x_mode == X_MODE_DISTRIBUTED && local_x_count > 0
            ? (float *)malloc((size_t)local_x_count * sizeof(float))
            : NULL;
    float *x_replicated =
        x_mode == X_MODE_REPLICATED && local.cols > 0
            ? (float *)malloc((size_t)local.cols * sizeof(float))
            : NULL;
    float *y_local =
        !cuda_aware_mpi && local_rows > 0
            ? (float *)malloc((size_t)local_rows * sizeof(float))
            : NULL;
    float *y = rank == 0 ? (float *)malloc((size_t)local.rows * sizeof(float)) : NULL;
    float *gathered_y = rank == 0 ? (float *)malloc((size_t)local.rows * sizeof(float)) : NULL;
    int allocation_error = ((rank == 0 && local.cols > 0 && !x_reference) ||
                            (x_mode == X_MODE_DISTRIBUTED &&
                             local_x_count > 0 && !x_owned) ||
                            (x_mode == X_MODE_REPLICATED &&
                             local.cols > 0 && !x_replicated) ||
                            (!cuda_aware_mpi && local_rows > 0 && !y_local) ||
                            (rank == 0 && (!y || !gathered_y)));
    MPI_Allreduce(&allocation_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        free(x_reference);
        free(x_owned);
        free(x_replicated);
        free(y_local);
        free(y);
        free(gathered_y);
        free_local_coo(&local);
        if (rank == 0) {
            free_csr(&reference_csr);
            free_coo(&global);
        }
        abort_all("Error allocating dense vectors", rank);
    }

    if (rank == 0) {
        srand(0);
        fill_dense(x_reference, (size_t)local.cols);
    }
    if (x_mode == X_MODE_DISTRIBUTED &&
        scatter_owned_x(x_reference, x_owned, local.cols, rank, size,
                        MPI_COMM_WORLD) != 0) {
        free(x_reference);
        free(x_owned);
        free(x_replicated);
        free(y_local);
        free(y);
        free(gathered_y);
        free_local_coo(&local);
        if (rank == 0) {
            free_csr(&reference_csr);
            free_coo(&global);
        }
        abort_all("Error distributing owned x values", rank);
    }
    if (x_mode == X_MODE_REPLICATED) {
        if (rank == 0 && local.cols > 0) {
            memcpy(x_replicated, x_reference,
                   (size_t)local.cols * sizeof(float));
        }
        MPI_Bcast(x_replicated, local.cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    double convert_start = MPI_Wtime();
    COO_Matrix local_as_coo;
    int remap_error = create_local_csr_coo(&local, rank, size, local_rows,
                                           &local_as_coo);
    MPI_Allreduce(&remap_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        free(local_as_coo.row);
        free(x_reference);
        free(x_owned);
        free(x_replicated);
        free(y_local);
        free(y);
        free(gathered_y);
        free_local_coo(&local);
        if (rank == 0) {
            free_csr(&reference_csr);
            free_coo(&global);
        }
        abort_all("Error remapping cyclic rows to local CSR rows", rank);
    }

    CSR_Matrix csr = {0};
    coo_to_csr(&local_as_coo, &csr);
    free(local_as_coo.row);

    DistributedXPlan x_plan;
    memset(&x_plan, 0, sizeof(x_plan));
    int plan_error = 0;
    if (x_mode == X_MODE_DISTRIBUTED) {
        plan_error = build_distributed_x_plan(&csr, local.cols, rank, size,
                                              &x_plan, MPI_COMM_WORLD);
        if (!plan_error) {
            plan_error = remap_csr_columns_to_distributed_x(&csr, &x_plan, rank,
                                                            size);
        }
    }
    MPI_Allreduce(&plan_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) {
        free_distributed_x_plan(&x_plan);
        free_csr(&csr);
        free(x_reference);
        free(x_owned);
        free(x_replicated);
        free(y_local);
        free(y);
        free(gathered_y);
        free_local_coo(&local);
        if (rank == 0) {
            free_csr(&reference_csr);
            free_coo(&global);
        }
        abort_all("Error building distributed x ghost plan", rank);
    }
    double convert_seconds = MPI_Wtime() - convert_start;

    CSR_Matrix d_csr = {0};
    float *d_x = NULL;
    float *d_y = NULL;
    float *d_gathered_y = NULL;

    double h2d_start = MPI_Wtime();
    csr_to_device(&csr, &d_csr, rank);
    CHECK_CUDA(cudaDeviceSynchronize(), rank);

    const int device_x_count = x_mode == X_MODE_DISTRIBUTED
                                   ? x_plan.compact_count
                                   : local.cols;
    if (device_x_count > 0) {
        CHECK_CUDA(cudaMalloc((void **)&d_x,
                              (size_t)device_x_count * sizeof(float)),
                   rank);
        if (x_mode == X_MODE_DISTRIBUTED && x_plan.owned_count > 0) {
            CHECK_CUDA(cudaMemcpy(d_x, x_owned,
                                  (size_t)x_plan.owned_count * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       rank);
        } else if (x_mode == X_MODE_REPLICATED) {
            CHECK_CUDA(cudaMemcpy(d_x, x_replicated,
                                  (size_t)local.cols * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       rank);
        }
    }
    if (local_rows > 0) {
        CHECK_CUDA(cudaMalloc((void **)&d_y, (size_t)local_rows * sizeof(float)),
                   rank);
    }
    if (cuda_aware_mpi && rank == 0 && local.rows > 0) {
        CHECK_CUDA(cudaMalloc((void **)&d_gathered_y,
                              (size_t)local.rows * sizeof(float)),
                   rank);
    }
    if (cuda_aware_mpi && x_mode == X_MODE_DISTRIBUTED) {
        prepare_cuda_aware_x_plan(&x_plan, size, rank);
    }
    CHECK_CUDA(cudaDeviceSynchronize(), rank);
    double h2d_seconds = MPI_Wtime() - h2d_start;

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
        printf("Dense x mode: %s\n", x_vector_mode_name(x_mode));
        printf("MPI buffer mode: %s\n",
               cuda_aware_mpi ? "CUDA-aware device buffers"
                              : "host-staged buffers");
        if (x_mode == X_MODE_DISTRIBUTED) {
            printf("Distributed x ownership: owner(j)=j%%%d; no full-vector Allgather/Bcast is used for SpMV\n",
                   size);
        } else {
            printf("Replicated x: every rank stores all %d dense-vector values\n",
                   local.cols);
        }
        printf("Selected kernel: %s\n", kernel_name);
        printf("Warmup repetitions: %d, measured repetitions: %d\n",
               warmup, reps);
        printf("Writing CSV metrics to %s\n", perf_stats_resolve_path(csv_path));
    }

    for (int i = 0; i < kernel_count; i++) {
        execute_kernel(kernel_kinds[i], &csr, &d_csr, &reference_csr, x_mode,
                       &x_plan, x_owned, d_x, d_y, d_gathered_y,
                       y_local, gathered_y, y, x_reference, local_rows, local.rows,
                       local.cols, local.global_nnz, rank, size,
                       row_counts, row_displs, matrix_basename(matrix_path),
                       reps, warmup, cuda_aware_mpi,
                       scatter_seconds, convert_seconds,
                       h2d_seconds, read_seconds, csv, MPI_COMM_WORLD);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gathered_y);
    free_device_csr(&d_csr);
    free_distributed_x_plan(&x_plan);
    free_csr(&csr);
    free(row_counts);
    free(row_displs);
    free(x_reference);
    free(x_owned);
    free(x_replicated);
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
    int cuda_aware_mpi = 0;
    XVectorMode x_mode = X_MODE_DISTRIBUTED;
    if (parse_args(argc, argv, &kernel_name, &reps, &warmup, &csv_path,
                   &x_mode, &cuda_aware_mpi) != 0) {
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
                       x_mode, reps, warmup, cuda_aware_mpi, csv, csv_path,
                       rank, size) != 0) {
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
