/*
 * Weak-scaling driver for the existing CUDA-aware MPI SpMV implementation.
 *
 * The strong-scaling driver is included in library mode so this executable
 * uses exactly the same ownership helpers, ghost exchange plan, CUDA-aware
 * communication path, CSR device transfer, and kernel runners.  Its main()
 * and file-oriented benchmark workflow are not part of this executable.
 */
#define SPMV_MPI_CUDA_LIBRARY_ONLY
#include "spmv_mpi_cuda.cu"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <sys/stat.h>

#include <limits>

typedef struct {
    const char *kernel_name;
    const char *csv_path;
    const char *baseline_path;
    int rows_per_rank;
    int nnz_per_row;
    int reps;
    int warmup;
    int validation_max_rows;
    unsigned long long seed;
    XVectorMode x_mode;
    int cuda_aware_mpi;
    int use_nccl;
} WeakArgs;

static const char *weak_csv_header(void) {
    return "implementation,matrix,kernel,x_mode,cuda_aware_mpi,nccl,processes,"
           "rows_per_rank,nnz_per_row,global_rows,global_cols,global_nnz,"
           "density,seed,warmup,reps,max_rank_time_s,std_max_rank_time_s,"
           "compute_time_s,communication_time_s,gflops,"
           "weak_scaling_efficiency,nnz_rank_min,nnz_rank_avg,nnz_rank_max,"
           "ghost_recv_values_rank_min,ghost_recv_values_rank_avg,"
           "ghost_recv_values_rank_max,ghost_send_values_rank_min,"
           "ghost_send_values_rank_avg,ghost_send_values_rank_max,"
           "communication_bytes_rank_min,communication_bytes_rank_avg,"
           "communication_bytes_rank_max,communication_payload_bytes_total,"
           "host_memory_bytes_rank_min,host_memory_bytes_rank_avg,"
           "host_memory_bytes_rank_max,device_memory_bytes_rank_min,"
           "device_memory_bytes_rank_avg,device_memory_bytes_rank_max,"
           "memory_bytes_rank_min,memory_bytes_rank_avg,memory_bytes_rank_max,"
           "validation_performed,valid,max_abs_error,generation_time_s,"
           "setup_time_s";
}

static void weak_usage(const char *program, FILE *stream) {
    fprintf(stream,
            "Usage: %s [--kernel name] [--rows-per-rank N] "
            "[--nnz-per-row N]\n"
            "          [--reps N] [--warmup N] [--seed N] "
            "[--x-mode block|cyclic|replicated]\n"
            "          [--cuda-aware-mpi | --nccl] [--validation-max-rows N] "
            "[--output path] [--baseline-file path]\n"
            "\n"
            "Each rank directly generates an exact rows-per-rank by "
            "nnz-per-row local CSR slice.\n",
            program);
    print_spmv_kernel_choices(stream);
}

static int parse_positive_int(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || !end || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        return 1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_nonnegative_int(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || !end || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return 1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_seed(const char *text, unsigned long long *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || !end || *end != '\0') {
        return 1;
    }
    *value = parsed;
    return 0;
}

static int take_value_option(int argc,
                             char **argv,
                             int *index,
                             const char *name,
                             const char **value) {
    const size_t length = strlen(name);
    if (strcmp(argv[*index], name) == 0) {
        if (*index + 1 >= argc) return -1;
        *value = argv[++(*index)];
        return 1;
    }
    if (strncmp(argv[*index], name, length) == 0 &&
        argv[*index][length] == '=') {
        *value = argv[*index] + length + 1;
        return **value != '\0' ? 1 : -1;
    }
    return 0;
}

static int parse_weak_args(int argc, char **argv, WeakArgs *args) {
    args->kernel_name = "adaptive";
    args->csv_path = "results/mpi_spmv_weak_scaling.csv";
    args->baseline_path = NULL;
    args->rows_per_rank = 262144;
    args->nnz_per_row = 32;
    args->reps = 50;
    args->warmup = 5;
    args->validation_max_rows = 65536;
    args->seed = 20260721ULL;
    args->x_mode = X_MODE_DISTRIBUTED_BLOCK;
    args->cuda_aware_mpi = 0;
    args->use_nccl = 0;

    for (int i = 1; i < argc; i++) {
        const char *value = NULL;
        if (strcmp(argv[i], "--cuda-aware-mpi") == 0) {
            if (args->use_nccl) return 1;
            args->cuda_aware_mpi = 1;
            continue;
        }
        if (strcmp(argv[i], "--nccl") == 0) {
            if (args->cuda_aware_mpi) return 1;
            args->use_nccl = 1;
            continue;
        }

        int matched = take_value_option(argc, argv, &i, "--kernel", &value);
        if (matched < 0) return 1;
        if (matched) {
            args->kernel_name = value;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--rows-per-rank", &value)) != 0) {
            if (matched < 0) return 1;
            if (parse_positive_int(value, &args->rows_per_rank)) return 1;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--nnz-per-row", &value)) != 0) {
            if (matched < 0) return 1;
            if (parse_positive_int(value, &args->nnz_per_row)) return 1;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--reps", &value)) != 0) {
            if (matched < 0) return 1;
            if (parse_positive_int(value, &args->reps)) return 1;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--warmup", &value)) != 0) {
            if (matched < 0) return 1;
            if (parse_nonnegative_int(value, &args->warmup)) return 1;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--validation-max-rows", &value)) !=
                   0) {
            if (matched < 0) return 1;
            if (parse_nonnegative_int(value, &args->validation_max_rows)) {
                return 1;
            }
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--seed", &value)) != 0) {
            if (matched < 0) return 1;
            if (parse_seed(value, &args->seed)) return 1;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--x-mode", &value)) != 0) {
            if (matched < 0) return 1;
            if (parse_x_vector_mode(value, &args->x_mode)) return 1;
            if (args->x_mode != X_MODE_DISTRIBUTED_BLOCK &&
                args->x_mode != X_MODE_DISTRIBUTED_CYCLIC &&
                args->x_mode != X_MODE_REPLICATED) return 1;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--output", &value)) != 0) {
            if (matched < 0) return 1;
            args->csv_path = value;
        } else if ((matched = take_value_option(
                        argc, argv, &i, "--baseline-file", &value)) != 0) {
            if (matched < 0) return 1;
            args->baseline_path = value;
        } else {
            return 1;
        }
    }
    return 0;
}

static uint64_t splitmix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static int integer_gcd(int a, int b) {
    while (b != 0) {
        const int remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

static float generated_value(uint64_t key) {
    const uint32_t bits = (uint32_t)(splitmix64(key) >> 40);
    return 2.0f * ((float)bits / 16777215.0f) - 1.0f;
}

static float generated_x_value(int global_col, unsigned long long seed) {
    return generated_value((uint64_t)seed ^
                           (UINT64_C(0xd6e8feb86659fd93) *
                            (uint64_t)(global_col + 1)));
}

static int local_global_row(int local_row,
                            int global_rows,
                            int rank,
                            int size,
                            XVectorMode mode) {
    const COORowPartition partition = row_partition_for_mode(mode);
    if (partition == COO_ROW_PARTITION_BLOCK) {
        return coo_partition_first_index(global_rows, rank, size, partition) +
               local_row;
    }
    return rank + local_row * size;
}

/*
 * A row gets a pseudorandom start and a pseudorandom modular stride. Making
 * the stride coprime with the global column count guarantees that the first K
 * positions are distinct, without a rejection table or any global matrix.
 */
static int generate_local_csr(CSR_Matrix *csr,
                              std::vector<double> *cpu_reference,
                              int global_rows,
                              int rows_per_rank,
                              int nnz_per_row,
                              int rank,
                              int size,
                              XVectorMode mode,
                              unsigned long long seed) {
    const long long local_nnz_ll =
        (long long)rows_per_rank * (long long)nnz_per_row;
    if (local_nnz_ll > INT_MAX || nnz_per_row > global_rows) {
        return 1;
    }

    csr->rows = rows_per_rank;
    csr->cols = global_rows;
    csr->nnz = (int)local_nnz_ll;
    csr->row_ptr = (int *)malloc((size_t)(rows_per_rank + 1) * sizeof(int));
    csr->col_idx = (int *)malloc((size_t)csr->nnz * sizeof(int));
    csr->values = (float *)malloc((size_t)csr->nnz * sizeof(float));
    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        free_csr(csr);
        return 1;
    }

    if (cpu_reference) {
        cpu_reference->assign((size_t)rows_per_rank, 0.0);
    }

    int offset = 0;
    for (int local_row = 0; local_row < rows_per_rank; local_row++) {
        csr->row_ptr[local_row] = offset;
        const int global_row = local_global_row(
            local_row, global_rows, rank, size, mode);
        const uint64_t row_key =
            splitmix64((uint64_t)seed ^
                       (UINT64_C(0xa0761d6478bd642f) *
                        (uint64_t)(global_row + 1)));
        const int start = (int)(splitmix64(row_key) % (uint64_t)global_rows);

        int stride = global_rows == 1
                         ? 1
                         : (int)(splitmix64(row_key ^
                                           UINT64_C(0xe7037ed1a0b428db)) %
                                 (uint64_t)global_rows);
        if (stride == 0) stride = 1;
        while (integer_gcd(stride, global_rows) != 1) {
            stride++;
            if (stride == global_rows) stride = 1;
        }

        double expected = 0.0;
        for (int entry = 0; entry < nnz_per_row; entry++) {
            const int global_col =
                (int)(((uint64_t)start +
                       (uint64_t)entry * (uint64_t)stride) %
                      (uint64_t)global_rows);
            const float value = generated_value(
                row_key ^ (UINT64_C(0x8ebc6af09c88c6e3) *
                           (uint64_t)(entry + 1)));
            csr->col_idx[offset] = global_col;
            csr->values[offset] = value;
            if (cpu_reference) {
                expected += (double)value *
                            (double)generated_x_value(global_col, seed);
            }
            offset++;
        }
        if (cpu_reference) {
            (*cpu_reference)[(size_t)local_row] = expected;
        }
    }
    csr->row_ptr[rows_per_rank] = offset;
    return 0;
}

static void fill_owned_x(float *x,
                         int count,
                         int global_cols,
                         int rank,
                         int size,
                         XVectorMode mode,
                         unsigned long long seed) {
    for (int local_col = 0; local_col < count; local_col++) {
        int global_col = local_col;
        if (mode == X_MODE_DISTRIBUTED_BLOCK) {
            global_col = coo_partition_first_index(
                             global_cols, rank, size,
                             COO_ROW_PARTITION_BLOCK) +
                         local_col;
        } else if (mode == X_MODE_DISTRIBUTED_CYCLIC) {
            global_col = rank + local_col * size;
        }
        x[local_col] = generated_x_value(global_col, seed);
    }
}

static FILE *open_weak_csv(const char *path) {
    mkdir("results", 0755);
    struct stat info;
    const int write_header = stat(path, &info) != 0 || info.st_size == 0;
    if (!write_header) {
        FILE *existing = fopen(path, "r");
        char header[8192];
        if (!existing || !fgets(header, sizeof(header), existing)) {
            if (existing) fclose(existing);
            return NULL;
        }
        fclose(existing);
        header[strcspn(header, "\r\n")] = '\0';
        if (strcmp(header, weak_csv_header()) != 0) {
            fprintf(stderr,
                    "Error: weak-scaling CSV schema in '%s' does not match; "
                    "use a new output path\n",
                    path);
            return NULL;
        }
    }

    FILE *csv = fopen(path, "a");
    if (!csv) return NULL;
    if (write_header) fprintf(csv, "%s\n", weak_csv_header());
    return csv;
}

static double mean(const std::vector<double> &values) {
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); i++) sum += values[i];
    return sum / (double)values.size();
}

static double standard_deviation(const std::vector<double> &values,
                                 double average) {
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); i++) {
        const double difference = values[i] - average;
        sum += difference * difference;
    }
    return sqrt(sum / (double)values.size());
}

static unsigned long long estimate_host_bytes(
    const CSR_Matrix *csr,
    const DistributedXPlan *plan,
    int x_host_count,
    int validation_performed,
    int rank_count) {
    unsigned long long bytes =
        (unsigned long long)(csr->rows + 1) * sizeof(int) +
        (unsigned long long)csr->nnz * (sizeof(int) + sizeof(float)) +
        (unsigned long long)x_host_count * sizeof(float);
    if (validation_performed) {
        bytes += (unsigned long long)csr->rows *
                 (sizeof(double) + sizeof(float));
    }
    if (plan) {
        const int total_send = distributed_x_total_send(plan, rank_count);
        int request_count = 0;
        for (int peer = 0; peer < rank_count; peer++) {
            request_count += plan->recv_counts[peer] > 0;
            request_count += plan->send_counts[peer] > 0;
        }
        bytes += (unsigned long long)4 * (unsigned long long)rank_count *
                 sizeof(int);
        bytes += (unsigned long long)plan->ghost_count *
                 (sizeof(int) + sizeof(float));
        bytes += (unsigned long long)total_send *
                 (sizeof(int) + sizeof(float));
        bytes += (unsigned long long)request_count * sizeof(MPI_Request);
    }
    return bytes;
}

static int baseline_matches(const WeakArgs *args,
                            int rows_per_rank,
                            int nnz_per_row,
                            const char *kernel,
                            const char *mode,
                            const char *stored_kernel,
                            const char *stored_mode,
                            int stored_rows,
                            int stored_nnz,
                            unsigned long long stored_seed,
                            int stored_backend) {
    return strcmp(kernel, stored_kernel) == 0 &&
           strcmp(mode, stored_mode) == 0 && rows_per_rank == stored_rows &&
           nnz_per_row == stored_nnz && args->seed == stored_seed &&
           (int)active_communication_backend == stored_backend;
}

static double load_or_store_baseline(const WeakArgs *args,
                                     int processes,
                                     double measured_time,
                                     int rank) {
    if (processes == 1) {
        if (rank == 0 && args->baseline_path) {
            FILE *file = fopen(args->baseline_path, "w");
            if (!file) {
                fprintf(stderr, "Warning: cannot write baseline file %s\n",
                        args->baseline_path);
            } else {
                fprintf(file, "%s %s %d %d %llu %d %.17g\n",
                        args->kernel_name, x_vector_mode_short_name(args->x_mode),
                        args->rows_per_rank, args->nnz_per_row, args->seed,
                        (int)active_communication_backend, measured_time);
                fclose(file);
            }
        }
        return measured_time;
    }

    if (rank != 0 || !args->baseline_path) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    FILE *file = fopen(args->baseline_path, "r");
    if (!file) {
        fprintf(stderr,
                "Warning: no one-GPU baseline file; efficiency will be NaN\n");
        return std::numeric_limits<double>::quiet_NaN();
    }
    char stored_kernel[64];
    char stored_mode[64];
    int stored_rows = 0;
    int stored_nnz = 0;
    unsigned long long stored_seed = 0;
    int stored_backend = 0;
    double baseline = 0.0;
    const int fields = fscanf(file, "%63s %63s %d %d %llu %d %lf",
                              stored_kernel, stored_mode, &stored_rows,
                              &stored_nnz, &stored_seed, &stored_backend,
                              &baseline);
    fclose(file);
    if (fields != 7 ||
        !baseline_matches(args, args->rows_per_rank, args->nnz_per_row,
                          args->kernel_name,
                          x_vector_mode_short_name(args->x_mode), stored_kernel,
                          stored_mode, stored_rows, stored_nnz, stored_seed,
                          stored_backend)) {
        fprintf(stderr,
                "Warning: one-GPU baseline configuration does not match; "
                "efficiency will be NaN\n");
        return std::numeric_limits<double>::quiet_NaN();
    }
    return baseline;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    WeakArgs args;
    if (parse_weak_args(argc, argv, &args) != 0) {
        if (rank == 0) weak_usage(argv[0], stderr);
        MPI_Finalize();
        return 1;
    }
    if (args.use_nccl && !nccl_backend_is_compiled()) {
        if (rank == 0) {
            fprintf(stderr,
                    "NCCL support is not compiled in; rebuild with "
                    "'make NCCL=1'.\n");
        }
        MPI_Finalize();
        return 1;
    }
    active_communication_backend = args.use_nccl
                                       ? COMM_NCCL
                                       : args.cuda_aware_mpi
                                             ? COMM_CUDA_AWARE_MPI
                                             : COMM_HOST_MPI;
    const int device_communication = communication_uses_device_buffers();

    SpmvKernelKind kernel_kind;
    if (strcmp(args.kernel_name, "all") == 0 ||
        parse_spmv_kernel_kind(args.kernel_name, &kernel_kind) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Select one valid kernel (not 'all').\n");
            print_spmv_kernel_choices(stderr);
        }
        MPI_Finalize();
        return 1;
    }

    const long long global_rows_ll =
        (long long)args.rows_per_rank * (long long)size;
    const long long local_nnz_ll =
        (long long)args.rows_per_rank * (long long)args.nnz_per_row;
    if (global_rows_ll > INT_MAX || local_nnz_ll > INT_MAX ||
        args.nnz_per_row > global_rows_ll) {
        if (rank == 0) {
            fprintf(stderr,
                    "Configuration exceeds the implementation's 32-bit CSR "
                    "index limits.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const int global_rows = (int)global_rows_ll;
    const long long global_nnz = local_nnz_ll * (long long)size;
    const int validation_performed =
        args.validation_max_rows > 0 &&
        global_rows <= args.validation_max_rows;

    select_cuda_device_for_rank(rank, MPI_COMM_WORLD);
    CHECK_CUDA(cudaFree(0), rank);
    initialize_nccl_backend(rank, size, MPI_COMM_WORLD);

    std::vector<double> cpu_reference;
    CSR_Matrix csr = {0};
    MPI_Barrier(MPI_COMM_WORLD);
    const double generation_start = MPI_Wtime();
    int generation_error = generate_local_csr(
        &csr, validation_performed ? &cpu_reference : NULL, global_rows,
        args.rows_per_rank, args.nnz_per_row, rank, size, args.x_mode,
        args.seed);
    int any_error = 0;
    MPI_Allreduce(&generation_error, &any_error, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (any_error) abort_all("Could not generate local CSR matrices", rank);
    const double generation_seconds = MPI_Wtime() - generation_start;

    const double setup_start = MPI_Wtime();
    DistributedXPlan x_plan;
    memset(&x_plan, 0, sizeof(x_plan));
    if (x_is_distributed(args.x_mode)) {
        int plan_error = build_distributed_x_plan(
            &csr, global_rows, rank, size, args.x_mode, &x_plan,
            MPI_COMM_WORLD);
        if (!plan_error) {
            plan_error = remap_csr_columns_to_distributed_x(
                &csr, &x_plan, rank, size);
        }
        MPI_Allreduce(&plan_error, &any_error, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if (any_error) abort_all("Could not build the ghost-vector plan", rank);
    }

    const int x_host_count = x_is_distributed(args.x_mode)
                                 ? x_plan.owned_count
                                 : global_rows;
    float *x_host =
        (float *)malloc((size_t)x_host_count * sizeof(float));
    if (!x_host) abort_all("Could not allocate the local dense vector", rank);
    fill_owned_x(x_host, x_host_count, global_rows, rank, size, args.x_mode,
                 args.seed);

    size_t device_free_before = 0;
    size_t device_total = 0;
    CHECK_CUDA(cudaMemGetInfo(&device_free_before, &device_total), rank);

    CSR_Matrix d_csr = {0};
    csr_to_device(&csr, &d_csr, rank);
    const int device_x_count = x_is_distributed(args.x_mode)
                                   ? x_plan.compact_count
                                   : global_rows;
    float *d_x = NULL;
    float *d_y = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_x,
                          (size_t)device_x_count * sizeof(float)),
               rank);
    CHECK_CUDA(cudaMalloc((void **)&d_y,
                          (size_t)args.rows_per_rank * sizeof(float)),
               rank);
    CHECK_CUDA(cudaMemcpy(d_x, x_host,
                          (size_t)x_host_count * sizeof(float),
                          cudaMemcpyHostToDevice),
               rank);
    if (device_communication && x_is_distributed(args.x_mode)) {
        prepare_cuda_aware_x_plan(&x_plan, size, rank);
    }

    SpmvKernelRunner *runner = create_spmv_kernel_runner(kernel_kind, rank);
    if (!runner) abort_all("Could not create the CUDA kernel runner", rank);
    runner->prep(csr, d_csr, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize(), rank);

    size_t device_free_after = 0;
    CHECK_CUDA(cudaMemGetInfo(&device_free_after, &device_total), rank);
    const unsigned long long device_memory =
        device_free_before > device_free_after
            ? (unsigned long long)(device_free_before - device_free_after)
            : 0ULL;
    const unsigned long long host_memory = estimate_host_bytes(
        &csr, x_is_distributed(args.x_mode) ? &x_plan : NULL, x_host_count,
        validation_performed, size);
    const unsigned long long total_memory = host_memory + device_memory;
    const double setup_seconds = MPI_Wtime() - setup_start;

    for (int iteration = 0; iteration < args.warmup; iteration++) {
        if (x_is_distributed(args.x_mode)) {
            exchange_ghost_values(&x_plan, x_host, d_x, size,
                                  device_communication, rank, MPI_COMM_WORLD);
            if (!device_communication && x_plan.ghost_count > 0) {
                CHECK_CUDA(cudaMemcpy(d_x + x_plan.owned_count,
                                      x_plan.ghost_values,
                                      (size_t)x_plan.ghost_count *
                                          sizeof(float),
                                      cudaMemcpyHostToDevice),
                           rank);
            }
        }
        runner->launch(d_csr, d_x, d_y);
    }
    CHECK_CUDA(cudaGetLastError(), rank);
    CHECK_CUDA(cudaDeviceSynchronize(), rank);

    cudaEvent_t compute_start;
    cudaEvent_t compute_stop;
    CHECK_CUDA(cudaEventCreate(&compute_start), rank);
    CHECK_CUDA(cudaEventCreate(&compute_stop), rank);

    std::vector<double> max_rank_times;
    std::vector<double> max_compute_times;
    std::vector<double> max_communication_times;
    if (rank == 0) {
        max_rank_times.resize((size_t)args.reps);
        max_compute_times.resize((size_t)args.reps);
        max_communication_times.resize((size_t)args.reps);
    }

    for (int iteration = 0; iteration < args.reps; iteration++) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double total_start = MPI_Wtime();
        double local_communication = 0.0;
        if (x_is_distributed(args.x_mode)) {
            const double communication_start = MPI_Wtime();
            exchange_ghost_values(&x_plan, x_host, d_x, size,
                                  device_communication, rank, MPI_COMM_WORLD);
            if (!device_communication && x_plan.ghost_count > 0) {
                CHECK_CUDA(cudaMemcpy(d_x + x_plan.owned_count,
                                      x_plan.ghost_values,
                                      (size_t)x_plan.ghost_count *
                                          sizeof(float),
                                      cudaMemcpyHostToDevice),
                           rank);
            }
            local_communication = MPI_Wtime() - communication_start;
        }

        CHECK_CUDA(cudaEventRecord(compute_start), rank);
        runner->launch(d_csr, d_x, d_y);
        CHECK_CUDA(cudaEventRecord(compute_stop), rank);
        CHECK_CUDA(cudaEventSynchronize(compute_stop), rank);
        float compute_milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&compute_milliseconds, compute_start,
                                        compute_stop),
                   rank);
        const double local_compute =
            (double)compute_milliseconds / 1000.0;
        const double local_total = MPI_Wtime() - total_start;

        double global_total = 0.0;
        double global_compute = 0.0;
        double global_communication = 0.0;
        MPI_Reduce(&local_total, &global_total, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(&local_compute, &global_compute, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        MPI_Reduce(&local_communication, &global_communication, 1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            max_rank_times[(size_t)iteration] = global_total;
            max_compute_times[(size_t)iteration] = global_compute;
            max_communication_times[(size_t)iteration] = global_communication;
        }
    }

    int valid = -1;
    double max_abs_error = std::numeric_limits<double>::quiet_NaN();
    if (validation_performed) {
        if (x_is_distributed(args.x_mode)) {
            exchange_ghost_values(&x_plan, x_host, d_x, size,
                                  device_communication, rank, MPI_COMM_WORLD);
            if (!device_communication && x_plan.ghost_count > 0) {
                CHECK_CUDA(cudaMemcpy(d_x + x_plan.owned_count,
                                      x_plan.ghost_values,
                                      (size_t)x_plan.ghost_count *
                                          sizeof(float),
                                      cudaMemcpyHostToDevice),
                           rank);
            }
        }
        runner->launch(d_csr, d_x, d_y);
        CHECK_CUDA(cudaGetLastError(), rank);
        CHECK_CUDA(cudaDeviceSynchronize(), rank);
        std::vector<float> gpu_y((size_t)args.rows_per_rank);
        CHECK_CUDA(cudaMemcpy(gpu_y.data(), d_y,
                              (size_t)args.rows_per_rank * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   rank);
        int local_valid = 1;
        double local_max_error = 0.0;
        for (int row = 0; row < args.rows_per_rank; row++) {
            const double error = fabs((double)gpu_y[(size_t)row] -
                                      cpu_reference[(size_t)row]);
            local_max_error = fmax(local_max_error, error);
            if (error > 1e-3 *
                            fmax(1.0, fabs(cpu_reference[(size_t)row]))) {
                local_valid = 0;
            }
        }
        MPI_Allreduce(&local_valid, &valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local_max_error, &max_abs_error, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
    }

    const long long local_nnz = local_nnz_ll;
    long long nnz_min = 0;
    long long nnz_max = 0;
    long long nnz_sum = 0;
    MPI_Reduce(&local_nnz, &nnz_min, 1, MPI_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &nnz_max, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &nnz_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    const long long local_recv =
        x_is_distributed(args.x_mode) ? x_plan.ghost_count : 0;
    const long long local_send =
        x_is_distributed(args.x_mode)
            ? distributed_x_total_send(&x_plan, size)
            : 0;
    const long long local_comm_bytes =
        (local_recv + local_send) * (long long)sizeof(float);
    long long recv_min = 0, recv_max = 0, recv_sum = 0;
    long long send_min = 0, send_max = 0, send_sum = 0;
    long long comm_min = 0, comm_max = 0, comm_sum = 0;
    MPI_Reduce(&local_recv, &recv_min, 1, MPI_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_recv, &recv_max, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_recv, &recv_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_send, &send_min, 1, MPI_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_send, &send_max, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_send, &send_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_bytes, &comm_min, 1, MPI_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_bytes, &comm_max, 1, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_bytes, &comm_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    unsigned long long host_min = 0, host_max = 0, host_sum = 0;
    unsigned long long device_min = 0, device_max = 0, device_sum = 0;
    unsigned long long memory_min = 0, memory_max = 0, memory_sum = 0;
    MPI_Reduce(&host_memory, &host_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&host_memory, &host_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&host_memory, &host_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&device_memory, &device_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&device_memory, &device_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&device_memory, &device_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&total_memory, &memory_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&total_memory, &memory_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&total_memory, &memory_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    double max_generation_seconds = 0.0;
    double max_setup_seconds = 0.0;
    MPI_Reduce(&generation_seconds, &max_generation_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&setup_seconds, &max_setup_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int csv_error = 0;
    if (rank == 0) {
        const double max_rank_time = mean(max_rank_times);
        const double compute_time = mean(max_compute_times);
        const double communication_time = mean(max_communication_times);
        const double time_std =
            standard_deviation(max_rank_times, max_rank_time);
        const double gflops = max_rank_time > 0.0
                                  ? 2.0 * (double)global_nnz /
                                        (max_rank_time * 1.0e9)
                                  : 0.0;
        const double baseline = load_or_store_baseline(
            &args, size, max_rank_time, rank);
        const double efficiency =
            isfinite(baseline) && max_rank_time > 0.0
                ? baseline / max_rank_time
                : std::numeric_limits<double>::quiet_NaN();
        const double density =
            (double)args.nnz_per_row / (double)global_rows;

        FILE *csv = open_weak_csv(args.csv_path);
        if (!csv) {
            fprintf(stderr, "Could not open weak-scaling CSV %s\n",
                    args.csv_path);
            csv_error = 1;
        } else {
            fprintf(
                csv,
                "MPI-CUDA-weak,generated-random,%s,%s,%d,%d,%d,%d,%d,%d,%d,"
                "%lld,%.9e,%llu,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9f,%.9f,"
                "%lld,%.3f,%lld,%lld,%.3f,%lld,%lld,%.3f,%lld,%lld,%.3f,"
                "%lld,%lld,%llu,%.3f,%llu,%llu,%.3f,%llu,%llu,%.3f,%llu,"
                "%d,%d,%.9e,%.9e,%.9e\n",
                args.kernel_name, x_vector_mode_short_name(args.x_mode),
                args.cuda_aware_mpi, args.use_nccl, size, args.rows_per_rank,
                args.nnz_per_row, global_rows, global_rows, global_nnz,
                density, args.seed, args.warmup, args.reps, max_rank_time,
                time_std, compute_time, communication_time, gflops,
                efficiency, nnz_min, (double)nnz_sum / size, nnz_max,
                recv_min, (double)recv_sum / size, recv_max, send_min,
                (double)send_sum / size, send_max, comm_min,
                (double)comm_sum / size, comm_max,
                send_sum * (long long)sizeof(float), host_min,
                (double)host_sum / size, host_max, device_min,
                (double)device_sum / size, device_max, memory_min,
                (double)memory_sum / size, memory_max, validation_performed,
                valid, max_abs_error, max_generation_seconds,
                max_setup_seconds);
            fflush(csv);
            fclose(csv);
        }

        printf("Weak scaling: %d GPU(s), %d rows/GPU, %d NNZ/row, "
               "global rows=%d, global NNZ=%lld, density=%.3e\n",
               size, args.rows_per_rank, args.nnz_per_row, global_rows,
               global_nnz, density);
        printf("Iterative communication backend: %s\n",
               communication_backend_name());
        printf("Timing (max rank): total=%.9f s, compute=%.9f s, "
               "communication=%.9f s, GFLOP/s=%.6f, efficiency=%.4f\n",
               max_rank_time, compute_time, communication_time, gflops,
               efficiency);
        printf("NNZ/rank min/avg/max: %lld / %.1f / %lld\n", nnz_min,
               (double)nnz_sum / size, nnz_max);
        printf("Communication/rank min/avg/max: %lld / %.1f / %lld bytes; "
               "aggregate one-way payload=%lld bytes/iteration\n",
               comm_min, (double)comm_sum / size, comm_max,
               send_sum * (long long)sizeof(float));
        printf("Memory/rank min/avg/max: %.2f / %.2f / %.2f MiB "
               "(host estimate + measured device allocation)\n",
               (double)memory_min / (1024.0 * 1024.0),
               (double)memory_sum / size / (1024.0 * 1024.0),
               (double)memory_max / (1024.0 * 1024.0));
        if (validation_performed) {
            printf("CPU validation: %s (max abs error %.3e)\n",
                   valid ? "PASS" : "FAIL", max_abs_error);
        } else {
            printf("CPU validation: skipped (global rows %d > limit %d)\n",
                   global_rows, args.validation_max_rows);
        }
        printf("CSV: %s\n", args.csv_path);
    }
    MPI_Bcast(&csv_error, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CHECK_CUDA(cudaEventDestroy(compute_start), rank);
    CHECK_CUDA(cudaEventDestroy(compute_stop), rank);
    runner->teardown();
    destroy_spmv_kernel_runner(runner);
    cudaFree(d_x);
    cudaFree(d_y);
    free_device_csr(&d_csr);
    free_distributed_x_plan(&x_plan);
    free_csr(&csr);
    free(x_host);

    finalize_nccl_backend(rank);
    MPI_Finalize();
    return csv_error || (validation_performed && !valid) ? 1 : 0;
}
