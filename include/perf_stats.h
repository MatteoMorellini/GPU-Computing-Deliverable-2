#ifndef PERF_STATS_H
#define PERF_STATS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {

    // -------------------------------------------------------
    // IDENTIFICATION METADATA
    // -------------------------------------------------------

    char name[256];            // Matrix filename (identifier of dataset)
    char format[32];           // Sparse storage format used (e.g., CSR, COO, ELL)
    char implementation[32];   // Kernel implementation (e.g., CPU single-core, OpenMP, CUDA)

    // -------------------------------------------------------
    // BASIC MATRIX DIMENSIONS
    // -------------------------------------------------------

    int rows;                  // Number of matrix rows (size of output vector y)
    int cols;                  // Number of matrix columns (size of input vector x)
    int nnz;                   // Total number of nonzero elements (dominant factor in SpMV cost)

    // -------------------------------------------------------
    // PERFORMANCE METRICS
    // These quantify runtime stability and computational throughput.
    // -------------------------------------------------------

    double avg_time_s;         // Average execution time of SpMV kernel (excluding preprocessing)

    double std_time_s;         // Runtime variability across repetitions; indicates stability of memory behavior
                               // and sensitivity to cache / OS scheduling effects

    // -------------------------------------------------------
    // COMPUTATIONAL THROUGHPUT
    // -------------------------------------------------------

    double gflops;             // Achieved floating-point throughput: 2 * nnz / execution_time
                               // standard performance metric for SpMV benchmarking

    // -------------------------------------------------------
    // ONE-TIME SETUP COSTS (per matrix, not part of kernel time)
    // -------------------------------------------------------

    double file_parse_s;       // Time to read the .mtx file into COO

    double format_conv_s;      // Time for COO -> CSR conversion plus any
                               // algorithm-specific host preprocessing
                               // (row_blocks, batches, cuSPARSE descriptors)

    double h2d_transfer_s;     // Time to allocate + copy CSR arrays
                               // (row_ptr, col_idx, values) to the device

    // -------------------------------------------------------
    // VALIDATION METRICS
    // -------------------------------------------------------

    int valid;                 // Indicates whether kernel output matches reference implementation

    double max_abs_error;      // Maximum absolute difference vs reference result; detects correctness issues
                               // caused by numerical errors or implementation bugs

} PerfStats;

// Resolve the output CSV path: RESULTS_CSV env var overrides the per-program default.
static inline const char *perf_stats_resolve_path(const char *default_path) {
    const char *env = getenv("RESULTS_CSV");
    return (env && env[0]) ? env : default_path;
}

// Open the CSV file in write mode and emit the header. Returns NULL on failure.
// Creates parent directory "results/" if needed.
static inline FILE *perf_stats_open_csv(const char *path) {
    mkdir("results", 0755); // ignore EEXIST
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Error: could not open results CSV '%s'\n", path);
        return NULL;
    }
    fprintf(f, "implementation,format,matrix,rows,cols,nnz,"
               "avg_time_s,std_time_s,gflops,"
               "file_parse_s,format_conv_s,h2d_transfer_s,"
               "valid,max_abs_error\n");
    return f;
}

// Append one row for a completed measurement and flush so the file is
// safe to read between runs of different executables.
static inline void perf_stats_write_csv_row(FILE *f, const PerfStats *s) {
    if (!f || !s) return;
    fprintf(f, "%s,%s,%s,%d,%d,%d,%.9e,%.9e,%.9f,%.9e,%.9e,%.9e,%d,%.9e\n",
            s->implementation, s->format, s->name,
            s->rows, s->cols, s->nnz,
            s->avg_time_s, s->std_time_s, s->gflops,
            s->file_parse_s, s->format_conv_s, s->h2d_transfer_s,
            s->valid, s->max_abs_error);
    fflush(f);
}

#endif // PERF_STATS_H
