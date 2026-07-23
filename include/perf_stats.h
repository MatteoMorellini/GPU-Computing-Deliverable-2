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
    char implementation[96];   // Kernel implementation and communication backend

    // -------------------------------------------------------
    // BASIC MATRIX DIMENSIONS
    // -------------------------------------------------------

    int rows;                  // Number of matrix rows (size of output vector y)
    int cols;                  // Number of matrix columns (size of input vector x)
    int nnz;                   // Total number of nonzero elements (dominant factor in SpMV cost)
    int processes;             // Number of MPI processes used for the distributed run

    // -------------------------------------------------------
    // PERFORMANCE METRICS
    // These quantify runtime stability and computational throughput.
    // -------------------------------------------------------

    double avg_time_s;         // Average execution time of one SpMV iteration (communication + local compute)

    double std_time_s;         // Runtime variability across repetitions; indicates stability of memory behavior
                               // and sensitivity to cache / OS scheduling effects

    double comm_time_s;        // Average per-iteration ghost/fold communication time

    double compute_time_s;     // Average per-iteration local CUDA SpMV kernel time

    // -------------------------------------------------------
    // COMPUTATIONAL THROUGHPUT
    // -------------------------------------------------------

    double gflops;             // Achieved floating-point throughput: 2 * nnz / execution_time
                               // standard performance metric for SpMV benchmarking

    // -------------------------------------------------------
    // ONE-TIME SETUP COSTS (per matrix, not part of kernel time)
    // -------------------------------------------------------

    double file_parse_s;       // Legacy end-to-end input timing

    double input_total_s;      // Complete input path until local COO is ready
    double input_read_parse_s; // Raw reader before validation/redistribution
    double input_file_io_s;    // MPI-IO phase through collective read and close
    double input_parse_s;      // In-memory text buffer to COO conversion
    double matrix_validation_s;// Entry-count collective validation
    double matrix_redistribution_s; // Complete packing + exchange phase
    double matrix_pack_s;      // Redistribution excluding timed MPI collectives
    double matrix_exchange_s;  // Count and COO payload MPI collectives

    long long matrix_source_nnz; // Expanded entries initially read by all ranks
    long long matrix_remote_nnz; // Entries whose row owner differs from reader
    double matrix_remote_fraction;

    double format_conv_s;      // Time for COO -> CSR conversion plus any
                               // algorithm-specific host preprocessing
                               // (row_blocks, batches, cuSPARSE descriptors)

    double h2d_transfer_s;     // Time to allocate + copy CSR arrays and compact owned x
                               // values to the device

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

static inline const char *perf_stats_csv_header(void) {
    return "implementation,format,matrix,rows,cols,nnz,processes,"
           "avg_time_s,std_time_s,gflops,"
           "comm_time_s,compute_time_s,"
           "file_parse_s,input_total_s,input_read_parse_s,"
           "input_file_io_s,input_parse_s,matrix_validation_s,"
           "matrix_redistribution_s,matrix_pack_s,matrix_exchange_s,"
           "matrix_source_nnz,matrix_remote_nnz,matrix_remote_fraction,"
           "format_conv_s,h2d_transfer_s,valid,max_abs_error";
}

// Open the CSV file in append mode and emit the header only for a new file.
// Creates parent directory "results/" if needed.
static inline FILE *perf_stats_open_csv(const char *path) {
    mkdir("results", 0755); // ignore EEXIST

    struct stat st;
    int write_header = stat(path, &st) != 0 || st.st_size == 0;

    if (!write_header) {
        FILE *existing = fopen(path, "r");
        char header[2048];
        if (!existing || !fgets(header, sizeof(header), existing)) {
            if (existing) fclose(existing);
            fprintf(stderr, "Error: could not read CSV header from '%s'\n", path);
            return NULL;
        }
        fclose(existing);
        header[strcspn(header, "\r\n")] = '\0';
        if (strcmp(header, perf_stats_csv_header()) != 0) {
            fprintf(stderr,
                    "Error: CSV schema in '%s' is outdated; use a new --output path or remove the old file\n",
                    path);
            return NULL;
        }
    }

    FILE *f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "Error: could not open results CSV '%s'\n", path);
        return NULL;
    }
    if (write_header) {
        fprintf(f, "%s\n", perf_stats_csv_header());
    }
    return f;
}

// Append one row for a completed measurement and flush so the file is
// safe to read between runs of different executables.
static inline void perf_stats_write_csv_row(FILE *f, const PerfStats *s) {
    if (!f || !s) return;
    fprintf(f, "%s,%s,%s,%d,%d,%d,%d,%.9e,%.9e,%.9f,"
               "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
               "%.9e,%.9e,%.9e,%lld,%lld,%.9e,%.9e,%.9e,%d,%.9e\n",
            s->implementation, s->format, s->name,
            s->rows, s->cols, s->nnz, s->processes,
            s->avg_time_s, s->std_time_s, s->gflops,
            s->comm_time_s, s->compute_time_s,
            s->file_parse_s, s->input_total_s, s->input_read_parse_s,
            s->input_file_io_s, s->input_parse_s,
            s->matrix_validation_s, s->matrix_redistribution_s,
            s->matrix_pack_s, s->matrix_exchange_s,
            s->matrix_source_nnz, s->matrix_remote_nnz,
            s->matrix_remote_fraction,
            s->format_conv_s, s->h2d_transfer_s,
            s->valid, s->max_abs_error);
    fflush(f);
}

#endif // PERF_STATS_H
