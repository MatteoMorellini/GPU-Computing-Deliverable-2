#include <mpi.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi_mtx_reader.h"

#define HEADER_READ_BLOCK 4096
#define BOUNDARY_READ_BLOCK 4096

typedef struct {
    int rows;
    int cols;
    int stored_nnz;
    int is_pattern;
    int is_symmetric;
    MPI_Offset data_offset;
} MpiMtxHeader;

static void report_mpi_error(int rank, const char *operation, int error_code) {
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error_code, message, &length);
    fprintf(stderr, "Rank %d: %s failed: %.*s\n",
            rank, operation, length, message);
}

static int parse_banner(char *line,
                        const char *filename,
                        MpiMtxHeader *header) {
    char object[64];
    char format[64];
    char field[64];
    char symmetry[64];

    if (sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s",
               object, format, field, symmetry) != 4) {
        fprintf(stderr, "Invalid Matrix Market header in %s\n", filename);
        return 1;
    }
    if (strcmp(object, "matrix") != 0 || strcmp(format, "coordinate") != 0) {
        fprintf(stderr, "Only coordinate matrix format is supported\n");
        return 1;
    }
    if (strcmp(field, "real") != 0 && strcmp(field, "pattern") != 0) {
        fprintf(stderr, "Unsupported Matrix Market field type: %s\n", field);
        return 1;
    }
    if (strcmp(symmetry, "general") != 0 &&
        strcmp(symmetry, "symmetric") != 0) {
        fprintf(stderr, "Unsupported Matrix Market symmetry type: %s\n",
                symmetry);
        return 1;
    }

    header->is_pattern = strcmp(field, "pattern") == 0;
    header->is_symmetric = strcmp(symmetry, "symmetric") == 0;
    return 0;
}

/*
 * Header discovery is intentionally performed only by rank 0.  MPI_File_read_at
 * is independent, so other ranks do not have to participate in these small
 * reads.  The matrix data itself is read collectively below.
 */
static int read_header_on_root(MPI_File file,
                               MPI_Offset file_size,
                               const char *filename,
                               MpiMtxHeader *header) {
    char *buffer = NULL;
    size_t capacity = 0;
    size_t used = 0;
    size_t scan = 0;
    int saw_banner = 0;
    int status = 1;

    while ((MPI_Offset)used < file_size) {
        MPI_Offset remaining = file_size - (MPI_Offset)used;
        int requested =
            remaining < HEADER_READ_BLOCK ? (int)remaining : HEADER_READ_BLOCK;
        size_t required = used + (size_t)requested + 1;
        if (required > capacity) {
            size_t next_capacity = capacity == 0 ? HEADER_READ_BLOCK + 1
                                                 : capacity * 2;
            while (next_capacity < required) {
                if (next_capacity > SIZE_MAX / 2) {
                    fprintf(stderr, "Matrix Market header is too large in %s\n",
                            filename);
                    goto cleanup;
                }
                next_capacity *= 2;
            }
            char *next = (char *)realloc(buffer, next_capacity);
            if (!next) {
                fprintf(stderr,
                        "Could not allocate Matrix Market header buffer for %s\n",
                        filename);
                goto cleanup;
            }
            buffer = next;
            capacity = next_capacity;
        }

        MPI_Status io_status;
        int mpi_status =
            MPI_File_read_at(file, (MPI_Offset)used, buffer + used, requested,
                             MPI_BYTE, &io_status);
        if (mpi_status != MPI_SUCCESS) {
            report_mpi_error(0, "MPI_File_read_at(header)", mpi_status);
            goto cleanup;
        }

        int received = 0;
        mpi_status = MPI_Get_count(&io_status, MPI_BYTE, &received);
        if (mpi_status != MPI_SUCCESS || received < 0) {
            if (mpi_status != MPI_SUCCESS) {
                report_mpi_error(0, "MPI_Get_count(header)", mpi_status);
            }
            goto cleanup;
        }
        if (received == 0) {
            break;
        }

        used += (size_t)received;
        buffer[used] = '\0';

        while (scan < used) {
            char *newline = (char *)memchr(buffer + scan, '\n', used - scan);
            if (!newline) {
                break;
            }

            size_t line_end = (size_t)(newline - buffer);
            buffer[line_end] = '\0';
            char *line = buffer + scan;
            size_t line_length = line_end - scan;
            if (line_length > 0 && line[line_length - 1] == '\r') {
                line[line_length - 1] = '\0';
            }

            if (!saw_banner) {
                if (parse_banner(line, filename, header) != 0) {
                    goto cleanup;
                }
                saw_banner = 1;
            } else if (line[0] != '%') {
                if (sscanf(line, "%d %d %d", &header->rows, &header->cols,
                           &header->stored_nnz) != 3) {
                    fprintf(stderr, "Invalid size line in %s\n", filename);
                    goto cleanup;
                }
                if (header->rows < 0 || header->cols < 0 ||
                    header->stored_nnz < 0) {
                    fprintf(stderr,
                            "Negative Matrix Market dimensions in %s\n",
                            filename);
                    goto cleanup;
                }
                header->data_offset = (MPI_Offset)line_end + 1;
                status = 0;
                goto cleanup;
            }
            scan = line_end + 1;
        }

        if (received < requested) {
            break;
        }
    }

    /*
     * Accept a dimensions line ending exactly at EOF without a final newline.
     * A banner without a dimensions line remains invalid.
     */
    if (saw_banner && scan < used) {
        char *line = buffer + scan;
        size_t line_length = used - scan;
        if (line_length > 0 && line[line_length - 1] == '\r') {
            line[line_length - 1] = '\0';
        }
        if (line[0] != '%' &&
            sscanf(line, "%d %d %d", &header->rows, &header->cols,
                   &header->stored_nnz) == 3 &&
            header->rows >= 0 && header->cols >= 0 &&
            header->stored_nnz >= 0) {
            header->data_offset = (MPI_Offset)used;
            status = 0;
        }
    }

    if (status != 0) {
        fprintf(stderr, "Error reading size line from %s\n", filename);
    }

cleanup:
    free(buffer);
    return status;
}

static int read_one_byte(MPI_File file,
                         MPI_Offset offset,
                         char *value,
                         int rank,
                         const char *context) {
    MPI_Status io_status;
    int mpi_status =
        MPI_File_read_at(file, offset, value, 1, MPI_BYTE, &io_status);
    if (mpi_status != MPI_SUCCESS) {
        report_mpi_error(rank, context, mpi_status);
        return 1;
    }

    int received = 0;
    mpi_status = MPI_Get_count(&io_status, MPI_BYTE, &received);
    if (mpi_status != MPI_SUCCESS) {
        report_mpi_error(rank, "MPI_Get_count(boundary)", mpi_status);
        return 1;
    }
    return received == 1 ? 0 : 1;
}

/*
 * Return the first line start at or after nominal_start.  Advancing each
 * nominal boundary this way assigns a line to the rank whose nominal range
 * contains that line's first byte.
 */
static int find_aligned_start(MPI_File file,
                              MPI_Offset data_offset,
                              MPI_Offset file_size,
                              MPI_Offset nominal_start,
                              MPI_Offset *aligned_start,
                              int rank) {
    if (nominal_start <= data_offset) {
        *aligned_start = data_offset;
        return 0;
    }
    if (nominal_start >= file_size) {
        *aligned_start = file_size;
        return 0;
    }

    char preceding = '\0';
    if (read_one_byte(file, nominal_start - 1, &preceding, rank,
                      "MPI_File_read_at(boundary prefix)") != 0) {
        return 1;
    }
    if (preceding == '\n') {
        *aligned_start = nominal_start;
        return 0;
    }

    char block[BOUNDARY_READ_BLOCK];
    MPI_Offset position = nominal_start;
    while (position < file_size) {
        MPI_Offset remaining = file_size - position;
        int requested = remaining < BOUNDARY_READ_BLOCK
                            ? (int)remaining
                            : BOUNDARY_READ_BLOCK;
        MPI_Status io_status;
        int mpi_status = MPI_File_read_at(file, position, block, requested,
                                          MPI_BYTE, &io_status);
        if (mpi_status != MPI_SUCCESS) {
            report_mpi_error(rank, "MPI_File_read_at(boundary scan)",
                             mpi_status);
            return 1;
        }

        int received = 0;
        mpi_status = MPI_Get_count(&io_status, MPI_BYTE, &received);
        if (mpi_status != MPI_SUCCESS || received < 0) {
            if (mpi_status != MPI_SUCCESS) {
                report_mpi_error(rank, "MPI_Get_count(boundary scan)",
                                 mpi_status);
            }
            return 1;
        }
        if (received == 0) {
            break;
        }

        char *newline = (char *)memchr(block, '\n', (size_t)received);
        if (newline) {
            *aligned_start =
                position + (MPI_Offset)(newline - block) + 1;
            return 0;
        }
        position += received;
    }

    *aligned_start = file_size;
    return 0;
}

static MPI_Offset balanced_partition_offset(MPI_Offset bytes,
                                            int part,
                                            int parts) {
    const MPI_Offset base = bytes / parts;
    const MPI_Offset remainder = bytes % parts;
    const MPI_Offset preceding_extra =
        (MPI_Offset)part < remainder ? (MPI_Offset)part : remainder;
    return base * part + preceding_extra;
}

static int append_entry(COO_Matrix *mat,
                        int *capacity,
                        int row,
                        int col,
                        double value) {
    if (mat->nnz == *capacity) {
        if (*capacity > INT_MAX / 2) {
            return 1;
        }
        int next_capacity = *capacity == 0 ? 1024 : *capacity * 2;

        int *next_rows =
            (int *)realloc(mat->row, (size_t)next_capacity * sizeof(int));
        if (!next_rows) {
            return 1;
        }
        mat->row = next_rows;

        int *next_cols =
            (int *)realloc(mat->col, (size_t)next_capacity * sizeof(int));
        if (!next_cols) {
            return 1;
        }
        mat->col = next_cols;

        double *next_data =
            (double *)realloc(mat->data,
                              (size_t)next_capacity * sizeof(double));
        if (!next_data) {
            return 1;
        }
        mat->data = next_data;
        *capacity = next_capacity;
    }

    mat->row[mat->nnz] = row;
    mat->col[mat->nnz] = col;
    mat->data[mat->nnz] = value;
    mat->nnz++;
    return 0;
}

static int parse_data_buffer(char *buffer,
                             size_t bytes,
                             MPI_Offset file_offset,
                             const char *filename,
                             const MpiMtxHeader *header,
                             COO_Matrix *mat,
                             int *local_stored_nnz,
                             int rank) {
    char *cursor = buffer;
    char *end = buffer + bytes;
    int capacity = 0;

    while (cursor < end) {
        char *newline = (char *)memchr(cursor, '\n', (size_t)(end - cursor));
        char *line_end = newline ? newline : end;
        if (newline) {
            *newline = '\0';
        }

        char *text = cursor;
        while (text < line_end && (*text == ' ' || *text == '\t')) {
            text++;
        }
        if (text < line_end && *text != '\0' && *text != '\r' &&
            *text != '%') {
            int row = 0;
            int col = 0;
            double value = 1.0;
            int parsed =
                header->is_pattern
                    ? sscanf(text, "%d %d", &row, &col)
                    : sscanf(text, "%d %d %lf", &row, &col, &value);
            if (parsed != (header->is_pattern ? 2 : 3)) {
                MPI_Offset line_offset =
                    file_offset + (MPI_Offset)(cursor - buffer);
                fprintf(stderr,
                        "Rank %d could not parse Matrix Market data at byte %lld in %s\n",
                        rank, (long long)line_offset, filename);
                return 1;
            }

            row--;
            col--;
            if (row < 0 || row >= mat->rows ||
                col < 0 || col >= mat->cols) {
                fprintf(stderr,
                        "Rank %d found invalid index row=%d col=%d in %s\n",
                        rank, row, col, filename);
                return 1;
            }

            if (append_entry(mat, &capacity, row, col, value) != 0 ||
                (header->is_symmetric && row != col &&
                 append_entry(mat, &capacity, col, row, value) != 0)) {
                fprintf(stderr,
                        "Rank %d could not allocate its MPI-IO COO chunk\n",
                        rank);
                return 1;
            }
            (*local_stored_nnz)++;
        }

        if (!newline) {
            break;
        }
        cursor = newline + 1;
    }
    return 0;
}

int read_mtx_chunk_mpi_io_timed(const char *filename,
                                COO_Matrix *mat,
                                int *declared_stored_nnz,
                                int *local_stored_nnz,
                                MpiMtxReadMetrics *metrics,
                                MPI_Comm comm) {
    const double io_start = MPI_Wtime();
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    memset(mat, 0, sizeof(*mat));
    if (declared_stored_nnz) {
        *declared_stored_nnz = 0;
    }
    if (local_stored_nnz) {
        *local_stored_nnz = 0;
    }
    if (metrics) {
        memset(metrics, 0, sizeof(*metrics));
    }

    MPI_File file = MPI_FILE_NULL;
    int mpi_status =
        MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    int local_error = mpi_status != MPI_SUCCESS;
    if (local_error) {
        report_mpi_error(rank, "MPI_File_open", mpi_status);
    }
    int any_error = 0;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        return 1;
    }

    mpi_status = MPI_File_set_errhandler(file, MPI_ERRORS_RETURN);
    local_error = mpi_status != MPI_SUCCESS;
    if (local_error) {
        report_mpi_error(rank, "MPI_File_set_errhandler", mpi_status);
    }
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        MPI_File_close(&file);
        return 1;
    }

    MPI_Offset file_size = 0;
    mpi_status = MPI_File_get_size(file, &file_size);
    if (mpi_status != MPI_SUCCESS) {
        report_mpi_error(rank, "MPI_File_get_size", mpi_status);
        local_error = 1;
    }

    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        MPI_File_close(&file);
        return 1;
    }

    MpiMtxHeader header = {0};
    if (rank == 0) {
        local_error =
            read_header_on_root(file, file_size, filename, &header);
    }
    MPI_Bcast(&local_error, 1, MPI_INT, 0, comm);
    if (local_error) {
        MPI_File_close(&file);
        return 1;
    }

    int metadata[5] = {0};
    if (rank == 0) {
        metadata[0] = header.rows;
        metadata[1] = header.cols;
        metadata[2] = header.stored_nnz;
        metadata[3] = header.is_pattern;
        metadata[4] = header.is_symmetric;
    }
    MPI_Bcast(metadata, 5, MPI_INT, 0, comm);
    MPI_Bcast(&header.data_offset, 1, MPI_OFFSET, 0, comm);
    if (rank != 0) {
        header.rows = metadata[0];
        header.cols = metadata[1];
        header.stored_nnz = metadata[2];
        header.is_pattern = metadata[3];
        header.is_symmetric = metadata[4];
    }

    mat->rows = header.rows;
    mat->cols = header.cols;
    if (declared_stored_nnz) {
        *declared_stored_nnz = header.stored_nnz;
    }
    if (header.data_offset > file_size) {
        if (rank == 0) {
            fprintf(stderr, "Invalid data section in %s\n", filename);
        }
        MPI_File_close(&file);
        free_coo(mat);
        return 1;
    }

    const MPI_Offset data_bytes = file_size - header.data_offset;
    const MPI_Offset nominal_start =
        header.data_offset +
        balanced_partition_offset(data_bytes, rank, size);

    MPI_Offset aligned_start = header.data_offset;
    local_error =
        find_aligned_start(file, header.data_offset, file_size,
                           nominal_start, &aligned_start, rank);
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        MPI_File_close(&file);
        free_coo(mat);
        return 1;
    }

    MPI_Offset aligned_end = file_size;
    mpi_status = MPI_Sendrecv(
        &aligned_start, 1, MPI_OFFSET,
        rank == 0 ? MPI_PROC_NULL : rank - 1, 0,
        &aligned_end, 1, MPI_OFFSET,
        rank == size - 1 ? MPI_PROC_NULL : rank + 1, 0,
        comm, MPI_STATUS_IGNORE);
    if (mpi_status != MPI_SUCCESS) {
        report_mpi_error(rank, "MPI_Sendrecv(aligned boundaries)", mpi_status);
        local_error = 1;
    }
    if (aligned_end < aligned_start || aligned_end > file_size) {
        fprintf(stderr, "Rank %d received invalid MPI-IO byte boundaries\n",
                rank);
        local_error = 1;
    }
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        MPI_File_close(&file);
        free_coo(mat);
        return 1;
    }

    const MPI_Offset local_bytes = aligned_end - aligned_start;
    if ((uintmax_t)local_bytes > (uintmax_t)(SIZE_MAX - 1)) {
        fprintf(stderr, "Rank %d MPI-IO chunk is too large for this process\n",
                rank);
        local_error = 1;
    }
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        MPI_File_close(&file);
        free_coo(mat);
        return 1;
    }

    char *buffer = (char *)malloc((size_t)local_bytes + 1);
    local_error = buffer == NULL;
    MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
    if (any_error) {
        if (local_error) {
            fprintf(stderr, "Rank %d could not allocate its MPI-IO buffer\n",
                    rank);
        }
        free(buffer);
        MPI_File_close(&file);
        free_coo(mat);
        return 1;
    }

    MPI_Offset max_bytes = 0;
    MPI_Allreduce(&local_bytes, &max_bytes, 1, MPI_OFFSET, MPI_MAX, comm);
    const MPI_Offset max_read_count = INT_MAX;
    for (MPI_Offset completed = 0; completed < max_bytes;
         completed += max_read_count) {
        MPI_Offset local_remaining =
            completed < local_bytes ? local_bytes - completed : 0;
        int requested = local_remaining > max_read_count
                            ? INT_MAX
                            : (int)local_remaining;
        size_t buffer_position =
            completed < local_bytes ? (size_t)completed : (size_t)local_bytes;
        MPI_Offset file_position =
            completed < local_bytes ? completed : local_bytes;
        MPI_Status io_status;
        mpi_status = MPI_File_read_at_all(
            file, aligned_start + file_position, buffer + buffer_position,
            requested, MPI_BYTE, &io_status);

        local_error = mpi_status != MPI_SUCCESS;
        if (local_error) {
            report_mpi_error(rank, "MPI_File_read_at_all", mpi_status);
        } else {
            int received = 0;
            mpi_status = MPI_Get_count(&io_status, MPI_BYTE, &received);
            if (mpi_status != MPI_SUCCESS) {
                report_mpi_error(rank, "MPI_Get_count(data)", mpi_status);
                local_error = 1;
            } else if (received != requested) {
                fprintf(stderr,
                        "Rank %d MPI-IO short read: requested %d, received %d\n",
                        rank, requested, received);
                local_error = 1;
            }
        }

        MPI_Allreduce(&local_error, &any_error, 1, MPI_INT, MPI_MAX, comm);
        if (any_error) {
            break;
        }
    }
    buffer[(size_t)local_bytes] = '\0';

    mpi_status = MPI_File_close(&file);
    if (mpi_status != MPI_SUCCESS) {
        report_mpi_error(rank, "MPI_File_close", mpi_status);
        any_error = 1;
    }
    if (any_error) {
        free(buffer);
        free_coo(mat);
        return 1;
    }
    if (metrics) {
        metrics->file_io_s = MPI_Wtime() - io_start;
    }

    int parsed_stored_nnz = 0;
    const double parse_start = MPI_Wtime();
    local_error =
        parse_data_buffer(buffer, (size_t)local_bytes, aligned_start,
                          filename, &header, mat, &parsed_stored_nnz, rank);
    free(buffer);
    if (metrics) {
        metrics->parse_s = MPI_Wtime() - parse_start;
    }
    if (local_error) {
        free_coo(mat);
        return 1;
    }
    if (local_stored_nnz) {
        *local_stored_nnz = parsed_stored_nnz;
    }
    return 0;
}

int read_mtx_chunk_mpi_io(const char *filename,
                          COO_Matrix *mat,
                          int *declared_stored_nnz,
                          int *local_stored_nnz,
                          MPI_Comm comm) {
    return read_mtx_chunk_mpi_io_timed(
        filename, mat, declared_stored_nnz, local_stored_nnz, NULL, comm);
}
