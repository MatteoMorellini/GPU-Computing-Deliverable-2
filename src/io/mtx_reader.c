#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mtx_reader.h"

typedef struct {
    int rows;
    int cols;
    int stored_nnz;
    int is_pattern;
    int is_symmetric;
    long data_offset;
} MtxHeader;

static int read_header(FILE *f, const char *filename, MtxHeader *header) {
    char line[1024];
    char object[64], format[64], field[64], symmetry[64];

    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Error reading header from %s\n", filename);
        return 1;
    }

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

    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Error reading size line from %s\n", filename);
            return 1;
        }
    } while (line[0] == '%');

    if (sscanf(line, "%d %d %d", &header->rows, &header->cols,
               &header->stored_nnz) != 3) {
        fprintf(stderr, "Invalid size line in %s\n", filename);
        return 1;
    }
    if (header->rows < 0 || header->cols < 0 || header->stored_nnz < 0) {
        fprintf(stderr, "Negative Matrix Market dimensions in %s\n", filename);
        return 1;
    }

    header->is_pattern = strcmp(field, "pattern") == 0;
    header->is_symmetric = strcmp(symmetry, "symmetric") == 0;
    header->data_offset = ftell(f);
    if (header->data_offset < 0) {
        fprintf(stderr, "Could not determine data offset in %s\n", filename);
        return 1;
    }
    return 0;
}

static int append_entry(COO_Matrix *mat,
                        int *capacity,
                        int row,
                        int col,
                        double value) {
    if (mat->nnz == *capacity) {
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

void read_mtx(const char *filename, COO_Matrix *mat) {
    FILE *f = fopen(filename, "r");
    char line[1024];
    char object[64], format[64], field[64], symmetry[64];

    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Error reading header from %s\n", filename);
        exit(1);
    }

    if (sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s",
               object, format, field, symmetry) != 4) {
        fprintf(stderr, "Invalid Matrix Market header in %s\n", filename);
        exit(1);
    }

    if (strcmp(object, "matrix") != 0 || strcmp(format, "coordinate") != 0) {
        fprintf(stderr, "Only coordinate matrix format is supported\n");
        exit(1);
    }

    if (strcmp(field, "real") != 0 && strcmp(field, "pattern") != 0) {
        fprintf(stderr, "Unsupported Matrix Market field type: %s\n", field);
        fclose(f);
        exit(1);
    }

    if (strcmp(symmetry, "general") != 0 && strcmp(symmetry, "symmetric") != 0) {
        fprintf(stderr, "Unsupported Matrix Market symmetry type: %s\n", symmetry);
        fclose(f);
        exit(1);
    }

    int is_symmetric = (strcmp(symmetry, "symmetric") == 0);

    do {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Error reading size line from %s\n", filename);
            fclose(f);
            exit(1);
        }
    } while (line[0] == '%');

    int stored_nnz;
    if (sscanf(line, "%d %d %d", &mat->rows, &mat->cols, &stored_nnz) != 3) {
        fprintf(stderr, "Invalid size line in %s\n", filename);
        fclose(f);
        exit(1);
    }

    // worst case for symmetric: every entry is off-diagonal so nnz doubles
    int max_nnz = is_symmetric ? 2 * stored_nnz : stored_nnz;

    mat->row = malloc((size_t)max_nnz * sizeof(int));
    mat->col = malloc((size_t)max_nnz * sizeof(int));
    mat->data = malloc((size_t)max_nnz * sizeof(double));

    if (!mat->row || !mat->col || !mat->data) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    int k = 0; // actual number of entries stored in memory
    for (int i = 0; i < stored_nnz; i++) {
        int r, c;
        double v;
        int ret;

        if (strcmp(field, "pattern") == 0) {
            ret = fscanf(f, "%d %d", &r, &c);
            if (ret != 2) {
                fprintf(stderr, "Parse error at entry %d in %s\n", i, filename);
                goto cleanup;
            }
            // for pattern matrices, we assume a value of 1.0 for all non-zero entries
            v = 1.0;
        } else if (strcmp(field, "real") == 0) {
            ret = fscanf(f, "%d %d %lf", &r, &c, &v);
            if (ret != 3) {
                fprintf(stderr, "Parse error at entry %d in %s\n", i, filename);
                goto cleanup;
            }
        } else {
            fprintf(stderr, "Unsupported Matrix Market field type: %s\n", field);
            goto cleanup;
        }

        // Convert from 1-based to 0-based indexing
        r--;
        c--;

        if (r < 0 || r >= mat->rows || c < 0 || c >= mat->cols) {
            fprintf(stderr, "Invalid index at entry %d: row=%d col=%d\n", i, r, c);
            goto cleanup;
        }

        /* Store the entry as given */
        mat->row[k] = r;
        mat->col[k] = c;
        mat->data[k] = v;
        k++;

        /* If symmetric and off-diagonal, store mirrored entry too */
        if (is_symmetric && r != c) {
            mat->row[k] = c;
            mat->col[k] = r;
            mat->data[k] = v;
            k++;
        }
    }
    mat->nnz = k; // actual number of entries stored in memory

    fclose(f);
    return;

cleanup:
    if (f) fclose(f);
    free_coo(mat);
    exit(1);
}

int read_mtx_chunk(const char *filename,
                   COO_Matrix *mat,
                   int *declared_stored_nnz,
                   int *local_stored_nnz,
                   int rank,
                   int size) {
    FILE *f = NULL;
    MtxHeader header;
    char *line = NULL;
    size_t line_capacity = 0;
    int capacity = 0;
    int status = 1;

    memset(mat, 0, sizeof(*mat));
    if (declared_stored_nnz) {
        *declared_stored_nnz = 0;
    }
    if (local_stored_nnz) {
        *local_stored_nnz = 0;
    }
    if (rank < 0 || size <= 0 || rank >= size) {
        fprintf(stderr, "Invalid file-chunk rank %d/%d\n", rank, size);
        return 1;
    }

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return 1;
    }
    if (read_header(f, filename, &header) != 0) {
        goto cleanup;
    }

    mat->rows = header.rows;
    mat->cols = header.cols;
    if (declared_stored_nnz) {
        *declared_stored_nnz = header.stored_nnz;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "Could not seek to the end of %s\n", filename);
        goto cleanup;
    }
    long file_end = ftell(f);
    if (file_end < header.data_offset) {
        fprintf(stderr, "Invalid data section in %s\n", filename);
        goto cleanup;
    }

    const long data_bytes = file_end - header.data_offset;
    const long chunk_start =
        header.data_offset + (long)(((long long)data_bytes * rank) / size);
    const long chunk_end =
        header.data_offset + (long)(((long long)data_bytes * (rank + 1)) / size);

    /*
     * If a nominal chunk starts in the middle of a line, discard that partial
     * line.  The preceding rank owns it because its line start is before that
     * rank's nominal end.  A start immediately after '\n' is already aligned.
     */
    long first_line = chunk_start;
    if (rank > 0 && chunk_start > header.data_offset) {
        if (fseek(f, chunk_start - 1, SEEK_SET) != 0) {
            goto cleanup;
        }
        int preceding = fgetc(f);
        if (preceding != '\n') {
            if (getline(&line, &line_capacity, f) < 0 && ferror(f)) {
                goto cleanup;
            }
            first_line = ftell(f);
        }
    }

    if (fseek(f, first_line, SEEK_SET) != 0) {
        goto cleanup;
    }

    while (rank == size - 1 || ftell(f) < chunk_end) {
        long line_start = ftell(f);
        if (line_start < 0 || line_start >= file_end) {
            break;
        }
        if (getline(&line, &line_capacity, f) < 0) {
            if (ferror(f)) {
                fprintf(stderr, "Error reading data chunk from %s\n", filename);
                goto cleanup;
            }
            break;
        }

        char *cursor = line;
        while (*cursor == ' ' || *cursor == '\t') {
            cursor++;
        }
        if (*cursor == '\0' || *cursor == '\n' || *cursor == '\r' ||
            *cursor == '%') {
            continue;
        }

        int row = 0;
        int col = 0;
        double value = 1.0;
        int parsed = header.is_pattern
                         ? sscanf(cursor, "%d %d", &row, &col)
                         : sscanf(cursor, "%d %d %lf", &row, &col, &value);
        if (parsed != (header.is_pattern ? 2 : 3)) {
            fprintf(stderr,
                    "Rank %d could not parse Matrix Market data at byte %ld in %s\n",
                    rank, line_start, filename);
            goto cleanup;
        }

        row--;
        col--;
        if (row < 0 || row >= mat->rows || col < 0 || col >= mat->cols) {
            fprintf(stderr,
                    "Rank %d found invalid index row=%d col=%d in %s\n",
                    rank, row, col, filename);
            goto cleanup;
        }

        if (append_entry(mat, &capacity, row, col, value) != 0 ||
            (header.is_symmetric && row != col &&
             append_entry(mat, &capacity, col, row, value) != 0)) {
            fprintf(stderr, "Rank %d could not allocate its COO chunk\n", rank);
            goto cleanup;
        }
        if (local_stored_nnz) {
            (*local_stored_nnz)++;
        }
    }

    status = 0;

cleanup:
    if (f) {
        fclose(f);
    }
    free(line);
    if (status != 0) {
        free_coo(mat);
    }
    return status;
}

void free_coo(COO_Matrix *mat) {
    free(mat->row);
    free(mat->col);
    free(mat->data);
    mat->row = NULL;
    mat->col = NULL;
    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
    mat->nnz = 0;
}
