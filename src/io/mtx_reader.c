#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mtx_reader.h"

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
