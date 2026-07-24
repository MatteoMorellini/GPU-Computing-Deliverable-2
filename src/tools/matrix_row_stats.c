#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t rows;
    uint64_t cols;
    uint64_t stored_nnz;
    int symmetric;
} MatrixHeader;

static int read_header(FILE *stream,
                       const char *path,
                       MatrixHeader *header,
                       char **line,
                       size_t *line_capacity) {
    char object[32];
    char format[32];
    char field[32];
    char symmetry[32];

    if (getline(line, line_capacity, stream) < 0) {
        fprintf(stderr, "%s: cannot read the Matrix Market banner\n", path);
        return 1;
    }
    if (sscanf(*line,
               "%%%%MatrixMarket %31s %31s %31s %31s",
               object,
               format,
               field,
               symmetry) != 4) {
        fprintf(stderr, "%s: invalid Matrix Market banner\n", path);
        return 1;
    }
    if (strcmp(object, "matrix") != 0 || strcmp(format, "coordinate") != 0) {
        fprintf(stderr, "%s: only coordinate matrices are supported\n", path);
        return 1;
    }
    if (strcmp(symmetry, "general") != 0 &&
        strcmp(symmetry, "symmetric") != 0) {
        fprintf(stderr,
                "%s: unsupported Matrix Market symmetry '%s'\n",
                path,
                symmetry);
        return 1;
    }

    do {
        if (getline(line, line_capacity, stream) < 0) {
            fprintf(stderr, "%s: missing matrix dimensions\n", path);
            return 1;
        }
    } while ((*line)[0] == '%');

    if (sscanf(*line,
               "%" SCNu64 " %" SCNu64 " %" SCNu64,
               &header->rows,
               &header->cols,
               &header->stored_nnz) != 3) {
        fprintf(stderr, "%s: invalid matrix dimensions\n", path);
        return 1;
    }
    header->symmetric = strcmp(symmetry, "symmetric") == 0;
    if (header->symmetric && header->rows != header->cols) {
        fprintf(stderr, "%s: a symmetric matrix must be square\n", path);
        return 1;
    }
    return 0;
}

static int parse_index(const char **cursor, uint64_t *value) {
    const unsigned char *position = (const unsigned char *)*cursor;
    uint64_t parsed = 0;

    while (isspace(*position)) {
        position++;
    }
    if (!isdigit(*position)) {
        return 1;
    }
    do {
        unsigned int digit = (unsigned int)(*position - '0');
        if (parsed > (UINT64_MAX - digit) / 10) {
            return 1;
        }
        parsed = parsed * 10 + digit;
        position++;
    } while (isdigit(*position));

    *cursor = (const char *)position;
    *value = parsed;
    return 0;
}

static int parse_coordinate(const char *line, uint64_t *row, uint64_t *col) {
    return parse_index(&line, row) || parse_index(&line, col);
}

static uint32_t select_degree(const uint32_t *degrees,
                              uint64_t rows,
                              uint64_t rank,
                              uint32_t maximum) {
    uint32_t low = 0;
    uint32_t high = maximum;

    /*
     * Value-domain binary search avoids sorting or allocating a histogram
     * whose size is controlled by a potentially pathological maximum degree.
     */
    while (low < high) {
        uint32_t middle = low + (high - low) / 2;
        uint64_t at_or_below = 0;

        for (uint64_t row = 0; row < rows; row++) {
            at_or_below += degrees[row] <= middle;
        }
        if (at_or_below > rank) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}

static const char *matrix_name(const char *path, char *buffer, size_t size) {
    const char *name = strrchr(path, '/');
    size_t length;

    name = name ? name + 1 : path;
    length = strlen(name);
    if (length > 4 && strcmp(name + length - 4, ".mtx") == 0) {
        length -= 4;
    }
    if (length >= size) {
        length = size - 1;
    }
    memcpy(buffer, name, length);
    buffer[length] = '\0';
    return buffer;
}

static int analyze_matrix(const char *path) {
    const size_t input_buffer_size = 8U * 1024U * 1024U;
    FILE *stream = NULL;
    char *input_buffer = NULL;
    char *line = NULL;
    size_t line_capacity = 0;
    MatrixHeader header = {0};
    uint32_t *degrees = NULL;
    uint64_t entries_read = 0;
    uint64_t expanded_nnz = 0;
    uint32_t maximum = 0;
    int status = 1;

    fprintf(stderr, "Analyzing %s\n", path);
    stream = fopen(path, "rb");
    if (!stream) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        goto cleanup;
    }

    input_buffer = malloc(input_buffer_size);
    if (input_buffer) {
        (void)setvbuf(stream, input_buffer, _IOFBF, input_buffer_size);
    }
    if (read_header(stream,
                    path,
                    &header,
                    &line,
                    &line_capacity) != 0) {
        goto cleanup;
    }
    if (header.rows > SIZE_MAX / sizeof(*degrees)) {
        fprintf(stderr, "%s: row count is too large for this system\n", path);
        goto cleanup;
    }

    degrees = calloc((size_t)header.rows, sizeof(*degrees));
    if (!degrees && header.rows != 0) {
        fprintf(stderr,
                "%s: cannot allocate %.1f MiB for row counters\n",
                path,
                (double)header.rows * sizeof(*degrees) / (1024.0 * 1024.0));
        goto cleanup;
    }

    while (entries_read < header.stored_nnz &&
           getline(&line, &line_capacity, stream) >= 0) {
        uint64_t row;
        uint64_t col;

        if (line[0] == '%' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (parse_coordinate(line, &row, &col) != 0) {
            fprintf(stderr,
                    "%s: invalid coordinate at stored entry %" PRIu64 "\n",
                    path,
                    entries_read + 1);
            goto cleanup;
        }
        if (row == 0 || row > header.rows || col == 0 ||
            col > header.cols) {
            fprintf(stderr,
                    "%s: coordinate (%" PRIu64 ", %" PRIu64
                    ") is outside the matrix\n",
                    path,
                    row,
                    col);
            goto cleanup;
        }

        row--;
        col--;
        if (degrees[row] == UINT32_MAX) {
            fprintf(stderr, "%s: row %" PRIu64 " degree overflow\n", path, row + 1);
            goto cleanup;
        }
        degrees[row]++;
        if (degrees[row] > maximum) {
            maximum = degrees[row];
        }
        expanded_nnz++;

        if (header.symmetric && row != col) {
            if (degrees[col] == UINT32_MAX) {
                fprintf(stderr,
                        "%s: row %" PRIu64 " degree overflow\n",
                        path,
                        col + 1);
                goto cleanup;
            }
            degrees[col]++;
            if (degrees[col] > maximum) {
                maximum = degrees[col];
            }
            expanded_nnz++;
        }
        entries_read++;
    }

    if (entries_read != header.stored_nnz) {
        fprintf(stderr,
                "%s: expected %" PRIu64 " stored entries, found %" PRIu64 "\n",
                path,
                header.stored_nnz,
                entries_read);
        goto cleanup;
    }

    {
        char name[256];

        printf("%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",",
               matrix_name(path, name, sizeof(name)),
               header.rows,
               header.cols,
               header.stored_nnz,
               expanded_nnz);
        if (header.rows == 0) {
            printf("NA,NA,NA,0\n");
        } else {
            uint32_t upper =
                select_degree(degrees, header.rows, header.rows / 2, maximum);
            const long double mean =
                (long double)expanded_nnz / (long double)header.rows;
            long double squared_deviation_sum = 0.0L;

            for (uint64_t row = 0; row < header.rows; row++) {
                const long double difference =
                    (long double)degrees[row] - mean;
                squared_deviation_sum += difference * difference;
            }
            const long double standard_deviation =
                sqrtl(squared_deviation_sum / (long double)header.rows);

            if (header.rows % 2 != 0) {
                printf("%" PRIu32, upper);
            } else {
                uint32_t lower = select_degree(
                    degrees, header.rows, header.rows / 2 - 1, maximum);
                uint64_t sum = (uint64_t)lower + upper;
                if (sum % 2 == 0) {
                    printf("%" PRIu64, sum / 2);
                } else {
                    printf("%" PRIu64 ".5", sum / 2);
                }
            }
            printf(",%.6Lf,%.6Lf,%" PRIu32 "\n",
                   mean,
                   standard_deviation,
                   maximum);
        }
        fflush(stdout);
    }

    fprintf(stderr,
            "Finished %s: expanded nnz=%" PRIu64 ", max=%" PRIu32 "\n",
            path,
            expanded_nnz,
            maximum);
    status = 0;

cleanup:
    free(degrees);
    free(line);
    if (stream) {
        fclose(stream);
    }
    free(input_buffer);
    return status;
}

int main(int argc, char **argv) {
    int status = 0;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s MATRIX.mtx [MATRIX.mtx ...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    puts("matrix,rows,columns,stored_nnz,expanded_nnz,median,mean,std,max");
    for (int argument = 1; argument < argc; argument++) {
        if (analyze_matrix(argv[argument]) != 0) {
            status = 1;
        }
    }
    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
