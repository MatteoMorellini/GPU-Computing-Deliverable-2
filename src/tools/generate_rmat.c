#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph_generator.h"
#include "utils.h"

typedef enum {
    OUTPUT_SYMMETRIC,
    OUTPUT_GENERAL,
} OutputFormat;

typedef struct {
    int scale;
    uint64_t edge_factor;
    int edge_factor_set;
    uint64_t edge_count;
    int edge_count_set;
    uint64_t seed1;
    uint64_t seed2;
    size_t chunk_edges;
    const char *output_path;
    OutputFormat format;
    int drop_self_loops;
    int force;
} Config;

static void print_usage(FILE *stream, const char *program) {
    fprintf(stream,
            "Usage: %s --scale N --output FILE [options]\n"
            "\n"
            "Generate Graph500 R-MAT edge tuples with initiator\n"
            "a=0.57, b=0.19, c=0.19, d=0.05 and write Matrix Market.\n"
            "\n"
            "Required:\n"
            "  --scale N              log2(number of vertices), 1..30\n"
            "  --output FILE          output Matrix Market file\n"
            "\n"
            "Graph size (choose at most one):\n"
            "  --edge-factor N        tuples / vertices (default: 16)\n"
            "  --edges N              exact number of generated tuples\n"
            "\n"
            "Reproducibility and output:\n"
            "  --seed1 N              first Graph500 seed (default: 2)\n"
            "  --seed2 N              second Graph500 seed (default: 3)\n"
            "  --format symmetric     canonicalize each tuple and use a\n"
            "                         symmetric header (default)\n"
            "  --format general       write one directed entry per tuple\n"
            "  --drop-self-loops      omit tuples whose endpoints match\n"
            "  --chunk-edges N        generation buffer size (default: 1048576)\n"
            "  --force                overwrite an existing output file\n"
            "  --help                 show this help\n"
            "\n"
            "Parallel edges are deliberately retained, matching the raw\n"
            "Graph500 generator. Matrix Market indices are one-based.\n",
            program);
}

static int parse_u64(const char *text, uint64_t *value) {
    char *end = NULL;

    if (!text || text[0] == '\0' || text[0] == '-') {
        return 0;
    }

    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || !end || *end != '\0') {
        return 0;
    }

    *value = (uint64_t)parsed;
    return 1;
}

static int require_value(int argc,
                         char **argv,
                         int *index,
                         const char **value) {
    if (*index + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", argv[*index]);
        return 0;
    }
    *value = argv[++(*index)];
    return 1;
}

static int parse_arguments(int argc, char **argv, Config *config) {
    *config = (Config){
        .scale = -1,
        .edge_factor = 16,
        .seed1 = 2,
        .seed2 = 3,
        .chunk_edges = 1024U * 1024U,
        .format = OUTPUT_SYMMETRIC,
    };

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        const char *value = NULL;
        uint64_t parsed = 0;

        if (strcmp(arg, "--help") == 0) {
            print_usage(stdout, argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(arg, "--drop-self-loops") == 0) {
            config->drop_self_loops = 1;
        } else if (strcmp(arg, "--force") == 0) {
            config->force = 1;
        } else if (strcmp(arg, "--scale") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_u64(value, &parsed) || parsed > INT_MAX) {
                fprintf(stderr, "Invalid --scale value\n");
                return 0;
            }
            config->scale = (int)parsed;
        } else if (strcmp(arg, "--edge-factor") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_u64(value, &parsed) || parsed == 0) {
                fprintf(stderr, "Invalid --edge-factor value\n");
                return 0;
            }
            config->edge_factor = parsed;
            config->edge_factor_set = 1;
        } else if (strcmp(arg, "--edges") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_u64(value, &parsed) || parsed == 0) {
                fprintf(stderr, "Invalid --edges value\n");
                return 0;
            }
            config->edge_count = parsed;
            config->edge_count_set = 1;
        } else if (strcmp(arg, "--seed1") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_u64(value, &config->seed1)) {
                fprintf(stderr, "Invalid --seed1 value\n");
                return 0;
            }
        } else if (strcmp(arg, "--seed2") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_u64(value, &config->seed2)) {
                fprintf(stderr, "Invalid --seed2 value\n");
                return 0;
            }
        } else if (strcmp(arg, "--chunk-edges") == 0) {
            if (!require_value(argc, argv, &i, &value) ||
                !parse_u64(value, &parsed) || parsed == 0 ||
                parsed > SIZE_MAX) {
                fprintf(stderr, "Invalid --chunk-edges value\n");
                return 0;
            }
            config->chunk_edges = (size_t)parsed;
        } else if (strcmp(arg, "--output") == 0) {
            if (!require_value(argc, argv, &i, &config->output_path) ||
                config->output_path[0] == '\0') {
                fprintf(stderr, "Invalid --output value\n");
                return 0;
            }
        } else if (strcmp(arg, "--format") == 0) {
            if (!require_value(argc, argv, &i, &value)) {
                return 0;
            }
            if (strcmp(value, "symmetric") == 0) {
                config->format = OUTPUT_SYMMETRIC;
            } else if (strcmp(value, "general") == 0) {
                config->format = OUTPUT_GENERAL;
            } else {
                fprintf(stderr,
                        "Invalid --format value '%s' (expected symmetric or "
                        "general)\n",
                        value);
                return 0;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            return 0;
        }
    }

    if (config->scale < 1 || config->scale > 30) {
        fprintf(stderr,
                "--scale must be in 1..30 for this project's 32-bit Matrix "
                "Market reader\n");
        return 0;
    }
    if (!config->output_path) {
        fprintf(stderr, "--output is required\n");
        return 0;
    }
    if (config->edge_factor_set && config->edge_count_set) {
        fprintf(stderr, "Use either --edge-factor or --edges, not both\n");
        return 0;
    }
    if (config->chunk_edges > SIZE_MAX / sizeof(packed_edge)) {
        fprintf(stderr, "--chunk-edges is too large for this platform\n");
        return 0;
    }

    return 1;
}

static int write_dimensions(FILE *output,
                            off_t dimensions_offset,
                            uint64_t vertices,
                            uint64_t stored_nnz) {
    if (fseeko(output, dimensions_offset, SEEK_SET) != 0) {
        return 0;
    }

    return fprintf(output,
                   "%20" PRIu64 " %20" PRIu64 " %20" PRIu64 "\n",
                   vertices,
                   vertices,
                   stored_nnz) > 0;
}

int main(int argc, char **argv) {
    Config config;
    if (!parse_arguments(argc, argv, &config)) {
        print_usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    const uint64_t vertices = UINT64_C(1) << config.scale;
    uint64_t requested_edges = config.edge_count;
    if (!config.edge_count_set) {
        if (config.edge_factor > UINT64_MAX / vertices) {
            fprintf(stderr, "Requested edge count overflows 64 bits\n");
            return EXIT_FAILURE;
        }
        requested_edges = config.edge_factor * vertices;
    }

    if (requested_edges > INT64_MAX) {
        fprintf(stderr, "Graph500 accepts at most INT64_MAX edge tuples\n");
        return EXIT_FAILURE;
    }
    if (requested_edges > INT_MAX) {
        fprintf(stderr,
                "Requested tuples exceed this project's 32-bit Matrix Market "
                "NNZ limit\n");
        return EXIT_FAILURE;
    }
    if (config.format == OUTPUT_SYMMETRIC &&
        requested_edges > (uint64_t)INT_MAX / 2U) {
        fprintf(stderr,
                "Symmetric expansion could exceed this project's INT_MAX "
                "in-memory NNZ limit\n");
        return EXIT_FAILURE;
    }

    packed_edge *edge_buffer =
        malloc(config.chunk_edges * sizeof(*edge_buffer));
    if (!edge_buffer) {
        fprintf(stderr,
                "Could not allocate a %zu-edge generation buffer\n",
                config.chunk_edges);
        return EXIT_FAILURE;
    }

    const char *mode = config.force ? "w+" : "wx+";
    FILE *output = fopen(config.output_path, mode);
    if (!output) {
        fprintf(stderr,
                "Could not create %s: %s%s\n",
                config.output_path,
                strerror(errno),
                config.force ? "" : " (use --force to overwrite it)");
        free(edge_buffer);
        return EXIT_FAILURE;
    }
    (void)setvbuf(output, NULL, _IOFBF, 8U * 1024U * 1024U);

    const char *symmetry =
        config.format == OUTPUT_SYMMETRIC ? "symmetric" : "general";
    if (fprintf(output,
                "%%%%MatrixMarket matrix coordinate pattern %s\n"
                "%% Graph500 reference R-MAT generator\n"
                "%% initiator: a=0.57 b=0.19 c=0.19 d=0.05\n"
                "%% scale=%d requested_edge_tuples=%" PRIu64
                " seed1=%" PRIu64 " seed2=%" PRIu64 "\n"
                "%% parallel_edges=retained self_loops=%s\n",
                symmetry,
                config.scale,
                requested_edges,
                config.seed1,
                config.seed2,
                config.drop_self_loops ? "dropped" : "retained") < 0) {
        fprintf(stderr, "Could not write Matrix Market header\n");
        goto fail;
    }

    off_t dimensions_offset = ftello(output);
    if (dimensions_offset < 0 ||
        !write_dimensions(output, dimensions_offset, vertices, 0) ||
        fseeko(output, 0, SEEK_END) != 0) {
        fprintf(stderr, "Could not reserve the Matrix Market dimensions line\n");
        goto fail;
    }

    uint_fast32_t seed[5];
    make_mrg_seed(config.seed1, config.seed2, seed);

    uint64_t stored_nnz = 0;
    uint64_t generated_self_loops = 0;
    uint64_t stored_self_loops = 0;

    for (uint64_t start = 0; start < requested_edges;) {
        uint64_t remaining = requested_edges - start;
        size_t count = remaining < config.chunk_edges ? (size_t)remaining
                                                       : config.chunk_edges;
        uint64_t end = start + count;

        generate_kronecker_range(seed,
                                 config.scale,
                                 (int64_t)start,
                                 (int64_t)end,
                                 edge_buffer);

        for (size_t i = 0; i < count; ++i) {
            int64_t row = get_v0_from_edge(&edge_buffer[i]);
            int64_t col = get_v1_from_edge(&edge_buffer[i]);
            if (row == col) {
                generated_self_loops++;
                if (config.drop_self_loops) {
                    continue;
                }
                stored_self_loops++;
            }

            if (config.format == OUTPUT_SYMMETRIC && row < col) {
                int64_t swap = row;
                row = col;
                col = swap;
            }

            if (fprintf(output,
                        "%" PRId64 " %" PRId64 "\n",
                        row + 1,
                        col + 1) < 0) {
                fprintf(stderr, "Error while writing %s\n", config.output_path);
                goto fail;
            }
            stored_nnz++;
        }

        start = end;
    }

    if (fflush(output) != 0 ||
        !write_dimensions(output,
                          dimensions_offset,
                          vertices,
                          stored_nnz) ||
        fflush(output) != 0) {
        fprintf(stderr, "Could not finalize the Matrix Market header\n");
        goto fail;
    }

    if (fclose(output) != 0) {
        output = NULL;
        fprintf(stderr, "Could not finalize %s: %s\n", config.output_path,
                strerror(errno));
        remove(config.output_path);
        free(edge_buffer);
        return EXIT_FAILURE;
    }
    output = NULL;
    free(edge_buffer);

    uint64_t expanded_nnz = stored_nnz;
    if (config.format == OUTPUT_SYMMETRIC) {
        expanded_nnz = 2U * stored_nnz - stored_self_loops;
    }

    printf("Graph500 R-MAT matrix written to %s\n", config.output_path);
    printf("  vertices:             %" PRIu64 "\n", vertices);
    printf("  generated tuples:     %" PRIu64 "\n", requested_edges);
    printf("  generated self-loops: %" PRIu64 "\n", generated_self_loops);
    printf("  stored entries:       %" PRIu64 "\n", stored_nnz);
    printf("  reader-expanded NNZ:  %" PRIu64 "\n", expanded_nnz);
    printf("  parallel edges:       retained\n");
    return EXIT_SUCCESS;

fail:
    {
        int saved_errno = errno;
        fclose(output);
        remove(config.output_path);
        free(edge_buffer);
        errno = saved_errno;
    }
    return EXIT_FAILURE;
}
