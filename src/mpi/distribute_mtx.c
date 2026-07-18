#include <mpi.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi_coo_distribution.h"
#include "mtx_reader.h"

#define MATRICES_DIR "matrices"

static void print_local_summary(const LocalCOO_Matrix *local, int rank, int size) {
    printf("Rank %d/%d owns rows i where i %% %d == %d and received %d entries\n",
           rank, size, size, rank, local->local_nnz);

    if (local->local_nnz > 0) {
        int first = 0;
        int last = local->local_nnz - 1;
        printf("Rank %d first local entry: row=%d col=%d value=%g\n",
               rank, local->row[first], local->col[first], local->data[first]);
        printf("Rank %d last local entry:  row=%d col=%d value=%g\n",
               rank, local->row[last], local->col[last], local->data[last]);
    }
}

static int has_mtx_extension(const char *name) {
    const char *ext = strrchr(name, '.');
    return ext && strcmp(ext, ".mtx") == 0;
}

static int compare_strings(const void *lhs, const void *rhs) {
    const char *a = *(const char *const *)lhs;
    const char *b = *(const char *const *)rhs;
    return strcmp(a, b);
}

static void free_matrix_paths(char **paths, int count) {
    for (int i = 0; i < count; i++) {
        free(paths[i]);
    }
    free(paths);
}

static int list_matrix_paths(char ***paths_out, int *count_out) {
    DIR *dir = opendir(MATRICES_DIR);
    if (!dir) {
        perror("Error opening matrices directory");
        return 1;
    }

    char **paths = NULL;
    int count = 0;
    int capacity = 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.' || !has_mtx_extension(entry->d_name)) {
            continue;
        }

        if (count == capacity) {
            int next_capacity = capacity == 0 ? 8 : capacity * 2;
            char **next_paths =
                realloc(paths, (size_t)next_capacity * sizeof(char *));
            if (!next_paths) {
                closedir(dir);
                free_matrix_paths(paths, count);
                return 1;
            }
            paths = next_paths;
            capacity = next_capacity;
        }

        size_t needed = strlen(MATRICES_DIR) + 1 + strlen(entry->d_name) + 1;
        paths[count] = malloc(needed);
        if (!paths[count]) {
            closedir(dir);
            free_matrix_paths(paths, count);
            return 1;
        }
        snprintf(paths[count], needed, "%s/%s", MATRICES_DIR, entry->d_name);
        count++;
    }

    closedir(dir);
    qsort(paths, (size_t)count, sizeof(char *), compare_strings);

    *paths_out = paths;
    *count_out = count;
    return 0;
}

static void broadcast_matrix_paths(char ***paths,
                                   int *count,
                                   int rank,
                                   MPI_Comm comm) {
    MPI_Bcast(count, 1, MPI_INT, 0, comm);

    if (rank != 0) {
        *paths = calloc((size_t)*count, sizeof(char *));
        if (*count > 0 && !*paths) {
            fprintf(stderr, "Rank %d could not allocate matrix path list\n", rank);
            MPI_Abort(comm, 1);
        }
    }

    for (int i = 0; i < *count; i++) {
        int path_len = 0;
        if (rank == 0) {
            path_len = (int)strlen((*paths)[i]) + 1;
        }

        MPI_Bcast(&path_len, 1, MPI_INT, 0, comm);

        if (rank != 0) {
            (*paths)[i] = malloc((size_t)path_len);
            if (!(*paths)[i]) {
                fprintf(stderr, "Rank %d could not allocate matrix path\n", rank);
                MPI_Abort(comm, 1);
            }
        }

        MPI_Bcast((*paths)[i], path_len, MPI_CHAR, 0, comm);
    }
}

static void run_distribution_for_matrix(const char *matrix_path,
                                        int rank,
                                        int size) {
    COO_Matrix global = {0};
    double read_seconds = 0.0;

    if (rank == 0) {
        printf("\n=== %s ===\n", matrix_path);

        double read_start = MPI_Wtime();
        read_mtx(matrix_path, &global);
        read_seconds = MPI_Wtime() - read_start;

        printf("Rank 0 read %s: %d rows, %d cols, %d non-zeros in %.6f s\n",
               matrix_path, global.rows, global.cols, global.nnz, read_seconds);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_start = MPI_Wtime();

    LocalCOO_Matrix local = {0};
    if (distribute_coo_entries(&global, &local, 0, MPI_COMM_WORLD) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error distributing COO entries for %s\n",
                    matrix_path);
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

    for (int printing_rank = 0; printing_rank < size; printing_rank++) {
        if (rank == printing_rank) {
            print_local_summary(&local, rank, size);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Distributed %d/%d entries across %d ranks in %.6f s\n",
               total_distributed, global.nnz, size, scatter_seconds);
    }

    free_local_coo(&local);
    if (rank == 0) {
        free_coo(&global);
    }
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char **matrix_paths = NULL;
    int matrix_count = 0;
    int list_error = 0;

    if (rank == 0) {
        list_error = list_matrix_paths(&matrix_paths, &matrix_count);
    }

    MPI_Bcast(&list_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (list_error) {
        if (rank == 0) {
            fprintf(stderr, "Error listing .mtx files in %s/\n", MATRICES_DIR);
        }
        free_matrix_paths(matrix_paths, matrix_count);
        MPI_Finalize();
        return 1;
    }

    broadcast_matrix_paths(&matrix_paths, &matrix_count, rank, MPI_COMM_WORLD);

    if (matrix_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "No .mtx files found in %s/\n", MATRICES_DIR);
        }
        free_matrix_paths(matrix_paths, matrix_count);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Running COO distribution for %d Matrix Market files in %s/\n",
               matrix_count, MATRICES_DIR);
    }

    for (int i = 0; i < matrix_count; i++) {
        run_distribution_for_matrix(matrix_paths[i], rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free_matrix_paths(matrix_paths, matrix_count);
    MPI_Finalize();
    return 0;
}
