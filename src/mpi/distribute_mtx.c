#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi_coo_distribution.h"
#include "mtx_reader.h"

static void print_local_summary(const LocalCOO_Matrix *local, int rank, int size) {
    int begin = local->global_offset;
    int end = begin + local->local_nnz;

    printf("Rank %d/%d received %d entries: global COO interval [%d, %d)\n",
           rank, size, local->local_nnz, begin, end);

    if (local->local_nnz > 0) {
        int first = 0;
        int last = local->local_nnz - 1;
        printf("Rank %d first local entry: row=%d col=%d value=%g\n",
               rank, local->row[first], local->col[first], local->data[first]);
        printf("Rank %d last local entry:  row=%d col=%d value=%g\n",
               rank, local->row[last], local->col[last], local->data[last]);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <matrix-market-file.mtx>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    COO_Matrix global = {0};
    double read_seconds = 0.0;

    if (rank == 0) {
        double read_start = MPI_Wtime();
        read_mtx(argv[1], &global);
        read_seconds = MPI_Wtime() - read_start;

        printf("Rank 0 read %s: %d rows, %d cols, %d non-zeros in %.6f s\n",
               argv[1], global.rows, global.cols, global.nnz, read_seconds);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_start = MPI_Wtime();

    LocalCOO_Matrix local = {0};
    if (distribute_coo_entries(&global, &local, 0, MPI_COMM_WORLD) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error distributing COO entries\n");
        }
        free_coo(&global);
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

    if (rank == 0) {
        printf("Distributed %d/%d entries across %d ranks in %.6f s\n",
               total_distributed, global.nnz, size, scatter_seconds);
    }

    free_local_coo(&local);
    if (rank == 0) {
        free_coo(&global);
    }

    MPI_Finalize();
    return 0;
}
