#ifndef MPI_MTX_READER_H
#define MPI_MTX_READER_H

#include <mpi.h>

#include "mtx_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* MPI file open, metadata, boundary discovery, collective read, close. */
    double file_io_s;

    /* Conversion of the in-memory text buffer into COO entries. */
    double parse_s;
} MpiMtxReadMetrics;

/*
 * Collectively read the Matrix Market data section with MPI-IO.
 *
 * Rank 0 parses the header and broadcasts its metadata.  The data section is
 * split into balanced nominal byte ranges, whose boundaries are advanced to
 * the next complete line.  Every rank then reads its exact line-aligned range
 * with MPI_File_read_at_all and parses it into a temporary COO matrix.
 *
 * Returned row and column indices remain global.  Symmetric entries are
 * expanded in the same way as read_mtx().  declared_stored_nnz counts entries
 * declared by the file, while local_stored_nnz counts source lines parsed by
 * this rank before symmetric expansion.
 */
int read_mtx_chunk_mpi_io(const char *filename,
                          COO_Matrix *mat,
                          int *declared_stored_nnz,
                          int *local_stored_nnz,
                          MPI_Comm comm);
int read_mtx_chunk_mpi_io_timed(const char *filename,
                                COO_Matrix *mat,
                                int *declared_stored_nnz,
                                int *local_stored_nnz,
                                MpiMtxReadMetrics *metrics,
                                MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif
