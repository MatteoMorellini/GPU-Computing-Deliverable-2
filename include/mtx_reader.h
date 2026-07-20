#ifndef MTX_READER_H
#define MTX_READER_H

typedef struct {
    int rows;
    int cols;
    int nnz;
    int *row;
    int *col;
    double *data;
} COO_Matrix;

#ifdef __cplusplus
extern "C" {
#endif

void read_mtx(const char *filename, COO_Matrix *mat);
/*
 * Read a disjoint byte range of the Matrix Market data section.
 *
 * Every rank calls this function with the same filename and process count.
 * Returned row/column indices remain global.  Lines that cross byte-range
 * boundaries are owned by exactly one rank, and symmetric entries are expanded
 * in the same way as read_mtx().
 *
 * The function returns zero on success instead of terminating the process so
 * that an MPI caller can propagate failures collectively.
 */
int read_mtx_chunk(const char *filename,
                   COO_Matrix *mat,
                   int *declared_stored_nnz,
                   int *local_stored_nnz,
                   int rank,
                   int size);
void free_coo(COO_Matrix *mat);

#ifdef __cplusplus
}
#endif

#endif
