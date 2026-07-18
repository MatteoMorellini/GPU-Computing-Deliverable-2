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
void free_coo(COO_Matrix *mat);

#ifdef __cplusplus
}
#endif

#endif
