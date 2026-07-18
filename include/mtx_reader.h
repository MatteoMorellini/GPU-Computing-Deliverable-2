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

void read_mtx(const char *filename, COO_Matrix *mat);
void free_coo(COO_Matrix *mat);

#endif
