#include <stdio.h>
// we import mtx_reader.h to guarantee that the implementation matches 
// the declared interface
#include <stdlib.h>
#include "coo_to_csr.h"

void coo_to_csr(COO_Matrix *coo, CSR_Matrix *csr){
    csr->rows = coo->rows;
    csr->cols = coo->cols;
    csr->nnz = coo->nnz;
        
    // zero-initialize row_ptr 
    csr->row_ptr = calloc(coo->rows + 1, sizeof(int));
    csr->col_idx = csr->nnz > 0 ? malloc(csr->nnz * sizeof(int)) : NULL;
    csr->values = csr->nnz > 0 ? malloc(csr->nnz * sizeof(float)) : NULL;
    if (!csr->row_ptr || (csr->nnz > 0 && (!csr->col_idx || !csr->values))){
        fprintf(stderr, "Error allocating memory for CSR matrix\n");
        exit(1);
    }

    for (int i = 0; i < coo->nnz; i++) {
        csr->row_ptr[coo->row[i] + 1]++;
    }

    for (int i = 0; i < csr->rows; i++) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }
        
    // rows might be out-of-order, so we need to keep track of the next 
    // available position for each row
    int *next = csr->rows > 0 ? malloc(csr->rows * sizeof(int)) : NULL;
    if (csr->rows > 0 && !next) {
        fprintf(stderr, "Error allocating CSR row cursor\n");
        exit(1);
    }
    for (int i = 0; i < csr->rows; i++) {
        next[i] = csr->row_ptr[i];
    }

    for (int i = 0; i < coo->nnz; i++) {
        int row = coo->row[i];
        int dest = next[row];

        csr->col_idx[dest] = coo->col[i];
        csr->values[dest]  = (float) coo->data[i];

        next[row]++;
    }

    free(next);
}

void free_csr(CSR_Matrix *csr){
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    csr->row_ptr = NULL;
    csr->col_idx = NULL;
    csr->values = NULL;
    csr->rows = 0;
    csr->cols = 0;
    csr->nnz = 0;
}
