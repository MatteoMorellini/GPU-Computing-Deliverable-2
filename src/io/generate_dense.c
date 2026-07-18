#include <stdio.h>
#include <stdlib.h>
#include "generate_dense.h"

void fill_dense(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        x[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    }
}

void free_dense(float* dense){
    free(dense);
}