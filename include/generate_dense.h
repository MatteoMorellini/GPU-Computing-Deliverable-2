#include <stdio.h>
#include <stdlib.h>
#ifndef GENERATE_DENSE_H
#define GENERATE_DENSE_H

#ifdef __cplusplus
extern "C" {
#endif

void fill_dense(float *x, size_t n);
void free_dense(float* dense);

#ifdef __cplusplus
}
#endif

#endif
