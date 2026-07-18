#ifndef SPMV_KERNEL_RUNNER_CUH
#define SPMV_KERNEL_RUNNER_CUH

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>

extern "C" {
#include "coo_to_csr.h"
}

enum SpmvKernelKind {
    SPMV_KERNEL_SCALAR = 0,
    SPMV_KERNEL_VECTOR,
    SPMV_KERNEL_ADAPTIVE,
    SPMV_KERNEL_ADAPTIVE_PAPER,
    SPMV_KERNEL_PARTIAL,
    SPMV_KERNEL_CUSPARSE
};

struct SpmvKernelRunner {
    virtual ~SpmvKernelRunner() {}

    virtual const char *name() const = 0;

    virtual double prep(const CSR_Matrix &h_csr,
                        const CSR_Matrix &d_csr,
                        float *d_x,
                        float *d_y) = 0;

    virtual void launch(const CSR_Matrix &d_csr,
                        const float *d_x,
                        float *d_y) = 0;

    virtual void teardown() = 0;
};

const char *spmv_kernel_kind_name(SpmvKernelKind kind);
int parse_spmv_kernel_kind(const char *name, SpmvKernelKind *kind);
SpmvKernelRunner *create_spmv_kernel_runner(SpmvKernelKind kind, int rank);
void destroy_spmv_kernel_runner(SpmvKernelRunner *runner);
void print_spmv_kernel_choices(FILE *stream);

#endif
