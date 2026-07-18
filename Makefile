MPICC ?= mpicc
MPICXX ?= mpicxx
NVCC ?= nvcc

.DEFAULT_GOAL := all

CFLAGS = -Wall -Wextra -O2 -Iinclude
NVCCFLAGS = -O2 -arch=sm_80 -Iinclude

DISTRIBUTE_TARGET = bin/distribute_mtx
SPMV_TARGET = bin/mpi_spmv_cuda

LIB_SRC = src/io/mtx_reader.c \
          src/io/coo_to_csr.c \
          src/io/generate_dense.c \
          src/mpi/mpi_coo_distribution.c
DISTRIBUTE_MAIN_SRC = src/mpi/distribute_mtx.c
SPMV_CUDA_SRC = src/mpi/spmv_mpi_cuda.cu \
                src/kernels/spmv_kernel_runners.cu
HEADERS = $(wildcard include/*.h include/*.cuh)

LIB_OBJ = $(LIB_SRC:.c=.o)
DISTRIBUTE_MAIN_OBJ = $(DISTRIBUTE_MAIN_SRC:.c=.o)

all: $(DISTRIBUTE_TARGET) $(SPMV_TARGET)

bin:
	mkdir -p bin

%.o: %.c $(HEADERS)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(DISTRIBUTE_TARGET): $(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ) | bin
	$(MPICC) $(CFLAGS) $(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ) -o $@

$(SPMV_TARGET): $(LIB_OBJ) $(SPMV_CUDA_SRC) $(HEADERS) | bin
	$(NVCC) $(NVCCFLAGS) -ccbin $(MPICXX) $(SPMV_CUDA_SRC) $(LIB_OBJ) -o $@ -lcusparse

clean:
	rm -f $(DISTRIBUTE_TARGET) $(SPMV_TARGET) $(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ)

.PHONY: all clean
