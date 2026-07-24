MPICC ?= mpicc
MPICXX ?= mpicxx
MPIRUN ?= mpirun
NVCC ?= nvcc
RMAT_CC ?= cc
STATS_CC ?= cc
GRAPH500_DIR ?= graph500

.DEFAULT_GOAL := all

CFLAGS = -Wall -Wextra -O2 -Iinclude
NVCCFLAGS = -O2 -arch=sm_80 -Iinclude
NCCL ?= 0
NCCL_ROOT ?= $(EBROOTNCCL)
ifeq ($(NCCL),1)
NVCCFLAGS += -DUSE_NCCL
ifneq ($(strip $(NCCL_ROOT)),)
NVCCFLAGS += -I$(NCCL_ROOT)/include
NCCL_LIBDIR := $(if $(wildcard $(NCCL_ROOT)/lib/libnccl.so),$(NCCL_ROOT)/lib,$(NCCL_ROOT)/lib64)
NCCL_LDFLAGS = -L$(NCCL_LIBDIR) -lnccl -Xlinker -rpath -Xlinker $(NCCL_LIBDIR)
else
NCCL_LDFLAGS = -lnccl
endif
endif
METIS_ROOT ?= $(EBROOTMETIS)
ifneq ($(strip $(METIS_ROOT)),)
CFLAGS += -I$(METIS_ROOT)/include
NVCCFLAGS += -I$(METIS_ROOT)/include
METIS_LDFLAGS = -L$(METIS_ROOT)/lib -lmetis
else
METIS_LDFLAGS = -lmetis
endif

DISTRIBUTE_TARGET = bin/distribute_mtx
PARTITION_ANALYSIS_TARGET = bin/analyze_partition_nnz
COMM_ANALYSIS_TARGET = bin/analyze_partition_communication
SPMV_TARGET = bin/mpi_spmv_cuda
WEAK_SPMV_TARGET = bin/mpi_spmv_cuda_weak
RMAT_TARGET = bin/generate_rmat
ROW_STATS_TARGET = bin/matrix_row_stats

LIB_SRC = src/io/mtx_reader.c \
          src/io/coo_to_csr.c \
          src/io/generate_dense.c \
          src/mpi/matrix_partition.c \
          src/mpi/mpi_mtx_reader.c \
          src/mpi/mpi_coo_distribution.c
DISTRIBUTE_MAIN_SRC = src/mpi/distribute_mtx.c
PARTITION_ANALYSIS_SRC = src/tools/analyze_partition_nnz.c
COMM_ANALYSIS_SRC = src/tools/analyze_partition_communication.c
SPMV_CUDA_SRC = src/mpi/spmv_mpi_cuda.cu \
                src/kernels/spmv_kernel_runners.cu
WEAK_SPMV_CUDA_SRC = src/mpi/spmv_mpi_cuda_weak.cu \
                     src/kernels/spmv_kernel_runners.cu
RMAT_SRC = src/tools/generate_rmat.c
ROW_STATS_SRC = src/tools/matrix_row_stats.c
GRAPH500_GENERATOR_SRC = $(GRAPH500_DIR)/generator/graph_generator.c \
                         $(GRAPH500_DIR)/generator/splittable_mrg.c \
                         $(GRAPH500_DIR)/generator/utils.c
GRAPH500_GENERATOR_HEADERS = $(wildcard $(GRAPH500_DIR)/generator/*.h) \
                             $(GRAPH500_DIR)/generator/mrg_transitions.c
HEADERS = $(wildcard include/*.h include/*.cuh)

LIB_OBJ = $(LIB_SRC:.c=.o)
DISTRIBUTE_MAIN_OBJ = $(DISTRIBUTE_MAIN_SRC:.c=.o)
PARTITION_ANALYSIS_OBJ = $(PARTITION_ANALYSIS_SRC:.c=.o)
COMM_ANALYSIS_OBJ = $(COMM_ANALYSIS_SRC:.c=.o)

all: $(DISTRIBUTE_TARGET) $(PARTITION_ANALYSIS_TARGET) \
     $(COMM_ANALYSIS_TARGET) $(SPMV_TARGET) $(WEAK_SPMV_TARGET)

bin:
	mkdir -p bin

%.o: %.c $(HEADERS)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(DISTRIBUTE_TARGET): $(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ) | bin
	$(MPICC) $(CFLAGS) $(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ) -o $@ $(METIS_LDFLAGS) -lm

$(PARTITION_ANALYSIS_TARGET): src/io/mtx_reader.o \
                              src/mpi/matrix_partition.o \
                              $(PARTITION_ANALYSIS_OBJ) | bin
	$(MPICC) $(CFLAGS) $^ -o $@ $(METIS_LDFLAGS) -lm

$(COMM_ANALYSIS_TARGET): src/io/mtx_reader.o \
                         src/mpi/matrix_partition.o \
                         $(COMM_ANALYSIS_OBJ) | bin
	$(MPICC) $(CFLAGS) $^ -o $@ $(METIS_LDFLAGS) -lm

$(SPMV_TARGET): $(LIB_OBJ) $(SPMV_CUDA_SRC) $(HEADERS) | bin
	$(NVCC) $(NVCCFLAGS) -ccbin $(MPICXX) $(SPMV_CUDA_SRC) $(LIB_OBJ) -o $@ -lcusparse $(NCCL_LDFLAGS) $(METIS_LDFLAGS)

$(WEAK_SPMV_TARGET): $(LIB_OBJ) $(WEAK_SPMV_CUDA_SRC) \
                     src/mpi/spmv_mpi_cuda.cu $(HEADERS) | bin
	$(NVCC) $(NVCCFLAGS) -ccbin $(MPICXX) $(WEAK_SPMV_CUDA_SRC) $(LIB_OBJ) -o $@ -lcusparse $(NCCL_LDFLAGS) $(METIS_LDFLAGS)

$(RMAT_TARGET): $(RMAT_SRC) $(GRAPH500_GENERATOR_SRC) \
                $(GRAPH500_GENERATOR_HEADERS) | bin
	$(RMAT_CC) -std=c11 -Wall -Wextra -Wno-unused-parameter -O3 \
		-I$(GRAPH500_DIR)/generator \
		$(RMAT_SRC) $(GRAPH500_GENERATOR_SRC) -o $@

$(ROW_STATS_TARGET): $(ROW_STATS_SRC) | bin
	$(STATS_CC) -std=c11 -Wall -Wextra -O3 $(ROW_STATS_SRC) -o $@ -lm

rmat: $(RMAT_TARGET)

matrix-stats: $(ROW_STATS_TARGET)

clean:
	rm -f $(DISTRIBUTE_TARGET) $(PARTITION_ANALYSIS_TARGET) \
		$(COMM_ANALYSIS_TARGET) $(SPMV_TARGET) $(WEAK_SPMV_TARGET) \
		$(RMAT_TARGET) \
		$(ROW_STATS_TARGET) \
		$(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ) \
		$(PARTITION_ANALYSIS_OBJ) $(COMM_ANALYSIS_OBJ)

.PHONY: all clean matrix-stats rmat
