MPICC ?= mpicc
MPICXX ?= mpicxx
NVCC ?= nvcc
GRAPH500_DIR ?= graph500

.DEFAULT_GOAL := all

CFLAGS = -Wall -Wextra -O2 -Iinclude
NVCCFLAGS = -O2 -arch=sm_80 -Iinclude
HOSTCFLAGS = -std=c11 -Wall -Wextra -O3

# Optional NCCL backend: make NCCL=1 [NCCL_ROOT=/path/to/nccl]
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

HEADERS = $(wildcard include/*.h include/*.cuh)
CUDA_SRC = $(wildcard src/mpi/*.cu src/kernels/*.cu)

LIB_OBJ = src/io/mtx_reader.o \
          src/io/coo_to_csr.o \
          src/io/generate_dense.o \
          src/mpi/matrix_partition.o \
          src/mpi/mpi_mtx_reader.o \
          src/mpi/mpi_coo_distribution.o
MAIN_OBJ = src/mpi/distribute_mtx.o \
           src/tools/analyze_partition_nnz.o \
           src/tools/analyze_partition_communication.o
# The analysis tools only need the serial reader and the partitioners.
TOOL_OBJ = src/io/mtx_reader.o src/mpi/matrix_partition.o

GRAPH500_SRC = $(addprefix $(GRAPH500_DIR)/generator/, \
                 graph_generator.c splittable_mrg.c utils.c)
GRAPH500_DEPS = $(wildcard $(GRAPH500_DIR)/generator/*.h) \
                $(GRAPH500_DIR)/generator/mrg_transitions.c

MPI_BINS = bin/distribute_mtx bin/analyze_partition_nnz \
           bin/analyze_partition_communication
CUDA_BINS = bin/mpi_spmv_cuda bin/mpi_spmv_cuda_weak

all: $(MPI_BINS) $(CUDA_BINS)

rmat: bin/generate_rmat

matrix-stats: bin/matrix_row_stats

bin:
	mkdir -p bin

%.o: %.c $(HEADERS)
	$(MPICC) $(CFLAGS) -c $< -o $@

bin/distribute_mtx: src/mpi/distribute_mtx.o $(LIB_OBJ)
bin/analyze_partition_nnz: src/tools/analyze_partition_nnz.o $(TOOL_OBJ)
bin/analyze_partition_communication: src/tools/analyze_partition_communication.o \
                                     $(TOOL_OBJ)

$(MPI_BINS): | bin
	$(MPICC) $(CFLAGS) $^ -o $@ $(METIS_LDFLAGS) -lm

# spmv_mpi_cuda_weak.cu #includes spmv_mpi_cuda.cu, so each binary compiles
# only its own entry-point translation unit plus the shared kernel runners.
bin/mpi_spmv_cuda: MAIN_SRC = src/mpi/spmv_mpi_cuda.cu
bin/mpi_spmv_cuda_weak: MAIN_SRC = src/mpi/spmv_mpi_cuda_weak.cu

$(CUDA_BINS): $(CUDA_SRC) $(LIB_OBJ) $(HEADERS) | bin
	$(NVCC) $(NVCCFLAGS) -ccbin $(MPICXX) \
		$(MAIN_SRC) src/kernels/spmv_kernel_runners.cu $(LIB_OBJ) \
		-o $@ -lcusparse $(NCCL_LDFLAGS) $(METIS_LDFLAGS)

bin/generate_rmat: src/tools/generate_rmat.c $(GRAPH500_SRC) $(GRAPH500_DEPS) | bin
	$(CC) $(HOSTCFLAGS) -Wno-unused-parameter -I$(GRAPH500_DIR)/generator \
		src/tools/generate_rmat.c $(GRAPH500_SRC) -o $@

bin/matrix_row_stats: src/tools/matrix_row_stats.c | bin
	$(CC) $(HOSTCFLAGS) $< -o $@ -lm

clean:
	rm -rf bin
	rm -f $(LIB_OBJ) $(MAIN_OBJ)

.PHONY: all clean matrix-stats rmat
