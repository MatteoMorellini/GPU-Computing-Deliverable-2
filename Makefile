MPICC ?= mpicc
MPICXX ?= mpicxx
MPIRUN ?= mpirun
NVCC ?= nvcc
RMAT_CC ?= cc
GRAPH500_DIR ?= graph500

.DEFAULT_GOAL := all

CFLAGS = -Wall -Wextra -O2 -Iinclude
NVCCFLAGS = -O2 -arch=sm_80 -Iinclude
METIS_ROOT ?= $(EBROOTMETIS)
ifneq ($(strip $(METIS_ROOT)),)
CFLAGS += -I$(METIS_ROOT)/include
NVCCFLAGS += -I$(METIS_ROOT)/include
METIS_LDFLAGS = -L$(METIS_ROOT)/lib -lmetis
else
METIS_LDFLAGS = -lmetis
endif

DISTRIBUTE_TARGET = bin/distribute_mtx
SPMV_TARGET = bin/mpi_spmv_cuda
WEAK_SPMV_TARGET = bin/mpi_spmv_cuda_weak
MPI_IO_TEST_TARGET = bin/test_mpi_io_reader
PARTITION_TEST_TARGET = bin/test_matrix_partition
RMAT_TARGET = bin/generate_rmat

LIB_SRC = src/io/mtx_reader.c \
          src/io/coo_to_csr.c \
          src/io/generate_dense.c \
          src/mpi/matrix_partition.c \
          src/mpi/mpi_mtx_reader.c \
          src/mpi/mpi_coo_distribution.c
DISTRIBUTE_MAIN_SRC = src/mpi/distribute_mtx.c
SPMV_CUDA_SRC = src/mpi/spmv_mpi_cuda.cu \
                src/kernels/spmv_kernel_runners.cu
WEAK_SPMV_CUDA_SRC = src/mpi/spmv_mpi_cuda_weak.cu \
                     src/kernels/spmv_kernel_runners.cu
MPI_IO_TEST_SRC = tests/test_mpi_io_reader.c
PARTITION_TEST_SRC = tests/test_matrix_partition.c
RMAT_SRC = src/tools/generate_rmat.c
GRAPH500_GENERATOR_SRC = $(GRAPH500_DIR)/generator/graph_generator.c \
                         $(GRAPH500_DIR)/generator/splittable_mrg.c \
                         $(GRAPH500_DIR)/generator/utils.c
GRAPH500_GENERATOR_HEADERS = $(wildcard $(GRAPH500_DIR)/generator/*.h) \
                             $(GRAPH500_DIR)/generator/mrg_transitions.c
HEADERS = $(wildcard include/*.h include/*.cuh)

LIB_OBJ = $(LIB_SRC:.c=.o)
DISTRIBUTE_MAIN_OBJ = $(DISTRIBUTE_MAIN_SRC:.c=.o)

all: $(DISTRIBUTE_TARGET) $(SPMV_TARGET) $(WEAK_SPMV_TARGET)

bin:
	mkdir -p bin

%.o: %.c $(HEADERS)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(DISTRIBUTE_TARGET): $(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ) | bin
	$(MPICC) $(CFLAGS) $(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ) -o $@ $(METIS_LDFLAGS) -lm

$(SPMV_TARGET): $(LIB_OBJ) $(SPMV_CUDA_SRC) $(HEADERS) | bin
	$(NVCC) $(NVCCFLAGS) -ccbin $(MPICXX) $(SPMV_CUDA_SRC) $(LIB_OBJ) -o $@ -lcusparse $(METIS_LDFLAGS)

$(WEAK_SPMV_TARGET): $(LIB_OBJ) $(WEAK_SPMV_CUDA_SRC) \
                     src/mpi/spmv_mpi_cuda.cu $(HEADERS) | bin
	$(NVCC) $(NVCCFLAGS) -ccbin $(MPICXX) $(WEAK_SPMV_CUDA_SRC) $(LIB_OBJ) -o $@ -lcusparse $(METIS_LDFLAGS)

$(MPI_IO_TEST_TARGET): $(LIB_OBJ) $(MPI_IO_TEST_SRC) $(HEADERS) | bin
	$(MPICC) $(CFLAGS) $(LIB_OBJ) $(MPI_IO_TEST_SRC) -o $@ $(METIS_LDFLAGS) -lm

$(PARTITION_TEST_TARGET): $(LIB_OBJ) $(PARTITION_TEST_SRC) $(HEADERS) | bin
	$(MPICC) $(CFLAGS) $(LIB_OBJ) $(PARTITION_TEST_SRC) -o $@ $(METIS_LDFLAGS) -lm

$(RMAT_TARGET): $(RMAT_SRC) $(GRAPH500_GENERATOR_SRC) \
                $(GRAPH500_GENERATOR_HEADERS) | bin
	$(RMAT_CC) -std=c11 -Wall -Wextra -Wno-unused-parameter -O3 \
		-I$(GRAPH500_DIR)/generator \
		$(RMAT_SRC) $(GRAPH500_GENERATOR_SRC) -o $@

rmat: $(RMAT_TARGET)

test-mpi-io: $(MPI_IO_TEST_TARGET)
	@for ranks in 1 2 3 4 5; do \
		$(MPIRUN) -np $$ranks ./$(MPI_IO_TEST_TARGET) \
			dummy_matrix/tiny.mtx \
			tests/data/mpi_io_general.mtx \
			tests/data/mpi_io_symmetric_pattern.mtx \
			tests/data/mpi_io_single_entry.mtx || exit $$?; \
	done

test-rmat: $(RMAT_TARGET)
	bash tests/test_generate_rmat.sh ./$(RMAT_TARGET)

test-partitions: $(PARTITION_TEST_TARGET)
	@for ranks in 1 2 3 4 6; do \
		$(MPIRUN) -np $$ranks ./$(PARTITION_TEST_TARGET) || exit $$?; \
	done

clean:
	rm -f $(DISTRIBUTE_TARGET) $(SPMV_TARGET) $(WEAK_SPMV_TARGET) \
		$(MPI_IO_TEST_TARGET) $(RMAT_TARGET) \
		$(PARTITION_TEST_TARGET) \
		$(LIB_OBJ) $(DISTRIBUTE_MAIN_OBJ)

.PHONY: all clean rmat test-mpi-io test-rmat test-partitions
