MPICC ?= mpicc

.DEFAULT_GOAL := all

CFLAGS = -Wall -Wextra -O2 -Iinclude

TARGET = bin/distribute_mtx

LIB_SRC = src/io/mtx_reader.c \
          src/mpi/mpi_coo_distribution.c
MAIN_SRC = src/mpi/distribute_mtx.c

LIB_OBJ = $(LIB_SRC:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)

all: $(TARGET)

bin:
	mkdir -p bin

%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@

$(TARGET): $(LIB_OBJ) $(MAIN_OBJ) | bin
	$(MPICC) $(CFLAGS) $(LIB_OBJ) $(MAIN_OBJ) -o $@

clean:
	rm -f $(TARGET) $(LIB_OBJ) $(MAIN_OBJ)

.PHONY: all clean
