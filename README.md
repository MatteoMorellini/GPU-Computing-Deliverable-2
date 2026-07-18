# GPU Computing - Deliverable 2: MPI Multi-GPU SpMV

Rank 0 reads the complete Matrix Market file in COO format and distributes rows
with 1D modulo cyclic ownership. Each rank receives COO entries for rows
`owner(i) = i mod P`, converts those entries to CSR, computes its part of the
sparse matrix-vector product on one CUDA GPU, and rank 0 merges the owned row
results.

**Author:** Matteo Morellini (268427) - University of Trento

## Layout

- `include/mtx_reader.h`, `src/io/mtx_reader.c` - same COO Matrix Market reader
  interface used in deliverable 1.
- `include/coo_to_csr.h`, `src/io/coo_to_csr.c` - COO to CSR conversion copied
  from deliverable 1.
- `include/generate_dense.h`, `src/io/generate_dense.c` - dense input vector
  generation copied from deliverable 1.
- `include/mpi_coo_distribution.h`, `src/mpi/mpi_coo_distribution.c` - MPI
  helper that broadcasts matrix metadata and scatters COO entries.
- `src/mpi/distribute_mtx.c` - distribution-only smoke-test executable.
- `src/mpi/spmv_mpi_cuda.cu` - MPI multi-GPU SpMV executable.
- `include/spmv_kernel_runner.cuh`, `src/kernels/spmv_kernel_runners.cu` -
  common runner interface and selectable CUDA/cuSPARSE kernel implementations.
- `matrices/tiny.mtx` - small input file for smoke testing.

## Build

From this directory:

```bash
module load OpenMpi/4.1.5-CUDA-12.3.2
make
```

The build uses `mpicc` for C helpers and `nvcc -ccbin mpicxx` for the CUDA MPI
driver. The CUDA build targets `sm_80` because the partial-overlap kernel uses
CUDA async-copy pipeline APIs, and it links cuSPARSE for the `cusparse` backend.
It produces:

```bash
bin/distribute_mtx
bin/mpi_spmv_cuda
```

If your environment exposes a different MPI module, load the one that provides
`mpicc`.

## Run

Smoke test with the included matrix:

```bash
module load OpenMpi/4.1.5-CUDA-12.3.2
mpirun -np 4 ./bin/mpi_spmv_cuda matrices/tiny.mtx --kernel scalar
```

On Slurm:

```bash
sbatch MPI_run.sh matrices/tiny.mtx scalar
```

For the larger matrices from deliverable 1, either copy or symlink them into
`matrices/`, then pass the desired `.mtx` path:

```bash
mpirun -np 4 ./bin/mpi_spmv_cuda matrices/ASIC_680ks.mtx --kernel adaptive
```

Available kernels:

```text
scalar
vector
adaptive
adaptive-paper
partial
cusparse
all
```

Use `--kernel all` to run all implementations sequentially on the same local CSR
matrix and print one timing/checksum block per implementation.

Benchmark controls:

```bash
mpirun -np 4 ./bin/mpi_spmv_cuda matrices/tiny.mtx --kernel all --reps 100 --warmup 5
mpirun -np 4 ./bin/mpi_spmv_cuda matrices/tiny.mtx --kernel vector --output results/my_run.csv
sbatch MPI_run.sh matrices/tiny.mtx vector 100 5
```

The default CSV output is:

```text
results/mpi_spmv.csv
```

It uses the same metric columns as deliverable 1:

```text
implementation,format,matrix,rows,cols,nnz,
avg_time_s,std_time_s,gflops,
file_parse_s,format_conv_s,h2d_transfer_s,
valid,max_abs_error
```

For MPI, each repetition measures local GPU kernel time on every rank and stores
the maximum rank time. `avg_time_s`, `std_time_s`, and `gflops` are computed
from those max-per-repetition times with global `nnz`, because distributed SpMV
finishes when the slowest rank finishes. The `format_conv_s` column contains the
max local COO-to-CSR time plus kernel-specific preprocessing. Row gather time is
printed separately because deliverable 1's CSV schema has no communication
column for it.

## Distribution Strategy

Rank 0 reads the full `.mtx` file into a `COO_Matrix`. The metadata `rows`,
`cols`, and `nnz` is broadcast to every process. Rows are assigned cyclically:

```text
owner(i) = i mod P
```

where `P` is the number of MPI processes and `i` is the zero-based global row
index. Rank 0 groups COO triples by `owner(row)` and scatters three arrays with
`MPI_Scatterv`:

- `row`
- `col`
- `data`

Each rank receives a `LocalCOO_Matrix` containing all nonzeros for its owned
cyclic rows plus the global matrix dimensions and global `nnz`.

## SpMV Strategy

Rank 0 generates the dense vector `x` once with `fill_dense` and broadcasts it to
every rank. Each process remaps its owned global rows to compact local CSR rows:

```text
global row = rank + local row * P
local row  = global row / P
```

The CUDA kernel computes only those local rows. Rank 0 gathers local `y` chunks
with `MPI_Gatherv` and reconstructs the global vector by placing each gathered
value back at `rank + local_row * P`.
