# GPU Computing - Deliverable 2: MPI Multi-GPU SpMV

Matrix Market input can be loaded entirely by rank 0 and scattered, or read in
parallel as disjoint per-rank byte chunks. Parsed COO entries are routed to their
row owner, converted to local CSR, and processed on one CUDA GPU. The benchmark
supports replicated dense vectors, cyclically distributed vectors, and a third
balanced block-distributed mode.

**Author:** Matteo Morellini (268427) - University of Trento

## Layout

- `include/mtx_reader.h`, `src/io/mtx_reader.c` - the deliverable 1 reader plus
  a byte-range reader used for parallel file input.
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
module load CUDA/12.3.2
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

Distribution-only check for every `.mtx` file in `matrices/`:

```bash
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
mpirun -np 4 ./bin/distribute_mtx
mpirun -np 4 ./bin/distribute_mtx --input-mode root
mpirun -np 4 ./bin/distribute_mtx --partition block
mpirun -np 4 ./bin/distribute_mtx --partition block \
  --matrix dummy_matrix/tiny.mtx
```

Run the MPI SpMV benchmark across every `.mtx` file in `matrices/`:

```bash
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel scalar
```

On Slurm, the run script requests two NVIDIA A30 GPUs to match deliverable 1's
GPU type. It runs every `.mtx` file in `matrices/` and writes
`results/mpi_spmv.csv`:

```bash
sbatch MPI_run.sh
```

To test whether Open MPI can select the same-node CUDA IPC path, run:

```bash
sbatch MPI_run.sh --cuda-ipc-test
```

The probe forces the `smcuda` transport so that it cannot silently fall back to
TCP, enables CUDA IPC diagnostics, and runs a generated two-rank ghost exchange.
It writes the verbose transport trace to
`results/cuda_ipc_probe_<job-id>.log` and reports `AVAILABLE`, `FAILED`, or
`INCONCLUSIVE`.

For the larger matrices from deliverable 1, either copy or symlink them into
`matrices/` before running the executable.

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
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel all --reps 100 --warmup 5
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --output results/my_run.csv
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --cuda-aware-mpi
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --x-mode block
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --input-mode root
sbatch MPI_run.sh vector 100 5
```

Matrix input is selected independently from dense-vector ownership:

- `--input-mode distributed` (default): every rank parses a disjoint file
  chunk, then entries are exchanged with their row owners.
- `--input-mode root`: rank 0 reads the complete file with the original reader,
  groups entries by row owner, and distributes them with `MPI_Scatterv`.

Dense-vector ownership modes are:

- `--x-mode replicated`: every process stores the complete dense vector.
- `--x-mode cyclic` (also `distributed` or `dist`): `owner(j) = j mod P`.
- `--x-mode block`: rank `r` owns one balanced contiguous interval of vector
  entries. Matrix rows use the matching block partition.

By default, MPI communication uses host buffers. Pass `--cuda-aware-mpi` only
when the loaded MPI implementation supports CUDA device pointers. In
distributed-x mode, this packs outgoing ghost values on the GPU and receives
incoming values directly into the compact device vector. The final result
gather also communicates device buffers directly. An MPI implementation without
CUDA-aware support may fail when this flag is enabled.

The default CSV output is:

```text
results/mpi_spmv.csv
```

Repeated runs append rows to the CSV. The header is written only when the file
does not exist yet or is empty.

It uses the same metric columns as deliverable 1:

```text
implementation,format,matrix,rows,cols,nnz,
processes,avg_time_s,std_time_s,gflops,
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

With `--input-mode distributed`, each rank opens the same `.mtx` file, reads the
header, and parses only its balanced byte range of the data section. A rank that
starts in the middle of a text line advances to the next line, while the
preceding rank finishes that line. This makes every stored Matrix Market entry
belong to exactly one reader. Pattern and symmetric inputs retain the behavior
of the serial reader, including off-diagonal expansion.

Parsed entries are sent with `MPI_Alltoallv` to the selected row owner. With
`--input-mode root`, rank 0 instead uses the original full-file reader and sends
the grouped entries with `MPI_Scatterv`. The default row assignment remains
cyclic:

```text
owner(i) = i mod P
```

where `P` is the number of MPI processes and `i` is the zero-based global row
index. Block mode instead assigns balanced contiguous row intervals. In both
cases ranks exchange three arrays:

- `row`
- `col`
- `data`

Each rank receives a `LocalCOO_Matrix` containing all nonzeros for its owned
rows plus the global matrix dimensions and global `nnz`. This exchange also
allows cyclic ownership even though physical file reads are contiguous.

## SpMV Strategy

Rank 0 generates the dense vector `x` once with `fill_dense`. It is broadcast in
replicated mode or scattered according to cyclic/block ownership in distributed
modes. Each process remaps its owned global rows to compact local CSR rows. For
cyclic rows:

```text
global row = rank + local row * P
local row  = global row / P
```

For block rows, the local row is `global row - first owned row`. The CUDA kernel
computes only those local rows. Rank 0 gathers local `y` chunks with
`MPI_Gatherv`; cyclic results are interleaved back into global order, while block
results are already contiguous. With `--cuda-aware-mpi`, the gather uses device
send and receive buffers before rank 0 copies the gathered values to the host
for reconstruction and validation.
