# GPU Computing - Deliverable 2: MPI Multi-GPU SpMV

Matrix Market input can be loaded entirely by rank 0 and scattered, read in
parallel as independent stdio byte chunks, or read collectively with MPI-IO.
Parsed COO entries are routed to their row owner, converted to local CSR, and
processed on one CUDA GPU. In addition to the legacy replicated and cyclic
layouts, the benchmark implements the 1D-Block, 1D-Random, 1D-GP/HP,
2D-Block, 2D-Random, and Algorithm-2 2D-GP/HP layouts from *Scalable Matrix
Computations on Large Scale-Free Graphs Using 2D Graph Partitioning*.

**Author:** Matteo Morellini (268427) - University of Trento

## Layout

- `include/mtx_reader.h`, `src/io/mtx_reader.c` - the deliverable 1 reader plus
  a byte-range reader used for parallel file input.
- `include/mpi_mtx_reader.h`, `src/mpi/mpi_mtx_reader.c` - collective
  line-aligned Matrix Market input using `MPI_File_read_at_all`.
- `include/coo_to_csr.h`, `src/io/coo_to_csr.c` - COO to CSR conversion copied
  from deliverable 1.
- `include/generate_dense.h`, `src/io/generate_dense.c` - dense input vector
  generation copied from deliverable 1.
- `include/mpi_coo_distribution.h`, `src/mpi/mpi_coo_distribution.c` - MPI
  helper that broadcasts matrix metadata and scatters COO entries.
- `src/mpi/distribute_mtx.c` - distribution-only smoke-test executable.
- `src/tools/analyze_partition_nnz.c` - distribution-only NNZ ownership
  analysis for six selected layouts at simulated process counts 2 and 4.
- `src/tools/analyze_partition_communication.c` - per-rank, per-SpMV
  communication-volume analysis for the same layouts and process counts.
- `src/mpi/spmv_mpi_cuda.cu` - MPI multi-GPU SpMV executable.
- `src/mpi/spmv_mpi_cuda_weak.cu` - separate direct-CSR weak-scaling
  benchmark; it reuses the strong driver's vector ownership and ghost exchange
  implementation without entering its Matrix Market workflow.
- `src/tools/generate_rmat.c` - chunked Matrix Market wrapper around the
  official Graph500 R-MAT/Kronecker generator.
- `scripts/` - Slurm launchers, dataset utilities, and shell test helpers.
- `include/spmv_kernel_runner.cuh`, `src/kernels/spmv_kernel_runners.cu` -
  common runner interface and selectable CUDA/cuSPARSE kernel implementations.
- `matrices/tiny.mtx` - small input file for smoke testing.

## Build

From this directory:

```bash
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
module load METIS/5.1.0-GCCcore-12.3.0
make
```

NCCL is optional. Load the NCCL module supplied by the cluster and enable it
at build time (set `NCCL_ROOT` only when the module does not export
`EBROOTNCCL`):

```bash
module load NCCL
make clean
make NCCL=1
# or: make clean && make NCCL=1 NCCL_ROOT=/path/to/nccl
```

Run `make clean` when switching an existing build between `NCCL=0` and
`NCCL=1`, because both configurations produce binaries with the same names.

The build uses `mpicc` for C helpers and `nvcc -ccbin mpicxx` for the CUDA MPI
driver. The CUDA build targets `sm_80` because the partial-overlap kernel uses
CUDA async-copy pipeline APIs, and it links cuSPARSE for the `cusparse` backend.
It produces:

```bash
bin/distribute_mtx
bin/analyze_partition_nnz
bin/analyze_partition_communication
bin/mpi_spmv_cuda
bin/mpi_spmv_cuda_weak
```

If your environment exposes a different MPI module, load the one that provides
`mpicc`.

## Graph500 R-MAT Matrix Generation

The R-MAT utility uses the official Graph500 generator with its fixed
initiator probabilities `a=0.57`, `b=0.19`, `c=0.19`, and `d=0.05`. Clone the
reference implementation once next to this project's `Makefile`, then build
the standalone utility:

```bash
git clone https://github.com/graph500/graph500 graph500
make rmat
```

Generation is chunked and therefore does not hold the complete edge-tuple list
in memory. The default seeds are Graph500's `2` and `3`, duplicate/parallel
edges and self-loops are retained, and an existing output is not overwritten
unless `--force` is passed.

Generate an undirected matrix whose Matrix Market reader-expanded COO contains
about 151 million entries:

```bash
./bin/generate_rmat \
  --scale 23 --edge-factor 9 \
  --output matrices/rmat-s23-ef9.mtx
```

This generates 75,497,472 edge tuples. With the default `symmetric` output,
each off-diagonal tuple is written once in the lower triangle and expanded in
memory to both directions by this project's readers. Thus the loaded NNZ is
`2 * stored_entries - stored_self_loops`, approximately 151 million. Repeated
coordinates are deliberately not consolidated.

To generate approximately 151 million raw tuples instead, without symmetric
expansion, use:

```bash
./bin/generate_rmat \
  --scale 23 --edge-factor 18 --format general \
  --output matrices/rmat-s23-ef18-general.mtx
```

Important options are:

```text
--edges N             request an exact tuple count instead of an edge factor
--seed1 N --seed2 N   select the reproducible Graph500 random stream
--format symmetric    undirected Matrix Market matrix (default)
--format general      one directed Matrix Market entry per raw tuple
--drop-self-loops     omit diagonal tuples and update the declared NNZ
--chunk-edges N       change the generation buffer size
--force               replace an existing output file
```

The program reports generated tuples, self-loops, stored entries, and the NNZ
seen after this project's symmetric Matrix Market expansion. Run its tests with:

```bash
make test-rmat
```

## Run

Distribution-only check for every `.mtx` file in `matrices/`:

```bash
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
mpirun -np 4 ./bin/distribute_mtx
mpirun -np 4 ./bin/distribute_mtx --input-mode root
mpirun -np 4 ./bin/distribute_mtx --input-mode mpi-io
mpirun -np 4 ./bin/distribute_mtx --partition block
mpirun -np 4 ./bin/distribute_mtx --partition block \
  --matrix dummy_matrix/tiny.mtx
```

Analyze NNZ ownership without running SpMV or requiring GPUs:

```bash
./bin/analyze_partition_nnz \
  --output results/partition_nnz.csv \
  --long-row-fraction 0.35 \
  dummy_matrix/tiny.mtx
```

The analyzer loads each matrix once and simulates both `P=2` and `P=4`. For
each process count it uses the same vertex and nonzero ownership functions as
the SpMV driver for `1d-cyclic`, `1d-block`, `1d-gp`, `1d-lra`, `2d-block`,
and `2d-gp`. It prints the per-rank NNZ counts and their minimum, average,
maximum, `max/avg`, and normalized spread. The optional CSV contains one row
per matrix, process count, and layout; its four rank columns are left empty
where `P=2`, and it records the fraction used for `1d-lra`. Existing output
files are preserved unless `--force` is supplied.

Submit the analysis for every matrix in `matrices/` as a CPU-only Slurm job:

```bash
sbatch scripts/partition_nnz_run.sh
```

Select one matrix or output path with exported variables:

```bash
sbatch --export=ALL,MATRIX=dummy_matrix/tiny.mtx,\
OUTPUT=results/tiny_partition_nnz.csv,LONG_ROW_FRACTION=0.35 \
scripts/partition_nnz_run.sh
```

Analyze per-rank communication volume without executing SpMV:

```bash
./bin/analyze_partition_communication \
  --output results/partition_communication.csv \
  --long-row-fraction 0.35 \
  dummy_matrix/tiny.mtx
```

For each rank, the CSV reports the number of values sent and received during
the expand phase (distinct remote `x` entries) and the 2D fold phase (remote
partial `y` rows), followed by combined values and bytes. A rank's total
volume is `total_send + total_recv`; summing that column across ranks therefore
counts every network value once at its sender and once at its receiver. The
analysis excludes self-copies, one-time plan construction and matrix input,
and the final result gather used for validation.

Submit the communication analysis for every matrix with:

```bash
sbatch scripts/communication_volume_run.sh
```

Run the MPI SpMV benchmark across every `.mtx` file in `matrices/`:

```bash
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel scalar
```

On Slurm, the run script requests four NVIDIA A30 GPUs to match its four MPI
tasks and uses the same GPU type as deliverable 1. It runs every `.mtx` file in
`matrices/` and writes
`results/mpi_spmv_detailed.csv`:

```bash
sbatch scripts/MPI_run.sh
```

`scripts/MPI_run.sh` runs the whitespace-separated layouts in `PARTITION_MODES`
(`2d-gp` by default). Select another subset or a single input with exported
variables, for example:

```bash
sbatch --export=ALL,PARTITION_MODES="1d-block 2d-block 2d-gp",\
PROCESS_GRID=2x2,MATRIX=dummy_matrix/tiny.mtx scripts/MPI_run.sh 50 5
```

The script builds `bin/mpi_spmv_cuda` with `NCCL=0` after loading its required
modules, so it can be submitted directly from a delivery whose `bin/`
directory is empty.

Other script controls are `PARTITION_SEED`, `PARTITION_FILE`,
`LONG_ROW_FRACTION`, `INPUT_MODE`, `KERNEL`, and `CUDA_AWARE_MPI=0|1`.

For the NCCL communication backend, build with `make NCCL=1` and submit the
separate NCCL launcher. It configures the NCCL library path for binaries built
without an embedded runtime path:

```bash
sbatch scripts/MPI_run_nccl.sh
```

When the cluster does not export `EBROOTNCCL`, pass its root at submission:
`NCCL_ROOT=/path/to/nccl sbatch scripts/MPI_run_nccl.sh`. Optionally set
`NCCL_MODULE` to load a site-specific NCCL module. Rebuild with `make clean &&
make NCCL=0` before returning to `scripts/MPI_run.sh`.

For an LRA-only batch run:

```bash
sbatch --export=ALL,PARTITION_MODES=1d-lra,LONG_ROW_FRACTION=0.35 scripts/MPI_run.sh
```

To test whether Open MPI can select the same-node CUDA IPC path, run:

```bash
sbatch scripts/MPI_run.sh --cuda-ipc-test
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
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --nccl
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --x-mode block
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --x-mode 1d-lra \
  --long-row-fraction 0.35 --matrix dummy_matrix/tiny.mtx
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --x-mode 2d-gp \
  --process-grid 2x2 --matrix dummy_matrix/tiny.mtx
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --input-mode root
mpirun -np 4 ./bin/mpi_spmv_cuda --kernel vector --input-mode mpi-io
sbatch scripts/MPI_run.sh vector 100 5
```

Matrix input is selected independently from dense-vector ownership:

- `--input-mode distributed` (default): every rank parses a disjoint file
  chunk using independent stdio operations, then entries are exchanged with
  their row owners.
- `--input-mode mpi-io`: rank 0 parses and broadcasts the header, then every
  rank collectively reads a line-aligned data chunk with
  `MPI_File_read_at_all`.
- `--input-mode root`: rank 0 reads the complete file with the original reader,
  groups entries by row owner, and distributes them with `MPI_Scatterv`.

Matrix/vector partition modes are:

- `--x-mode replicated`: every process stores the complete dense vector.
- `--x-mode cyclic` (also `distributed` or `dist`): `owner(j) = j mod P`.
- `--x-mode 1d-block` (also `block`): balanced contiguous `n/P` row blocks.
- `--x-mode 1d-random`: vertices are independently and uniformly assigned
  with a deterministic hash of the vertex and `--partition-seed`.
- `--x-mode 1d-gp`: METIS partitions the graph while weighting vertices by
  row nonzeros.
- `--x-mode 1d-hp`: a built-in balanced row-net hypergraph heuristic is used.
- `--x-mode 1d-lra` (also `lra` or `long-row-aware`): selects a
  `floor(--long-row-fraction * rows)` prefix or suffix based on the longest
  row, then independently NNZ-balances that long-row region and the
  complementary short-row region into consecutive sub-blocks. Every rank owns
  one sub-block from each region. For tiny nonempty matrices, the long-row
  block is clamped to at least one row. The default fraction is `0.35`.
- `--x-mode 2d-block`, `2d-random`, `2d-gp`, or `2d-hp`: the corresponding
  1D vertex partition is converted to a Cartesian nonzero partition using
  Algorithm 2 from the paper.

The default 2D process grid is the closest factorization of `P` to square.
Override it with `--process-grid ROWSxCOLS`; the product must equal `P`. Use
`--partition-seed N` to reproduce random layouts. Explicit GP/HP/LRA modes may
instead read an externally generated partition with `--partition-file PATH`.
The file must contain exactly one zero-based part per vertex, optionally
written as `vertex part`; parts must be in `[0,P)`. This is the route for using
a Zoltan, PaToH, or another production partition in place of the built-in
implementation. GP/HP/LRA and all 2D modes require a square matrix.

The LRA definition follows Section 3.3 of [Gao, Ji, and Wang, *Optimization of
Large-Scale Sparse Matrix-Vector Multiplication on Multi-GPU Systems*,
ACM TACO 21(4), 2024](https://doi.org/10.1145/3676847). The paper tunes the
long-row fraction by matrix and platform; `0.35` is one of its reported LRA
choices for irregular matrices averaging at least eight nonzeros per row.
Override it in `scripts/MPI_run.sh` with, for example,
`LONG_ROW_FRACTION=0.25`.

`--matrix PATH` runs only one Matrix Market input. Without it, the benchmark
continues to scan every `.mtx` file in `matrices/`.

By default, MPI communication uses host buffers. Pass `--cuda-aware-mpi` only
when the loaded MPI implementation supports CUDA device pointers. In
distributed-x mode, this packs outgoing ghost values on the GPU and receives
incoming values directly into the compact device vector. The final result
gather also communicates device buffers directly. An MPI implementation without
CUDA-aware support may fail when this flag is enabled.

Pass `--nccl` to select the hybrid MPI+NCCL path. MPI still launches and
coordinates ranks, reads Matrix Market data with MPI-IO, redistributes the
initial matrix, and exchanges the one-time ghost/fold metadata. NCCL handles
the repeated GPU-to-GPU ghost exchange, the 2D device result fold, and the
final variable-size device result gather. `--nccl` and `--cuda-aware-mpi` are
mutually exclusive. A binary built without `make NCCL=1` rejects `--nccl`
with a diagnostic instead of silently falling back to MPI.

The default detailed CSV output is:

```text
results/mpi_spmv_detailed.csv
```

Repeated runs append rows to the CSV when its header matches the current
schema. Use a new `--output` path or remove an older file before switching from
the previous schema.

It retains the original metrics and adds a detailed input breakdown:

```text
implementation,format,matrix,rows,cols,nnz,
processes,avg_time_s,std_time_s,gflops,
comm_time_s,compute_time_s,
file_parse_s,input_total_s,input_read_parse_s,
input_file_io_s,input_parse_s,matrix_validation_s,
matrix_redistribution_s,matrix_pack_s,matrix_exchange_s,
matrix_source_nnz,matrix_remote_nnz,matrix_remote_fraction,
format_conv_s,h2d_transfer_s,
valid,max_abs_error
```

`file_parse_s` is retained as the legacy input value. The detailed columns
separate raw reading, MPI file operations, in-memory parsing, entry-count
validation, local owner packing, and MPI row exchange. Traffic counts refer to
expanded COO entries, so symmetric off-diagonal entries count twice.
`matrix_remote_fraction` is the fraction whose row owner differs from the rank
that read the entry. `input_file_io_s` and `input_parse_s` are MPI-IO-specific
subcomponents and remain zero for the stdio and root readers. Each compound
phase uses the slowest rank for that phase, and its displayed subcomponents are
taken from that same rank, so `file I/O + buffer parse` and
`pack + MPI exchange` remain meaningful decompositions.

For MPI, each repetition measures local GPU kernel time on every rank and stores
the maximum rank time. `avg_time_s`, `std_time_s`, and `gflops` are computed
from those max-per-repetition times with global `nnz`, because distributed SpMV
finishes when the slowest rank finishes. The `format_conv_s` column contains the
max local COO-to-CSR time plus kernel-specific preprocessing. Row gather time is
printed separately because the CSV has no dedicated result-gather column.

## Distribution Strategy

With `--input-mode distributed`, each rank opens the same `.mtx` file through
stdio, reads the header, and parses only its balanced byte range of the data
section. A rank that starts in the middle of a text line advances to the next
line, while the preceding rank finishes that line.

With `--input-mode mpi-io`, all ranks collectively open the file. Rank 0 locates
the end of the banner, comments, and dimensions line and broadcasts the parsed
metadata and data offset. Each nominal byte boundary is advanced to the next
line start. Adjacent ranks exchange those aligned offsets, then every rank reads
its exact complete-line interval with `MPI_File_read_at_all`. Large intervals
are handled in coordinated rounds so the MPI `int` count limit is not exceeded.

Both parallel modes assign every stored Matrix Market entry to exactly one
reader. Pattern and symmetric inputs retain the behavior of the serial reader,
including off-diagonal expansion.

Parsed entries are sent with `MPI_Alltoallv` to the selected nonzero owner. With
`--input-mode root`, rank 0 instead uses the original full-file reader and sends
the grouped entries with `MPI_Scatterv`. The default row assignment remains
cyclic:

```text
owner(i) = i mod P
```

where `P` is the number of MPI processes and `i` is the zero-based global row
index. For every paper mode a vertex partition `rpart` assigns both vector
entries and graph vertices to one of `P` parts. A 1D mode sends `a(i,j)` to
`rpart(i)`. For a `Pr x Pc` 2D mode, Algorithm 2 sends it to:

```text
process_row = rpart(i) mod Pr
process_col = floor(rpart(j) / Pr)
rank        = process_row + process_col * Pr
```

The mapping deliberately uses the same `rpart` for rows and columns and does
not explicitly permute the matrix. In every mode ranks exchange three arrays:

- `row`
- `col`
- `data`

Each rank receives a `LocalCOO_Matrix` containing its nonzeros plus the global
matrix dimensions and global `nnz`. In a 2D mode a rank can contain partial
rows; this is why the SpMV path includes the fold described below.

## MPI-IO Reader Test

The MPI-IO test compares canonicalized local COO entries against the serial
reader for general, pattern, and symmetric inputs, using both cyclic and block
row ownership at process counts 1 through 5:

```bash
module load OpenMpi/4.1.5-CUDA-12.3.2
make test-mpi-io
```

## SpMV Strategy

Rank 0 generates the dense vector `x` once with `fill_dense`. It is broadcast in
replicated mode or scattered according to `rpart` in distributed modes. Each
process remaps its global rows to compact local CSR rows. For
cyclic rows:

```text
global row = rank + local row * P
local row  = global row / P
```

Block layouts use the analogous offset within a contiguous interval; random and
GP/HP layouts retain an explicit global-to-local lookup. Before every SpMV, an
expand plan sends only remotely referenced `x` entries.

For 1D layouts the CUDA kernel directly produces the owned `y` entries. For 2D
layouts it produces one partial value for every locally represented global row.
A precomputed fold plan exchanges those partial values with the `rpart(row)`
owner and sums them. Thus measured 2D communication includes both the paper's
expand and fold phases. Rank 0 finally gathers the owned `y` chunks and restores
global order for validation. With `--cuda-aware-mpi`, expand, fold, and final
gather use device buffers; otherwise they use host staging.

Run the ownership/nonzero-routing regression suite with:

```bash
module load OpenMpi/4.1.5-CUDA-12.3.2
module load METIS/5.1.0-GCCcore-12.3.0
make test-partitions
```

## Weak Scaling

Weak scaling is a separate executable and batch script; `scripts/MPI_run.sh` and the
Matrix Market strong-scaling path are unchanged. Submit the complete 1, 2, 3,
and 4 GPU experiment with:

```bash
sbatch scripts/weak_scaling_run.sh
```

The script keeps `ROWS_PER_GPU`, `NNZ_PER_ROW`, the kernel, vector ownership,
seed, warm-up count, and measured repetition count fixed. It runs process
counts in ascending order and stores the one-GPU time in a job-specific
baseline file, allowing later runs to report `T(1) / T(P)`. Parameters are
controlled with exported variables, for example:

```bash
sbatch --export=ALL,ROWS_PER_GPU=524288,NNZ_PER_ROW=48,REPS=100 \
  scripts/weak_scaling_run.sh
```

The defaults are 262144 rows per GPU, 32 nonzeros per row, 5 warm-ups, 50
measured repetitions, seed 20260721, the adaptive kernel, block-distributed
vectors, and CUDA-aware MPI. Set `CUDA_AWARE_MPI=0` for host staging or
`NCCL=1` for NCCL device exchange. Extra
command-line options passed to the script are appended to the executable
arguments and therefore override these defaults.

For a small validation run, use a global size no larger than
`--validation-max-rows` (65536 by default):

```bash
mpirun -np 4 ./bin/mpi_spmv_cuda_weak \
  --rows-per-rank 4096 --nnz-per-row 16 --reps 10 --warmup 2 \
  --validation-max-rows 65536 \
  --output results/weak_validation.csv
```

Every rank generates its local CSR rows directly. For each global row, the
generator hashes `(seed, global row)` into a start column and a modular stride.
The stride is adjusted to be coprime with the global column count, so the first
`NNZ_PER_ROW` columns are distinct without rejection sampling. Values and the
dense vector are also hash-generated, making a configuration reproducible
without assembling or scattering a global matrix or vector. Every rank has
exactly `ROWS_PER_GPU * NNZ_PER_ROW` nonzeros. Thus

```text
global rows = P * ROWS_PER_GPU
global NNZ  = P * ROWS_PER_GPU * NNZ_PER_ROW
density     = NNZ_PER_ROW / global rows
```

Global work grows linearly and density falls as `1/P`; keeping density fixed
would instead make square-matrix NNZ grow quadratically. Uniform random global
columns deliberately exercise the existing compact ghost-vector exchange.

Each measured iteration records the slowest rank's complete communication plus
kernel time. Communication time includes packing/exchange and any host-to-device
ghost copy; computation time is measured with CUDA events. GFLOP/s uses
`2 * global_NNZ / max_rank_time`. The CSV also records:

- mean and standard deviation of the per-iteration maximum-rank time;
- NNZ min/average/max across ranks;
- unique ghost send/receive values, per-rank bidirectional bytes, and aggregate
  one-way payload per iteration;
- estimated persistent host allocation, measured CUDA allocation, and their
  total min/average/max per rank;
- CPU-reference validity and maximum absolute error for small configurations;
- maximum-rank generation and setup times, which are not included in iteration
  timing.

Successful weak scaling means `max_rank_time_s` stays nearly flat and
`weak_scaling_efficiency` stays near 1 as `P` increases. Stable computation time
with rising communication time identifies vector exchange as the scaling
limit. Inspecting communication bytes distinguishes an algorithmic growth in
the ghost set from latency or bandwidth effects at roughly fixed volume.
GFLOP/s should grow close to linearly because global NNZ grows linearly, while
memory min/average/max should remain nearly constant and balanced per rank.
