# GPU Computing - Deliverable 2: MPI Matrix Market Distribution

Rank 0 reads the complete Matrix Market file in COO format and distributes
contiguous chunks of matrix entries to all MPI processes.

**Author:** Matteo Morellini (268427) - University of Trento

## Layout

- `include/mtx_reader.h`, `src/io/mtx_reader.c` - same COO Matrix Market reader
  interface used in deliverable 1.
- `include/mpi_coo_distribution.h`, `src/mpi/mpi_coo_distribution.c` - MPI
  helper that broadcasts matrix metadata and scatters COO entries.
- `src/mpi/distribute_mtx.c` - executable driver for the deliverable task.
- `matrices/tiny.mtx` - small input file for smoke testing.

## Build

From this directory:

```bash
module load OpenMpi/4.1.5-CUDA-12.3.2
make
```

The build uses `mpicc` and produces:

```bash
bin/distribute_mtx
```

If your environment exposes a different MPI module, load the one that provides
`mpicc`.

## Run

Smoke test with the included matrix:

```bash
module load OpenMpi/4.1.5-CUDA-12.3.2
mpirun -np 4 ./bin/distribute_mtx matrices/tiny.mtx
```

On Slurm:

```bash
sbatch MPI_run.sh matrices/tiny.mtx
```

For the larger matrices from deliverable 1, either copy or symlink them into
`matrices/`, then pass the desired `.mtx` path:

```bash
mpirun -np 4 ./bin/distribute_mtx matrices/ASIC_680ks.mtx
```

## Distribution Strategy

Rank 0 reads the full `.mtx` file into a `COO_Matrix`. The metadata
`rows`, `cols`, and `nnz` is broadcast to every process. Each rank computes the
same balanced partition:

```text
local_nnz(rank) = nnz / size + (rank < nnz % size)
```

Then the root process scatters three arrays with `MPI_Scatterv`:

- `row`
- `col`
- `data`

Each rank receives a `LocalCOO_Matrix` containing its local entry slice, the
global matrix dimensions, the global `nnz`, and the global offset of its first
entry.
