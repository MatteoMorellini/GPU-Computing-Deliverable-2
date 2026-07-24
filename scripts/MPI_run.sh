#!/bin/bash
#SBATCH --job-name=lra
#SBATCH --output=mpi_output_%j.out
#SBATCH --error=mpi_error_%j.err
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:2
#SBATCH --mem=96G 

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
module load METIS/5.1.0-GCCcore-12.3.0

# Build a non-NCCL executable so this launcher works from an empty bin/.
make -B NCCL=0 bin/mpi_spmv_cuda

REPS="${1:-50}"
WARMUP="${2:-5}"
OUTPUT="${3:-results/lra.csv}"
MPI_TASKS="${SLURM_NTASKS:?SLURM_NTASKS is unset; submit MPI_run.sh with sbatch}"
PARTITION_MODES="${PARTITION_MODES:-1d-lra}"
# 1d-cyclic 1d-block 1d-gp 1d-lra 2d-block 2d-gp
PARTITION_SEED="${PARTITION_SEED:-20260722}"
LONG_ROW_FRACTION="${LONG_ROW_FRACTION:-0.25}"
INPUT_MODE="${INPUT_MODE:-mpi-io}"
KERNEL="${KERNEL:-adaptive}"
CUDA_AWARE_MPI="${CUDA_AWARE_MPI:-1}"

EXTRA_ARGS=()
if [[ -n "${PROCESS_GRID:-}" ]]; then
    EXTRA_ARGS+=(--process-grid "$PROCESS_GRID")
fi
if [[ -n "${PARTITION_FILE:-}" ]]; then
    EXTRA_ARGS+=(--partition-file "$PARTITION_FILE")
fi
if [[ -n "${MATRIX:-}" ]]; then
    EXTRA_ARGS+=(--matrix "$MATRIX")
fi
if [[ "$CUDA_AWARE_MPI" == "1" ]]; then
    EXTRA_ARGS+=(--cuda-aware-mpi)
fi

# Override PARTITION_MODES with a whitespace-separated subset when desired.

for X_MODE in $PARTITION_MODES; do
    mpirun -np "$MPI_TASKS" \
        --mca mpi_common_cuda_register_memory 0 \
        ./bin/mpi_spmv_cuda \
        --kernel "$KERNEL" --reps "$REPS" --warmup "$WARMUP" \
        --output "$OUTPUT" \
        --x-mode "$X_MODE" \
        --partition-seed "$PARTITION_SEED" \
        --long-row-fraction "$LONG_ROW_FRACTION" \
        --input-mode "$INPUT_MODE" \
        "${EXTRA_ARGS[@]}"
done
