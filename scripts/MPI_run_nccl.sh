#!/bin/bash
#SBATCH --job-name=mpi_nccl
#SBATCH --output=mpi_nccl_output_%j.out
#SBATCH --error=mpi_nccl_error_%j.err
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:4
#SBATCH --mem=96G

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
module load METIS/5.1.0-GCCcore-12.3.0

if [[ -n "${NCCL_MODULE:-}" ]]; then
    module load "$NCCL_MODULE"
fi

NCCL_ROOT="${NCCL_ROOT:-${EBROOTNCCL:-/opt/shares/NVHPC/nvhpc_24.7_cuda_12.5/Linux_x86_64/24.7/comm_libs/nccl}}"
if [[ ! -r "$NCCL_ROOT/lib/libnccl.so" ]]; then
    echo "NCCL library not found under NCCL_ROOT=$NCCL_ROOT" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Build an NCCL-enabled executable so this launcher works from an empty bin/.
# -B is required because bin/mpi_spmv_cuda has the same name in both build
# configurations, so a stale NCCL=0 binary would otherwise look up to date.
make -B NCCL=1 NCCL_ROOT="$NCCL_ROOT" bin/mpi_spmv_cuda

REPS="${1:-50}"
WARMUP="${2:-5}"
OUTPUT="${3:-results/nccl.csv}"
PARTITION_MODES="${PARTITION_MODES:-1d-gp 2d-gp}"
PARTITION_SEED="${PARTITION_SEED:-20260722}"
LONG_ROW_FRACTION="${LONG_ROW_FRACTION:-0.25}"
INPUT_MODE="${INPUT_MODE:-mpi-io}"
KERNEL="${KERNEL:-adaptive}"

EXTRA_ARGS=(--nccl)
if [[ -n "${PROCESS_GRID:-}" ]]; then
    EXTRA_ARGS+=(--process-grid "$PROCESS_GRID")
fi
if [[ -n "${PARTITION_FILE:-}" ]]; then
    EXTRA_ARGS+=(--partition-file "$PARTITION_FILE")
fi
if [[ -n "${MATRIX:-}" ]]; then
    EXTRA_ARGS+=(--matrix "$MATRIX")
fi

for MPI_TASKS in 2 3 4; do
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
done
