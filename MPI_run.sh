#!/bin/bash
#SBATCH --job-name=2_gpu
#SBATCH --output=mpi_output_%j.out
#SBATCH --error=mpi_error_%j.err
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:2
#SBATCH --mem=96G

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
module load METIS/5.1.0-GCCcore-12.3.0

REPS="${1:-50}"
WARMUP="${2:-5}"
OUTPUT="${3:-results/mpi_spmv_detailed.csv}"
MPI_TASKS="${SLURM_NTASKS:-2}"
PARTITION_MODES="${PARTITION_MODES:- 1d-random 1d-gp 2d-block 2d-gp}"
# 1d-random 1d-gp 2d-block 2d-random 2d-gp
PARTITION_SEED="${PARTITION_SEED:-20260722}"
INPUT_MODE="${INPUT_MODE:-mpi-io}"
KERNEL="${KERNEL:-adaptive}"
CUDA_AWARE_MPI="${CUDA_AWARE_MPI:-1}"
NCCL="${NCCL:-0}"

if [[ "$NCCL" == "1" && -n "${NCCL_MODULE:-}" ]]; then
    module load "$NCCL_MODULE"
fi

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
if [[ "$NCCL" == "1" ]]; then
    EXTRA_ARGS+=(--nccl)
elif [[ "$CUDA_AWARE_MPI" == "1" ]]; then
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
        --input-mode "$INPUT_MODE" \
        "${EXTRA_ARGS[@]}"
done
