#!/bin/bash
#SBATCH --job-name=GPU_deliverable-2_weak
#SBATCH --output=weak_scaling_%j.out
#SBATCH --error=weak_scaling_%j.err
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:4

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

ROWS_PER_GPU="${ROWS_PER_GPU:-262144}"
NNZ_PER_ROW="${NNZ_PER_ROW:-32}"
REPS="${REPS:-50}"
WARMUP="${WARMUP:-5}"
SEED="${SEED:-20260721}"
KERNEL="${KERNEL:-adaptive}"
X_MODE="${X_MODE:-block}"
OUTPUT="${OUTPUT:-results/mpi_spmv_weak_scaling.csv}"
VALIDATION_MAX_ROWS="${VALIDATION_MAX_ROWS:-65536}"
CUDA_AWARE_MPI="${CUDA_AWARE_MPI:-1}"
NCCL="${NCCL:-0}"

if [[ "$NCCL" == "1" && -n "${NCCL_MODULE:-}" ]]; then
    module load "$NCCL_MODULE"
fi

mkdir -p results
RUN_ID="${SLURM_JOB_ID:-$$}"
BASELINE_FILE="${BASELINE_FILE:-results/weak_scaling_baseline_${RUN_ID}.txt}"

CUDA_AWARE_ARGS=()
if [[ "$NCCL" == "1" ]]; then
    CUDA_AWARE_ARGS+=(--nccl)
elif [[ "$CUDA_AWARE_MPI" == "1" ]]; then
    CUDA_AWARE_ARGS+=(--cuda-aware-mpi)
fi

for GPUS in 1 2 3 4; do
    mpirun -np "$GPUS" \
        --mca mpi_common_cuda_register_memory 0 \
        ./bin/mpi_spmv_cuda_weak \
        --kernel "$KERNEL" \
        --rows-per-rank "$ROWS_PER_GPU" \
        --nnz-per-row "$NNZ_PER_ROW" \
        --reps "$REPS" \
        --warmup "$WARMUP" \
        --seed "$SEED" \
        --x-mode "$X_MODE" \
        --validation-max-rows "$VALIDATION_MAX_ROWS" \
        --output "$OUTPUT" \
        --baseline-file "$BASELINE_FILE" \
        "${CUDA_AWARE_ARGS[@]}" \
        "$@"
done
