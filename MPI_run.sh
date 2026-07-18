#!/bin/bash
#SBATCH --job-name=GPU_deliverable-2_mpi
#SBATCH --output=mpi_output_%j.out
#SBATCH --error=mpi_error_%j.err
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:1

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

KERNEL="${1:-scalar}"
REPS="${2:-100}"
WARMUP="${3:-5}"
OUTPUT="${4:-results/mpi_spmv.csv}"
MPI_TASKS="${SLURM_NTASKS:-2}"

for KERNEL in adaptive; do
    mpirun -np "$MPI_TASKS" ./bin/mpi_spmv_cuda \
        --kernel "$KERNEL" --reps "$REPS" --warmup "$WARMUP" \
        --output "$OUTPUT"
done