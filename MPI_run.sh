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

REPS="${1:-100}"
WARMUP="${2:-5}"
OUTPUT="${3:-results/mpi_spmv.csv}"
MPI_TASKS="${SLURM_NTASKS:-4}"

# input-mode: root or distributed
# x-mode: cyclic or block or replicated
for X_MODE in cyclic block replicated; do
    for KERNEL in cusparse; do
        mpirun -np "$MPI_TASKS" \
            --mca mpi_common_cuda_register_memory 0 \
            ./bin/mpi_spmv_cuda \
            --kernel "$KERNEL" --reps "$REPS" --warmup "$WARMUP" \
            --output "$OUTPUT" \
            --x-mode "$X_MODE" \
            --cuda-aware-mpi \
            --input-mode root
    done
done
