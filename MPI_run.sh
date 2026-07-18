#!/bin/bash
#SBATCH --job-name=GPU_deliverable-2_mpi
#SBATCH --output=mpi_output_%j.out
#SBATCH --error=mpi_error_%j.err
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:-matrices/tiny.mtx}"

srun ./bin/distribute_mtx "$MATRIX"
