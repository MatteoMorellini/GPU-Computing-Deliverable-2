#!/bin/bash
#SBATCH --job-name=GPU_deliverable-2_mpi
#SBATCH --output=mpi_output_%j.out
#SBATCH --error=mpi_error_%j.err
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# rmat_22: approximately 38M expanded NNZ
./bin/generate_rmat \
    --scale 22 \
    --edges 18874368 \
    --output matrices/rmat_22.mtx
"""
# rmat_24: approximately 151M expanded NNZ
./bin/generate_rmat \
    --scale 24 \
    --edges 75497472 \
    --output matrices/rmat_24.mtx

# rmat_26: approximately 604M expanded NNZ
./bin/generate_rmat \
    --scale 26 \
    --edges 301989888 \
    --output matrices/rmat_26.mtx"""