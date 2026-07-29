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

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# The generator is built from the Graph500 reference sources, which live in an
# uninitialised submodule in a fresh clone.
if [[ ! -r graph500/generator/graph_generator.c ]]; then
    git submodule update --init --recursive graph500 ||
        {
            echo "Populate graph500/ first:" >&2
            echo "  git clone https://github.com/graph500/graph500 graph500" >&2
            exit 1
        }
fi

make rmat
mkdir -p matrices

# rmat_22: approximately 38M expanded NNZ
./bin/generate_rmat \
    --scale 22 \
    --edges 18874368 \
    --output matrices/rmat_22.mtx

# To generate a larger matrix, replace the parameters above with one of these:
# rmat_24: --scale 24 --edges 75497472 --output matrices/rmat_24.mtx
# rmat_26: --scale 26 --edges 301989888 --output matrices/rmat_26.mtx
