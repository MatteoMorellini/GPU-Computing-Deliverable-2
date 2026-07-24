#!/bin/bash
#SBATCH --job-name=partition_comm
#SBATCH --output=partition_comm_%j.out
#SBATCH --error=partition_comm_%j.err
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=96G

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

module load OpenMpi/4.1.5-CUDA-12.3.2
module load METIS/5.1.0-GCCcore-12.3.0

export OMPI_MCA_opal_warn_on_missing_libcuda=0

OUTPUT="${OUTPUT:-results/partition_communication_${SLURM_JOB_ID:-local}.csv}"
PARTITION_SEED="${PARTITION_SEED:-20260722}"
LONG_ROW_FRACTION="${LONG_ROW_FRACTION:-0.35}"

make bin/analyze_partition_communication

if [[ -n "${MATRIX:-}" ]]; then
    MATRIX_PATHS=("$MATRIX")
else
    shopt -s nullglob
    MATRIX_PATHS=(matrices/*.mtx)
    if (( ${#MATRIX_PATHS[@]} == 0 )); then
        echo "No .mtx files found in matrices/" >&2
        exit 1
    fi
fi

EXTRA_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--force)
fi

./bin/analyze_partition_communication \
    --output "$OUTPUT" \
    --partition-seed "$PARTITION_SEED" \
    --long-row-fraction "$LONG_ROW_FRACTION" \
    "${EXTRA_ARGS[@]}" \
    "${MATRIX_PATHS[@]}"
