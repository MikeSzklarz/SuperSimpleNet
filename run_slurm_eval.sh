#!/bin/bash

#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --partition=waccamaw

set -e

# --- Environment Setup ---
echo "Loading Conda environment..."
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate ssn_env
echo "Environment loaded."

# --- Evaluation ---
# All arguments passed to sbatch are forwarded here, e.g.:
#   sbatch --nodelist=waccamaw01 run_slurm_eval.sh \
#     --data-root /path/to/data \
#     --results-dir /path/to/results
echo "Running:"
echo "  python eval.py $@"
echo ""

mkdir -p logs/slurm

srun python -u eval.py "$@"

echo "Job finished successfully."
