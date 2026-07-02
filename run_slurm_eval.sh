#!/bin/bash

#SBATCH --job-name=ssn-eval
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=gpu

set -e

# --- Environment Setup ---
echo "Loading Conda environment..."
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate ssn_env
echo "Environment loaded."

# --- Evaluation ---
# All arguments passed to sbatch are forwarded here, e.g.:
#   sbatch run_slurm_eval.sh --data-root /path/to/data \
#                             --results-dir /path/to/results
echo "Running:"
echo "  python eval.py $@"
echo ""

mkdir -p logs/slurm

srun python -u eval.py "$@"

echo "Job finished successfully."
