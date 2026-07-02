#!/bin/bash

#SBATCH --job-name=ssn-train
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --time=50:00:00
#SBATCH --partition=gpu

set -e

# --- Environment Setup ---
echo "Loading Conda environment..."
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate ssn_env
echo "Environment loaded."

# --- Training ---
# All arguments passed to sbatch are forwarded here, e.g.:
#   sbatch run_slurm_train.sh --config configs/custom_fully_sup.txt \
#                              --data-root /path/to/data \
#                              --results-dir /path/to/results
echo "Running:"
echo "  python train.py $@"
echo ""

mkdir -p logs/slurm

srun python -u train.py "$@"

echo "Job finished successfully."
