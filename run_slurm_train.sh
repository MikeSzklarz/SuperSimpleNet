#!/bin/bash
# Usage: ./run_slurm.sh <dataset> <category> [slurm_flags...]
# Example: ./run_slurm.sh bowtie color_profile_1 --nodelist=waccamaw04

# --- 1. CONFIGURATION ---
DATASET=$1
CATEGORY=$2

# Check for mandatory arguments
if [ -z "$DATASET" ] || [ -z "$CATEGORY" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <dataset> <category> [additional_slurm_flags]"
    echo "Example: $0 bowtie color_profile_1 --nodelist=waccamaw04"
    exit 1
fi

# Shift the first two arguments (dataset and category) so only flags remain in $@
shift 2

# Job Name: ssn_<dataset>_<category>
JOB_NAME="ssn_${DATASET}_${CATEGORY}"
LOG_DIR="logs/slurm"
mkdir -p "$LOG_DIR"

# --- 2. SUBMIT BATCH JOB ---
# We pass "$@" to sbatch, which injects your nodelist/partition flags
echo "Submitting job: $JOB_NAME"
echo "Extra flags: $@"

sbatch "$@" <<EOT
#!/bin/bash
#SBATCH --job-name="$JOB_NAME"
#SBATCH --output="${LOG_DIR}/${JOB_NAME}_%j.out"
#SBATCH --error="${LOG_DIR}/${JOB_NAME}_%j.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1           
#SBATCH --mem=32G              
#SBATCH --time=24:00:00
#SBATCH --partition=waccamaw   # Default partition (can be overridden by flag)

set -e

# --- 3. ENVIRONMENT SETUP ---
echo "Loading environment..."
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate ssn_env

echo "Environment loaded."
echo "Running SuperSimpleNet Training on $DATASET / $CATEGORY"

# --- 4. RUN TRAINING ---
# Note: We use the variables captured at the very top of the script
python train.py "$DATASET" "$CATEGORY"

echo "Job finished successfully."
EOT