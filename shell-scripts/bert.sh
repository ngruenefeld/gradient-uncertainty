#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-bert
#SBATCH --comment="Running the BERT fine-tuning script"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1

# Default values
KEY_MODE="keyfile"
SAMPLE_SIZE=0
TEST_SAMPLE_SIZE=0

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --key_mode=*) KEY_MODE="${1#*=}";;
        --sample_size=*) SAMPLE_SIZE="${1#*=}";;
        --test_sample_size=*) TEST_SAMPLE_SIZE="${1#*=}";;
        *) echo "Unknown option: $1" ;;
    esac
    shift
done

# Get current commit hash
COMMIT_ID=$(git rev-parse HEAD)
echo "Running job with commit: $COMMIT_ID"

# Activate virtual environment
source env/bin/activate

# Build the command with all required parameters
CMD="python -um scripts.bert \"$SLURM_JOB_ID\" --key_mode \"$KEY_MODE\" --sample_size \"$SAMPLE_SIZE\" --test_sample_size \"$TEST_SAMPLE_SIZE\""

# Run the command
echo "Running command: $CMD"
eval $CMD

# Create data directory if it doesn't exist
mkdir -p data/bert

# Deactivate and commit results
deactivate
git add .
git commit -m "BERT Script Results for Run $SLURM_JOB_ID (Commit: ${COMMIT_ID:0:7})"
git push