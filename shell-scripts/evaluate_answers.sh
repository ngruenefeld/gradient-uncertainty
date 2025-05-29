#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-eval
#SBATCH --comment="Running the answer evaluation script"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1

# Default values
GPT_MODEL="gpt-4o-mini-2024-07-18"
KEY_MODE="keyfile"
OVERWRITE=true

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpt_model=*) GPT_MODEL="${1#*=}";;
        --key_mode=*) KEY_MODE="${1#*=}";;
        --no-overwrite) OVERWRITE=false;;
        *) 
            if [ -z "$JOB_NUMBER" ]; then
                JOB_NUMBER="$1"
            else
                echo "Unknown option: $1"
            fi
            ;;
    esac
    shift
done

# Check if job number is provided
if [ -z "$JOB_NUMBER" ]; then
    echo "Error: Job number is required as the first argument"
    echo "Usage: $0 <job_number> [options]"
    echo "Options:"
    echo "  --gpt_model=MODEL     GPT model to use (default: gpt-4o-mini-2024-07-18)"
    echo "  --key_mode=MODE       Key mode: keyfile or env (default: keyfile)"
    echo "  --no-overwrite        Don't overwrite existing evaluations (default: overwrite)"
    exit 1
fi

# Get current commit hash
COMMIT_ID=$(git rev-parse HEAD)
echo "Running evaluation job with commit: $COMMIT_ID"
echo "Job number: $JOB_NUMBER"

# Activate virtual environment
source env/bin/activate

# Build the command with all required parameters
CMD="python -um scripts.evaluate_answers \"$JOB_NUMBER\" --gpt_model \"$GPT_MODEL\" --key_mode \"$KEY_MODE\""

# Add no-overwrite flag if disabled (since overwrite is default true)
if [ "$OVERWRITE" = false ]; then
    CMD="$CMD --no-overwrite"
fi

echo "Running command: $CMD"

# Run the command
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully"
else
    echo "Evaluation failed with exit code $?"
fi

# Deactivate and commit results
deactivate
git add .
git commit -m "Answer Evaluation Results for Job $JOB_NUMBER (Commit: ${COMMIT_ID:0:7})"
git push