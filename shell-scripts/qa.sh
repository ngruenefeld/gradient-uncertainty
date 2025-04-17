#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-qa
#SBATCH --comment="Running the QA script"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1

# Default values
DATASET="truthful"
MODEL="gpt2"
GPT_MODEL="gpt-4o-mini-2024-07-18"
KEY_MODE="keyfile"
SAMPLE_SIZE=0
STREAMING=false
QUANTIZATION=0  # 0 = no quantization (default)
RESPONSE_ONLY=true  # Default to true
NORMALIZE=false  # Default to false

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset=*) DATASET="${1#*=}";;
        --model=*) MODEL="${1#*=}";;
        --gpt_model=*) GPT_MODEL="${1#*=}";;
        --key_mode=*) KEY_MODE="${1#*=}";;
        --sample_size=*) SAMPLE_SIZE="${1#*=}";;
        --streaming) STREAMING=true;;
        --quantization=*) QUANTIZATION="${1#*=}";;
        --response_only) RESPONSE_ONLY=true;;
        --no-response_only) RESPONSE_ONLY=false;;
        --normalize) NORMALIZE=true;;
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
CMD="python -um scripts.qa \"$SLURM_JOB_ID\" --dataset \"$DATASET\" --model \"$MODEL\" --gpt_model \"$GPT_MODEL\" --key_mode \"$KEY_MODE\" --sample_size \"$SAMPLE_SIZE\""

# Add streaming flag if enabled
if [ "$STREAMING" = true ]; then
    CMD="$CMD --streaming"
fi

# Add quantization parameter
CMD="$CMD --quantization $QUANTIZATION"

# Add no-response_only flag if false (since default is now true)
if [ "$RESPONSE_ONLY" = false ]; then
    CMD="$CMD --no-response_only"
fi

# Add normalize parameter (only add if true)
if [ "$NORMALIZE" = true ]; then
    CMD="$CMD --normalize"
fi

# Run the command
eval $CMD

# Deactivate and commit results
deactivate
git add .
git commit -m "QA Script Results for Run $SLURM_JOB_ID (Commit: ${COMMIT_ID:0:7})"
git push