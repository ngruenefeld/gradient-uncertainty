#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-multiling
#SBATCH --comment="Running the multilingual script"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1

# Default values
DATASET="finenews"
MODEL="gpt2"
GPT_MODEL="gpt-4o-mini-2024-07-18"
KEY_MODE="keyfile"
SAMPLE_SIZE=0
QUANTIZATION=0  # 0 = no quantization (default)
FULL_GRADIENT=false  # Default to false (which means response_only is true)
NORMALIZE=false  # Default to false
PERTURBATION_MODE="rephrase"
NUMBER_OF_PERTURBATIONS=3
MAX_TOKENS=0

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset=*) DATASET="${1#*=}";;
        --model=*) MODEL="${1#*=}";;
        --gpt_model=*) GPT_MODEL="${1#*=}";;
        --key_mode=*) KEY_MODE="${1#*=}";;
        --sample_size=*) SAMPLE_SIZE="${1#*=}";;
        --quantization=*) QUANTIZATION="${1#*=}";;
        --full_gradient) FULL_GRADIENT=true;;
        --normalize) NORMALIZE=true;;
        --perturbation_mode=*) PERTURBATION_MODE="${1#*=}";;
        --number_of_perturbations=*) NUMBER_OF_PERTURBATIONS="${1#*=}";;
        --max_tokens=*) MAX_TOKENS="${1#*=}";;
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
CMD="CUDA_LAUNCH_BLOCKING=1 python -um scripts.multiling \"$SLURM_JOB_ID\" --dataset \"$DATASET\" --model \"$MODEL\" --gpt_model \"$GPT_MODEL\" --key_mode \"$KEY_MODE\" --sample_size \"$SAMPLE_SIZE\" --perturbation_mode \"$PERTURBATION_MODE\" --number_of_perturbations \"$NUMBER_OF_PERTURBATIONS\" --max_tokens \"$MAX_TOKENS\""


# Add quantization parameter
CMD="$CMD --quantization $QUANTIZATION"

# Add full_gradient flag if true (full gradient calculation)
if [ "$FULL_GRADIENT" = true ]; then
    CMD="$CMD --full_gradient"
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
git commit -m "Multilingual Script Results for Run $SLURM_JOB_ID (Commit: ${COMMIT_ID:0:7})"
git push