#!/bin/bash
#
#SBATCH --job-name=gruenefeld-ma-llm
#SBATCH --comment="Running the LLM fine-tuning script"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=N.Gruenefeld@campus.lmu.de
#SBATCH --chdir=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty
#SBATCH --output=/home/g/gruenefeld/Documents/GitHub/gradient-uncertainty/slurm-outputs/slurm.%j.%N.out
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Default values
KEY_MODE="keyfile"
SAMPLE_SIZE=0
TEST_SAMPLE_SIZE=0
NORMALIZE=false  # Default to false
COUNTERFACTUAL="identity"  # Default to identity
DATASET="ag_news"  # Default dataset choice
MODEL="gpt2"  # Default model choice
REPLACEMENT_PROB=1.0  # Default replacement probability
QUANTIZATION=0  # 0 = no quantization (default)

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --key_mode=*) KEY_MODE="${1#*=}";;
        --sample_size=*) SAMPLE_SIZE="${1#*=}";;
        --test_sample_size=*) TEST_SAMPLE_SIZE="${1#*=}";;
        --normalize) NORMALIZE=true;;
        --counterfactual=*) COUNTERFACTUAL="${1#*=}";;
        --dataset=*) DATASET="${1#*=}";;
        --model=*) MODEL="${1#*=}";;
        --replacement_prob=*) REPLACEMENT_PROB="${1#*=}";;
        --quantization=*) QUANTIZATION="${1#*=}";;
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
CMD="python -um scripts.llama \"$SLURM_JOB_ID\" --key_mode \"$KEY_MODE\" --sample_size \"$SAMPLE_SIZE\" --test_sample_size \"$TEST_SAMPLE_SIZE\" --quantization $QUANTIZATION"

# Add normalize parameter (only add if true)
if [ "$NORMALIZE" = true ]; then
    CMD="$CMD --normalize"
fi

# Add counterfactual parameter (only add if not default)
if [ "$COUNTERFACTUAL" != "identity" ]; then
    CMD="$CMD --counterfactual \"$COUNTERFACTUAL\""
fi

# Add dataset parameter (only add if not default)
if [ "$DATASET" != "ag_news" ]; then
    CMD="$CMD --dataset \"$DATASET\""
fi

# Add model parameter (only add if not default)
if [ "$MODEL" != "gpt2" ]; then
    CMD="$CMD --model \"$MODEL\""
fi

if [ "$REPLACEMENT_PROB" != "1.0" ] && { [ "$COUNTERFACTUAL" = "synonym" ] || [ "$COUNTERFACTUAL" = "random" ]; }; then
    CMD="$CMD --replacement_prob \"$REPLACEMENT_PROB\""
fi

# Run the command
echo "Running command: $CMD"
eval $CMD

# Deactivate and commit results
deactivate
git add .
git commit -m "LLM Script Results for Run $SLURM_JOB_ID (Model: $MODEL, Dataset: $DATASET, Quantization: ${QUANTIZATION}bit, Commit: ${COMMIT_ID:0:7})"
git push