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

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset=*) DATASET="${1#*=}";;
        --model=*) MODEL="${1#*=}";;
        --gpt_model=*) GPT_MODEL="${1#*=}";;
        --key_mode=*) KEY_MODE="${1#*=}";;
        --sample_size=*) SAMPLE_SIZE="${1#*=}";;
        *) echo "Unknown option: $1" ;;
    esac
    shift
done

# Activate virtual environment
source env/bin/activate

# Run the Python script with all the parsed arguments
python -um scripts.qa "$SLURM_JOB_ID" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --gpt_model "$GPT_MODEL" \
    --key_mode "$KEY_MODE" \
    --sample_size "$SAMPLE_SIZE"

# Deactivate and commit results
deactivate
git add .
git commit -m "QA Script Results for Run $SLURM_JOB_ID"
git push