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
JOB_NUMBERS=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpt_model=*) GPT_MODEL="${1#*=}";;
        --key_mode=*) KEY_MODE="${1#*=}";;
        --no-overwrite) OVERWRITE=false;;
        --help|-h) 
            echo "Usage: $0 <job_number1> [job_number2 ...] [options]"
            echo "Multiple job numbers can be provided to process them sequentially."
            echo ""
            echo "Options:"
            echo "  --gpt_model=MODEL     GPT model to use (default: gpt-4o-mini-2024-07-18)"
            echo "  --key_mode=MODE       Key mode: keyfile or env (default: keyfile)"
            echo "  --no-overwrite        Don't overwrite existing evaluations (default: overwrite)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 12345                    # Single job"
            echo "  $0 12345 12346 12347        # Multiple jobs"
            echo "  $0 12345 --gpt_model=gpt-4-turbo --no-overwrite"
            exit 0
            ;;
        --*) 
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *) 
            # Assume it's a job number
            JOB_NUMBERS+=("$1")
            ;;
    esac
    shift
done

# Check if at least one job number is provided
if [ ${#JOB_NUMBERS[@]} -eq 0 ]; then
    echo "Error: At least one job number is required"
    echo "Usage: $0 <job_number1> [job_number2 ...] [options]"
    echo "Use --help for more information"
    exit 1
fi

# Get current commit hash
COMMIT_ID=$(git rev-parse HEAD)
echo "Running evaluation job with commit: $COMMIT_ID"
echo "Job numbers to process: ${JOB_NUMBERS[*]}"
echo "Total jobs: ${#JOB_NUMBERS[@]}"

# Activate virtual environment
source env/bin/activate

# Initialize counters
SUCCESSFUL_JOBS=0
FAILED_JOBS=0
FAILED_JOB_NUMBERS=()

# Process each job number
for i in "${!JOB_NUMBERS[@]}"; do
    JOB_NUMBER="${JOB_NUMBERS[$i]}"
    JOB_INDEX=$((i + 1))
    
    echo ""
    echo "========================================="
    echo "Processing job $JOB_INDEX/${#JOB_NUMBERS[@]}: $JOB_NUMBER"
    echo "========================================="
    
    # Build the command with all required parameters
    CMD="python -um scripts.evaluate_answers \"$JOB_NUMBER\" --gpt_model \"$GPT_MODEL\" --key_mode \"$KEY_MODE\""

    # Add no-overwrite flag if disabled (since overwrite is default true)
    if [ "$OVERWRITE" = false ]; then
        CMD="$CMD --no-overwrite"
    fi

    echo "Running command: $CMD"
    
    # Run the command and capture exit status
    eval $CMD
    EXIT_STATUS=$?
    
    # Check exit status and update counters
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "✅ Job $JOB_NUMBER completed successfully"
        SUCCESSFUL_JOBS=$((SUCCESSFUL_JOBS + 1))
    else
        echo "❌ Job $JOB_NUMBER failed with exit code $EXIT_STATUS"
        FAILED_JOBS=$((FAILED_JOBS + 1))
        FAILED_JOB_NUMBERS+=("$JOB_NUMBER")
    fi
done

echo ""
echo "========================================="
echo "EVALUATION SUMMARY"
echo "========================================="
echo "Total jobs processed: ${#JOB_NUMBERS[@]}"
echo "Successful: $SUCCESSFUL_JOBS"
echo "Failed: $FAILED_JOBS"

if [ $FAILED_JOBS -gt 0 ]; then
    echo "Failed job numbers: ${FAILED_JOB_NUMBERS[*]}"
fi

# Deactivate and commit results
deactivate

# Create commit message based on results
if [ $FAILED_JOBS -eq 0 ]; then
    COMMIT_MSG="Answer Evaluation Results for Jobs ${JOB_NUMBERS[*]} - All Successful (Commit: ${COMMIT_ID:0:7})"
else
    COMMIT_MSG="Answer Evaluation Results for Jobs ${JOB_NUMBERS[*]} - $SUCCESSFUL_JOBS/$((SUCCESSFUL_JOBS + FAILED_JOBS)) Successful (Commit: ${COMMIT_ID:0:7})"
fi

git add .
git commit -m "$COMMIT_MSG"
git push

echo ""
echo "Results committed to git with message: $COMMIT_MSG"

# Exit with appropriate code
if [ $FAILED_JOBS -gt 0 ]; then
    echo "Exiting with error code due to failed jobs"
    exit 1
else
    echo "All jobs completed successfully"
    exit 0
fi