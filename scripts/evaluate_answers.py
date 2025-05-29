import argparse
import os
import glob
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from utils.gpt import evaluate_answers


def load_results_file(file_path):
    try:
        df = pd.read_pickle(file_path)
        return df.to_dict("records")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_results_file(results, file_path):
    try:
        df = pd.DataFrame(results)
        df.to_pickle(file_path)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def needs_evaluation(result_entry):
    return "evaluation" not in result_entry or result_entry["evaluation"] is None


def evaluate_results_file(file_path, oai_client, gpt_model, overwrite=False):
    print(f"Processing file: {file_path}")

    # Load results
    results = load_results_file(file_path)
    if results is None:
        return False

    # Count entries that need evaluation
    entries_to_evaluate = [r for r in results if needs_evaluation(r) or overwrite]

    if not entries_to_evaluate:
        print(f"  No entries need evaluation in {file_path}")
        return True

    print(
        f"  Found {len(entries_to_evaluate)} entries to evaluate out of {len(results)} total"
    )

    # Process each entry that needs evaluation
    successful_evaluations = 0
    failed_evaluations = 0

    for i, result_entry in enumerate(tqdm(results, desc="Evaluating entries")):
        if not (needs_evaluation(result_entry) or overwrite):
            continue

        try:
            # Extract required information
            prompt = result_entry["question"]
            completion = result_entry["completion"]
            answers = result_entry["correct_answers"]

            # Evaluate the answer
            evaluation = evaluate_answers(
                prompt, completion, answers, oai_client, gpt_model
            )

            # Update the result entry
            result_entry["evaluation"] = evaluation["is_correct"]

            successful_evaluations += 1

        except Exception as e:
            print(f"    Error evaluating entry {i}: {e}")
            failed_evaluations += 1
            continue

    print(
        f"  Completed evaluations: {successful_evaluations} successful, {failed_evaluations} failed"
    )

    # Save updated results
    if successful_evaluations > 0:
        success = save_results_file(results, file_path)
        if success:
            print(f"  Updated results saved to {file_path}")
        else:
            print(f"  Error saving updated results to {file_path}")
        return success

    return True


def main(args):
    job_number = args.job_number
    gpt_model = args.gpt_model
    key_mode = args.key_mode
    overwrite = args.overwrite

    print(f"Job number: {job_number}")
    print(f"GPT Model: {gpt_model}")
    print(f"Key mode: {key_mode}")
    print(f"Overwrite existing evaluations: {overwrite}")

    # Setup API key
    if key_mode == "keyfile":
        with open(os.path.expanduser(".oai_api_key"), "r") as f:
            oai_api_key = f.read().strip()
    elif key_mode == "env":
        oai_api_key = os.getenv("OPENAI_API_KEY")
        if oai_api_key is None:
            raise ValueError(
                "API key not found. Please set the OPENAI_API_KEY environment variable."
            )
    else:
        raise ValueError("Invalid key mode. Please use 'keyfile' or 'env'.")

    oai_client = OpenAI(api_key=oai_api_key)

    # Find files to process
    file_pattern = f"data/*/results_{job_number}_*.pkl"

    files_to_process = glob.glob(file_pattern)

    if not files_to_process:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(files_to_process)} files to process:")
    for file_path in files_to_process:
        print(f"  {file_path}")

    # Process each file
    successful_files = 0
    failed_files = 0

    for file_path in files_to_process:
        try:
            success = evaluate_results_file(file_path, oai_client, gpt_model, overwrite)
            if success:
                successful_files += 1
            else:
                failed_files += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_files += 1

    print(f"\nEvaluation complete:")
    print(f"  Successfully processed: {successful_files} files")
    print(f"  Failed to process: {failed_files} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers in QA results files")

    parser.add_argument("job_number", help="Job number to identify files to evaluate")
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="GPT model to use for OpenAI API",
    )
    parser.add_argument(
        "--key_mode",
        type=str,
        default="keyfile",
        help="Whether to read the OpenAI API key from a file or use an environment variable",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        default=True,
        help="Don't overwrite existing evaluations (default: overwrite existing evaluations)",
    )

    args = parser.parse_args()
    main(args)
