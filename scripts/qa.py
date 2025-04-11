import argparse
import os
import random
import traceback

import pandas as pd
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.gpt import rephrase_text, evaluate_answers
from utils.utils import get_response, completion_gradient


def main(args):
    job_number = args.job_number
    dataset_name = args.dataset
    model_name = args.model
    gpt_model = args.gpt_model
    key_mode = args.key_mode
    sample_size = args.sample_size

    print(f"Job number: {job_number}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"GPT Model: {gpt_model}")
    print(f"Key mode: {key_mode}")
    print(f"Sample size: {sample_size}")

    if model_name == "gpt2":
        model_path = "gpt2"
    elif model_name == "llama-awq":
        model_path = "TheBloke/Llama-2-7B-Chat-AWQ"

    if key_mode == "keyfile":
        with open(os.path.expanduser(".api_key"), "r") as f:
            api_key = f.read().strip()
    elif key_mode == "env":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not found. Please set the OPENAI_API_KEY environment variable."
            )

    oai_client = OpenAI(api_key=api_key)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if dataset_name == "truthful":
        dataset = load_dataset("truthfulqa/truthful_qa", "generation")
        data = dataset["validation"]
    elif dataset_name == "natural":
        dataset = load_dataset("google-research-datasets/natural_questions", "dev")
        data = dataset["validation"]
    elif dataset_name == "trivia":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
        data = dataset["validation"]

    results = []
    processed_count = 0
    failed_count = 0

    if sample_size > 0:
        indices = random.sample(range(len(data)), sample_size)
    else:
        indices = range(len(data))

    total_samples = len(indices)

    for i in indices:
        current_sample = processed_count + failed_count + 1
        print(
            f"Processing sample {current_sample}/{total_samples} (dataset index: {i})"
        )
        try:
            if dataset_name == "natural":
                prompt = data[i]["question"]["text"]
                answers = [
                    text
                    for sublist in [
                        a["text"] for a in data[i]["annotations"]["short_answers"]
                    ]
                    for text in sublist
                ]
            elif dataset_name == "truthful":
                prompt = data[i]["question"]
                answers = data[i]["correct_answers"]
            elif dataset_name == "trivia":
                prompt = data[i]["question"]
                answers = data[i]["answer"]["aliases"]

            # Get response (with built-in error handling)
            completion_result = get_response(prompt, model, tokenizer, device)
            if isinstance(completion_result, dict) and "error" in completion_result:
                print(
                    f"Error getting response for sample {current_sample} (index {i}): {completion_result['error']}"
                )
                failed_count += 1
                continue
            completion = completion_result

            # Evaluate answers (with built-in error handling)
            evaluation = evaluate_answers(
                prompt, completion, answers, oai_client, gpt_model
            )

            # Calculate gradient (with built-in error handling)
            gradient_result = completion_gradient(
                prompt, completion, model, tokenizer, device
            )
            if isinstance(gradient_result, dict) and "error" in gradient_result:
                print(
                    f"Error calculating gradient for sample {current_sample} (index {i}): {gradient_result['error']}"
                )
                failed_count += 1
                continue

            gradient, completion_length = gradient_result
            gradient_norm = torch.norm(gradient).item()

            # Get rephrasings (with built-in error handling)
            rephrasings_result = rephrase_text(completion, oai_client, gpt_model)
            if "error" in rephrasings_result:
                print(
                    f"Error getting rephrasings for sample {current_sample} (index {i}): {rephrasings_result['error']}"
                )
                failed_count += 1
                continue

            rephrasings = rephrasings_result["rephrasings"]

            rephrasing_lengths = []
            rephrasing_gradient_norms = []
            rephrasing_gradient_std = 0.0

            # Initialize for statistics calculation
            n = 0
            mean = torch.zeros_like(gradient)
            M2 = torch.zeros_like(gradient)

            # Flag to track if any rephrasing fails
            rephrasing_error = False

            # Process each rephrasing
            for idx, phrasing in enumerate(rephrasings):
                rephrasing_gradient_result = completion_gradient(
                    prompt, phrasing, model, tokenizer, device
                )

                if (
                    isinstance(rephrasing_gradient_result, dict)
                    and "error" in rephrasing_gradient_result
                ):
                    print(
                        f"Error calculating gradient for rephrasing {idx} in sample {current_sample} (index {i}): {rephrasing_gradient_result['error']}"
                    )
                    # Mark that we had an error and should skip this sample
                    rephrasing_error = True
                    torch.cuda.empty_cache()
                    break

                rephrasing_gradient, rephrasing_length = rephrasing_gradient_result
                rephrasing_lengths.append(rephrasing_length)
                rephrasing_gradient_norm = torch.norm(rephrasing_gradient).item()
                rephrasing_gradient_norms.append(rephrasing_gradient_norm)

                # Update running variance calculation
                n += 1
                delta = rephrasing_gradient - mean
                mean += delta / n
                delta2 = rephrasing_gradient - mean
                M2 += delta * delta2

                # Free memory
                del rephrasing_gradient
                torch.cuda.empty_cache()

            # Skip this sample if any rephrasing had an error
            if rephrasing_error:
                failed_count += 1
                continue

            # Calculate standard deviation if we have enough data
            if n > 1:
                variance = M2 / (n - 1)
                std_dev = torch.sqrt(variance)
                rephrasing_gradient_std = torch.sum(std_dev).item()

            # Add successful result to our collection
            result_entry = {
                "question": prompt,
                "completion": completion,
                "completion_length": completion_length,
                "correct_answers": answers,
                "evaluation": evaluation["is_correct"],
                "completion_gradient": gradient_norm,
                "rephrased_completions": rephrasings,
                "rephrased_completion_lengths": rephrasing_lengths,
                "rephrased_gradients": rephrasing_gradient_norms,
                "rephrased_gradient_std": rephrasing_gradient_std,
            }
            results.append(result_entry)
            processed_count += 1
            print(
                f"Sample {current_sample} (index {i}) processed successfully with {n} rephrasings."
            )

        except Exception as e:
            print(f"Error processing sample {current_sample} (index {i}): {str(e)}")
            failed_count += 1
            # Clear all CUDA cache and continue
            torch.cuda.empty_cache()
            continue

        # Clear memory after each sample
        torch.cuda.empty_cache()

    # Save final results if we have any
    if results:
        df = pd.DataFrame(results)
        df.to_pickle(f"data/results_{job_number}_{model_name}_{dataset_name}.pkl")
        print(
            f"Processing complete. Saved {len(results)} successful results. Failed: {failed_count}"
        )
    else:
        print(
            f"Processing complete, but no successful results to save. All {failed_count} samples failed."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument("job_number")
    parser.add_argument(
        "--dataset",
        type=str,
        default="truthful",
        help="Dataset to use: truthful, natural, trivia",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model to use, including the path to the model",
    )
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
        "--sample_size",
        type=int,
        default=0,
        help="Set if you want to sample a specific number of examples from the dataset",
    )

    args = parser.parse_args()

    main(args)
