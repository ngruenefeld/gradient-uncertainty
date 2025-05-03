import argparse
import os
import random

import pandas as pd
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from utils.gpt import rephrase_text
from utils.utils import (
    get_response,
    completion_gradient,
    replace_tokens_with_synonyms,
    replace_tokens_with_random_tokens,
    load_multilingual_datasets,
)


def main(args):
    job_number = args.job_number
    dataset_name = args.dataset
    model_name = args.model
    gpt_model = args.gpt_model
    key_mode = args.key_mode
    sample_size = args.sample_size
    quant_bits = args.quantization
    full_gradient = args.full_gradient
    response_only = not full_gradient  # Inverse of full_gradient
    normalize = args.normalize
    perturbation_mode = args.perturbation_mode
    number_of_perturbations = args.number_of_perturbations

    mode = "full" if sample_size == 0 else "test" if sample_size < 100 else "sampled"

    print(f"Job number: {job_number}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"GPT Model: {gpt_model}")
    print(f"Key mode: {key_mode}")
    print(f"Sample size: {sample_size}")
    print(f"Mode: {mode}")
    print(
        f"Quantization bits: {quant_bits if quant_bits > 0 else 'None (full precision)'}"
    )
    print(f"Full gradient: {full_gradient}")
    print(f"Response only: {response_only}")
    print(f"Normalize: {normalize}")
    print(f"Perturbation mode: {perturbation_mode}")
    print(f"Number of perturbations: {number_of_perturbations}")

    if model_name == "gpt2":
        model_path = "gpt2"
    elif model_name == "llama-awq":
        model_path = "TheBloke/Llama-2-7B-Chat-AWQ"
    elif model_name == "llama-3-8b":
        model_path = "meta-llama/Meta-Llama-3-8B"
    elif model_name == "llama-3.1-8b":
        model_path = "meta-llama/Llama-3.1-8B"
    elif model_name == "llama-3.2-3b":
        model_path = "meta-llama/Llama-3.2-3B"
    elif model_name == "llama-3-chatqa-quantized":
        model_path = "nvidia/Llama3-ChatQA-1.5-8B"
    elif model_name == "deepseek-r1-distill-qwen-1.5b":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    elif model_name == "phi4":
        model_path = "microsoft/Phi-4"
    else:
        raise ValueError(
            f"Model {model_name} not recognized. Please use one of the following: gpt2, llama-awq, llama-3-8b, llama-3.1-8b, llama-3.2-3b, llama-3-chatqa-quantized, deepseek-r1-distill-qwen-1.5b, phi4, deepseek-v3."
        )

    if key_mode == "keyfile":
        with open(os.path.expanduser(".oai_api_key"), "r") as f:
            oai_api_key = f.read().strip()
        with open(os.path.expanduser(".hf_api_key"), "r") as f:
            hf_token = f.read().strip()
    elif key_mode == "env":
        oai_api_key = os.getenv("OPENAI_API_KEY")
        hf_token = os.getenv("HF_TOKEN")
        if oai_api_key is None:
            raise ValueError(
                "API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        if hf_token is None:
            raise ValueError(
                "API key not found. Please set the HF_TOKEN environment variable."
            )
    else:
        raise ValueError("Invalid key mode. Please use 'keyfile' or 'env'.")

    oai_client = OpenAI(api_key=oai_api_key)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with quantization if requested
    quantizable_models = [
        "llama-3-8b",
        "llama-3.1-8b",
        "llama-3.2-3b",
        "phi4",
    ]

    # Prepare common model loading parameters
    model_load_params = {
        "token": hf_token,
    }

    if quant_bits in [4, 8] and model_name in quantizable_models:
        print(f"Loading model in {quant_bits}-bit precision to reduce memory usage")

        # Configure quantization based on bit depth
        if quant_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quant_bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
            )

        # With quantization, we must use device_map="auto" instead of .to(device)
        model_load_params["quantization_config"] = quantization_config
        model_load_params["device_map"] = "auto"

        # Load the model with all parameters
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_params)
    else:
        # Load the model with base parameters
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    tokenizer_params = {"token": hf_token}

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_params)

    # Load datasets in streaming mode if requested
    data = load_multilingual_datasets(dataset_name)

    # Handle dataset sampling based on streaming mode
    if sample_size > 0:
        data_list = list(data)
        indices = random.sample(range(len(data_list)), sample_size)
        data_samples = [(i, data_list[i]) for i in indices]
    else:
        data_samples = list(enumerate(data))

    results = []
    processed_count = 0
    failed_count = 0

    total_samples = sample_size if sample_size > 0 else len(data_samples)

    for dataset_idx, item in data_samples:
        current_sample = processed_count + failed_count + 1
        print(
            f"Processing sample {current_sample}/{total_samples} (dataset index: {dataset_idx})"
        )
        try:
            article = item["text"]
            language = item["language"]
            origin = item["origin"]

            if language != "en":
                print(
                    f"Skipping sample {current_sample} (dataset index {dataset_idx}) due to unsupported language: {language}"
                )
                failed_count += 1
                continue

            prompt = "Summarize the following text:\n" + article

            # Get response (with built-in error handling)
            completion_result = get_response(prompt, model, tokenizer, device)
            if isinstance(completion_result, dict) and "error" in completion_result:
                print(
                    f"Error getting response for sample {current_sample} (dataset index {dataset_idx}): {completion_result['error']}"
                )
                failed_count += 1
                torch.cuda.empty_cache()  # Ensure memory is freed
                continue
            completion = completion_result

            # Clear memory after getting response
            torch.cuda.empty_cache()

            # Calculate gradient (with built-in error handling)
            gradient_result = completion_gradient(
                prompt,
                completion,
                model,
                tokenizer,
                device,
                response_only=response_only,
                normalize=normalize,
            )
            if isinstance(gradient_result, dict) and "error" in gradient_result:
                print(
                    f"Error calculating gradient for sample {current_sample} (dataset index {dataset_idx}): {gradient_result['error']}"
                )
                failed_count += 1
                torch.cuda.empty_cache()  # Ensure memory is freed
                continue

            gradient, completion_length = gradient_result
            gradient_norm = torch.norm(gradient).item()

            # Clear memory after calculating gradient
            torch.cuda.empty_cache()

            if perturbation_mode == "rephrase":
                # Get rephrasings
                rephrasings_result = rephrase_text(completion, oai_client, gpt_model)
                if "error" in rephrasings_result:
                    print(
                        f"Error getting rephrasings for sample {current_sample} (dataset index {dataset_idx}): {rephrasings_result['error']}"
                    )
                    failed_count += 1
                    torch.cuda.empty_cache()  # Ensure memory is freed
                    continue

                rephrasings = rephrasings_result["rephrasings"]

            elif perturbation_mode == "synonym":
                rephrasings = []
                for _ in range(number_of_perturbations):
                    synonym_inputs = tokenizer(
                        completion,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(device)
                    modified_input_ids = replace_tokens_with_synonyms(
                        synonym_inputs, tokenizer, device, replacement_prob=1.0
                    )
                    modified_sentence = tokenizer.decode(modified_input_ids[0])
                    rephrasings.append(modified_sentence)

            elif perturbation_mode == "random":
                rephrasings = []
                for _ in range(number_of_perturbations):
                    synonym_inputs = tokenizer(
                        completion,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(device)
                    modified_input_ids = replace_tokens_with_random_tokens(
                        synonym_inputs, tokenizer, device, replacement_prob=1.0
                    )
                    modified_sentence = tokenizer.decode(modified_input_ids[0])
                    rephrasings.append(modified_sentence)

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
                # Clear memory before processing each rephrasing
                torch.cuda.empty_cache()

                rephrasing_gradient_result = completion_gradient(
                    prompt,
                    phrasing,
                    model,
                    tokenizer,
                    device,
                    response_only=response_only,
                    normalize=normalize,
                )

                if (
                    isinstance(rephrasing_gradient_result, dict)
                    and "error" in rephrasing_gradient_result
                ):
                    print(
                        f"Error calculating gradient for rephrasing {idx} in sample {current_sample} (dataset index {dataset_idx}): {rephrasing_gradient_result['error']}"
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
                "language": language,
                "origin": origin,
                "completion": completion,
                "completion_length": completion_length,
                "completion_gradient": gradient_norm,
                "rephrased_completions": rephrasings,
                "rephrased_completion_lengths": rephrasing_lengths,
                "rephrased_gradients": rephrasing_gradient_norms,
                "rephrased_gradient_std": rephrasing_gradient_std,
            }
            results.append(result_entry)
            processed_count += 1
            print(
                f"Sample {current_sample} (dataset index {dataset_idx}) processed successfully with {n} rephrasings."
            )

        except Exception as e:
            print(
                f"Error processing sample {current_sample} (dataset index {dataset_idx}): {str(e)}"
            )
            failed_count += 1
            # Clear all CUDA cache and continue
            torch.cuda.empty_cache()
            continue

        # Clear memory after each sample
        torch.cuda.empty_cache()

    # Save final results if we have any
    if results:
        # Include quantization, response_only, and normalize info in the filename
        quant_suffix = (
            f"_{quant_bits}bit"
            if quant_bits in [4, 8] and model_name in quantizable_models
            else ""
        )
        response_suffix = "" if response_only else "_fullgradient"
        normalize_suffix = "_normalized" if normalize else ""

        df = pd.DataFrame(results)
        df.to_pickle(
            f"data/{mode}/results_{job_number}_{model_name}{quant_suffix}{response_suffix}{normalize_suffix}_{perturbation_mode}_{dataset_name}.pkl"
        )
        print(
            f"Processing complete. Saved {len(results)} successful results. Failed: {failed_count}"
        )
    else:
        print(
            f"Processing complete, but no successful results to save. All {failed_count} samples failed."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA uncertainty measurement")

    parser.add_argument("job_number", help="Unique identifier for this job run")
    parser.add_argument(
        "--dataset",
        type=str,
        default="finenews",
        help="Dataset to use: finenews",
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
    parser.add_argument(
        "--quantization",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="Quantization precision: 0 (none/default), 4 (4-bit), or 8 (8-bit)",
    )
    parser.add_argument(
        "--full_gradient",
        action="store_true",
        default=False,
        help="Only calculate gradients for the response tokens (default: True)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize the gradients (default: False)",
    )
    parser.add_argument(
        "--perturbation_mode",
        type=str,
        default="rephrase",
        help="Mode for perturbation: rephrase, synonym, or random",
    )
    parser.add_argument(
        "--number_of_perturbations",
        type=int,
        default=3,
        help="Number of perturbations to generate for each sample",
    )

    args = parser.parse_args()

    main(args)
