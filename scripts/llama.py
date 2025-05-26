import argparse
import os
import pandas as pd
import random

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
import torch

from utils.utils import (
    bert_gradient,
    load_bert_datasets,
    replace_tokens_with_synonyms,
    replace_tokens_with_random_tokens,
)


def process_test_samples(
    test_dataset,
    model,
    tokenizer,
    device,
    phase="before",
    results=None,
    normalize=False,
    counterfactual="identity",
    replacement_prob=1.0,
):
    if results is None:
        results = []

    failed_count = 0
    sample_count = len(test_dataset)

    print(
        f"Calculating uncertainties for {sample_count} test samples {phase} training (counterfactual mode: {counterfactual})..."
    )

    for idx, item in enumerate(test_dataset):
        try:
            sentence = item["text"]
            label = item["label"]
            origin = item["origin"]
            print(f"Processing test sample {idx+1}/{sample_count} ({phase} training)")

            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=128,
            ).to(device)

            if counterfactual == "constant":
                # Use the unknown token ID instead of input tokens
                unk_token_id = (
                    tokenizer.unk_token_id
                    if tokenizer.unk_token_id is not None
                    else tokenizer.pad_token_id
                )
                labels = torch.full_like(inputs.input_ids, unk_token_id).to(device)
            elif counterfactual == "synonym":
                # Replace tokens with synonyms
                synonym_input_ids = replace_tokens_with_synonyms(
                    inputs, tokenizer, device, replacement_prob=replacement_prob
                )
                labels = synonym_input_ids.clone().to(device)
            elif counterfactual == "random":
                # Replace tokens with random tokens
                random_input_ids = replace_tokens_with_random_tokens(
                    inputs, tokenizer, device, replacement_prob=replacement_prob
                )
                labels = random_input_ids.clone().to(device)
            elif counterfactual == "identity":
                labels = inputs.input_ids.clone().to(device)
            else:
                print(
                    f"Warning: Invalid counterfactual mode '{counterfactual}'. Using 'identity' mode."
                )
                labels = inputs.input_ids.clone().to(device)

            uncertainty = bert_gradient(
                inputs, labels, model, normalize=normalize
            ).item()

            if phase == "before":
                result_entry = {
                    "text": sentence,
                    "label": label,
                    "origin": origin,
                    "uncertainty_before": uncertainty,
                    "uncertainty_after": None,
                    "uncertainty_difference": None,
                }
                results.append(result_entry)
            else:
                # Update existing result entry
                results[idx]["uncertainty_after"] = uncertainty
                results[idx]["uncertainty_difference"] = (
                    uncertainty - results[idx]["uncertainty_before"]
                )

        except Exception as e:
            print(f"Error processing test sample {idx+1} {phase} training: {str(e)}")
            failed_count += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    return results, failed_count


def main(args):
    key_mode = args.key_mode
    sample_size = args.sample_size
    job_number = args.job_number
    normalize = args.normalize
    counterfactual = args.counterfactual
    dataset_choice = args.dataset
    model_name = args.model
    replacement_prob = args.replacement_prob
    quant_bits = args.quantization
    epochs = args.epochs

    print(f"Job number: {job_number}")
    print(f"Key mode: {key_mode}")
    print(f"Sample size: {sample_size}")
    print(f"Normalize: {normalize}")
    print(f"Counterfactual: {counterfactual}")
    print(f"Dataset: {dataset_choice}")
    print(f"Model: {model_name}")
    print(f"Replacement probability: {replacement_prob}")
    print(
        f"Quantization bits: {quant_bits if quant_bits > 0 else 'None (full precision)'}"
    )
    print(f"Epochs: {epochs}")

    if key_mode == "keyfile":
        with open(os.path.expanduser(".hf_api_key"), "r") as f:
            hf_token = f.read().strip()
    elif key_mode == "env":
        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "API key not found. Please set the HF_TOKEN environment variable."
            )
    else:
        raise ValueError("Invalid key mode. Please use 'keyfile' or 'env'.")

    # Pass the dataset choice parameter to load_bert_datasets
    train_dataset, test_dataset = load_bert_datasets(choice=dataset_choice)

    train_sample_size = (
        min(args.sample_size, len(train_dataset))
        if args.sample_size > 0
        else len(train_dataset)
    )
    if train_sample_size < len(train_dataset):
        train_indices = random.sample(range(len(train_dataset)), train_sample_size)
        train_dataset = train_dataset.select(train_indices)
        print(
            f"Using {train_sample_size} randomly sampled examples from the train dataset."
        )
    else:
        print(f"Using full train dataset with {train_sample_size} samples.")

    test_sample_size = (
        min(args.test_sample_size, len(test_dataset))
        if args.test_sample_size > 0
        else len(test_dataset)
    )
    if test_sample_size < len(test_dataset):
        test_indices = random.sample(range(len(test_dataset)), test_sample_size)
        test_dataset = test_dataset.select(test_indices)
        print(
            f"Using {test_sample_size} randomly sampled examples from the test dataset."
        )
    else:
        print(f"Using full test dataset with {test_sample_size} samples.")

    # Define model paths based on model selection
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
            f"Model {model_name} not recognized. Please use one of the following: gpt2, llama-awq, llama-3-8b, llama-3.1-8b, llama-3.2-3b, llama-3-chatqa-quantized, deepseek-r1-distill-qwen-1.5b, phi4."
        )

    print(f"Loading model: {model_path}")

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
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Using device: {device}")

    model.train()

    def tokenize_function(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )
        # For causal LM, labels are the same as input_ids but shifted
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "label", "origin"]
    )

    # Use DataCollatorForLanguageModeling without MLM for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False for causal language modeling
    )

    training_args = TrainingArguments(
        output_dir=f"/tmp/llama_run_{job_number}",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,  # Smaller batch size for LLMs
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # To maintain effective batch size
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        save_total_limit=0,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,  # Save memory for large models
        dataloader_pin_memory=False,  # Reduce memory usage
        learning_rate=5e-5,  # Lower learning rate for LLMs
        warmup_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Process test dataset and calculate uncertainty before training
    results, failed_count_before = process_test_samples(
        test_dataset,
        model,
        tokenizer,
        device,
        phase="before",
        normalize=normalize,
        counterfactual=counterfactual,
        replacement_prob=args.replacement_prob,
    )

    # Train the model
    print("Training model...")
    trainer.train()

    # Calculate uncertainty after training
    results, failed_count_after = process_test_samples(
        test_dataset,
        model,
        tokenizer,
        device,
        phase="after",
        results=results,
        normalize=normalize,
        counterfactual=counterfactual,
        replacement_prob=args.replacement_prob,
    )

    # Total failed count
    failed_count = failed_count_before + failed_count_after

    if results:
        mode = (
            "full" if sample_size == 0 else "test" if sample_size < 100 else "sampled"
        )
        df = pd.DataFrame(results)

        # Add normalization indicator to filename
        normalize_suffix = "_normalized" if normalize else ""
        # Add counterfactual mode to filename if it's not the default
        counterfactual_suffix = (
            f"_cf-{counterfactual}" if counterfactual != "identity" else ""
        )
        # Add dataset choice to filename
        dataset_suffix = f"_ds-{dataset_choice}"
        # Add quantization info to filename
        quant_suffix = (
            f"_{quant_bits}bit"
            if quant_bits in [4, 8] and model_name in quantizable_models
            else ""
        )

        df.to_pickle(
            f"data/{mode}/{model_name}_results_{job_number}{normalize_suffix}{counterfactual_suffix}{dataset_suffix}{quant_suffix}.pkl"
        )
        print(
            f"\nProcessing complete. Saved {len(results)} results. Failed: {failed_count}"
        )
    else:
        print(
            f"\nProcessing complete, but no successful results to save. All {failed_count} samples failed."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Causal language model fine-tuning and uncertainty measurement"
    )

    parser.add_argument("job_number", help="Unique identifier for this job run")
    parser.add_argument(
        "--key_mode",
        type=str,
        default="keyfile",
        choices=["keyfile", "env"],
        help="Whether to read the HuggingFace API key from a file or use an environment variable",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of examples to sample from the training dataset (0 = use full dataset)",
    )
    parser.add_argument(
        "--test_sample_size",
        type=int,
        default=0,
        help="Number of examples to sample from the test dataset (0 = use full dataset)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize the gradients (default: False)",
    )
    parser.add_argument(
        "--counterfactual",
        type=str,
        default="identity",
        choices=["identity", "constant", "synonym", "random"],
        help="How to choose labels for gradient calculation: 'identity' uses the same input, 'constant' uses the unknown token (default: identity)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ag_news",
        choices=["ag_news", "ag-pubmed", "mmlu", "scienceqa", "scienceqa-legalqa"],
        help="Which dataset to use for training and testing (default: ag_news)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=[
            "gpt2",
            "llama-awq",
            "llama-3-8b",
            "llama-3.1-8b",
            "llama-3.2-3b",
            "llama-3-chatqa-quantized",
            "deepseek-r1-distill-qwen-1.5b",
            "phi4",
        ],
        help="Which model to use for training and testing (default: gpt2)",
    )
    parser.add_argument(
        "--replacement_prob",
        type=float,
        default=1.0,
        help="Probability of replacing tokens when using 'synonym' or 'random' counterfactual modes (default: 1.0)",
    )
    parser.add_argument(
        "--quantization",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="Quantization precision: 0 (none/default), 4 (4-bit), or 8 (8-bit)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    args = parser.parse_args()

    main(args)
