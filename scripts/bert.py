import argparse
import os
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch

from utils.utils import bert_gradient


def process_test_samples(
    test_dataset,
    model,
    tokenizer,
    device,
    phase="before",
    results=None,
    normalize=False,
    counterfactual="identity",
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
            print(f"Processing test sample {idx+1}/{sample_count} ({phase} training)")

            inputs = tokenizer(sentence, return_tensors="pt").to(device)

            if counterfactual == "constant":
                # Use the unknown token ID instead of ones
                unk_token_id = tokenizer.unk_token_id
                labels = torch.full_like(inputs.input_ids, unk_token_id).to(device)
            elif counterfactual == "identity":
                labels = inputs.input_ids.clone().to(device)
            else:
                print(
                    f"Warning: Invalid counterfactual mode '{counterfactual}'. Using 'identity' mode."
                )
                labels = inputs.input_ids.clone().to(device)

            print(f"Inputs: {inputs}")
            print(f"Labels: {labels}")

            uncertainty = bert_gradient(
                inputs, labels, model, normalize=normalize
            ).item()

            if phase == "before":
                result_entry = {
                    "text": sentence,
                    "label": label,
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

    print(f"Job number: {job_number}")
    print(f"Key mode: {key_mode}")
    print(f"Sample size: {sample_size}")
    print(f"Normalize: {normalize}")
    print(f"Counterfactual: {counterfactual}")

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

    # Load dataset - both train and test splits
    train_dataset = load_dataset("fancyzhx/ag_news", split="train")
    test_dataset = load_dataset("fancyzhx/ag_news", split="test")

    label_names = train_dataset.features["label"].names
    sports_label = label_names.index("Sports")

    # Filter training dataset for sports articles (we still train on sports only)
    sports_train_dataset = train_dataset.filter(lambda x: x["label"] == sports_label)

    # Sample the training dataset if sample_size is specified
    if sample_size > 0:
        if sample_size > len(sports_train_dataset):
            print(
                f"Warning: Sample size {sample_size} is larger than training dataset size {len(sports_train_dataset)}. Using full dataset."
            )
        else:
            sports_train_dataset = sports_train_dataset.select(range(sample_size))
            print(f"Using {sample_size} samples from the training dataset.")
    else:
        print(f"Using full training dataset with {len(sports_train_dataset)} samples.")

    # Sample the test dataset if needed
    test_sample_size = (
        min(args.test_sample_size, len(test_dataset))
        if args.test_sample_size > 0
        else len(test_dataset)
    )
    if test_sample_size < len(test_dataset):
        test_dataset = test_dataset.select(range(test_sample_size))
    print(
        f"Using {test_sample_size} samples from the entire test dataset (all categories)."
    )

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForMaskedLM.from_pretrained(model_name, token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.train()

    def tokenize_function(example):
        tokens = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        tokens["labels"] = tokens["input_ids"].copy()  # Needed for MLM
        return tokens

    tokenized_train_dataset = sports_train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=f"/tmp/bert_run_{job_number}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        report_to=[],
        save_total_limit=0,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
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
    )

    # Total failed count
    failed_count = failed_count_before + failed_count_after

    if results:
        mode = (
            "full" if sample_size == 0 else "test" if sample_size < 100 else "sampled"
        )
        df = pd.DataFrame(results)

        # Add label names to the results
        df["label_name"] = df["label"].apply(lambda x: label_names[x])

        # Add normalization indicator to filename
        normalize_suffix = "_normalized" if normalize else ""
        # Add counterfactual mode to filename if it's not the default
        counterfactual_suffix = (
            f"_cf-{counterfactual}" if counterfactual != "identity" else ""
        )

        df.to_pickle(
            f"data/{mode}/bert_results_{job_number}{normalize_suffix}{counterfactual_suffix}.pkl"
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
        description="BERT fine-tuning and uncertainty measurement"
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
        choices=["identity", "constant"],
        help="How to choose labels for bert_gradient: 'identity' uses the same input, 'constant' uses the unknown token (default: identity)",
    )

    args = parser.parse_args()

    main(args)
