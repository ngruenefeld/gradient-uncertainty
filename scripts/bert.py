import argparse
import os

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


def main(args):
    key_mode = args.key_mode
    sample_size = args.sample_size
    job_number = args.job_number

    print(f"Job number: {job_number}")
    print(f"Key mode: {key_mode}")
    print(f"Sample size: {sample_size}")

    # Handle API token based on key mode
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

    # Load dataset
    dataset = load_dataset("fancyzhx/ag_news", split="train")

    label_names = dataset.features["label"].names
    sports_label = label_names.index("Sports")

    sports_dataset = dataset.filter(lambda x: x["label"] == sports_label)

    # Sample the dataset if sample_size is specified
    if sample_size > 0:
        if sample_size > len(sports_dataset):
            print(
                f"Warning: Sample size {sample_size} is larger than dataset size {len(sports_dataset)}. Using full dataset."
            )
        else:
            sports_dataset = sports_dataset.select(range(sample_size))
            print(f"Using {sample_size} samples from the dataset.")
    else:
        print(f"Using full dataset with {len(sports_dataset)} samples.")

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

    tokenized_dataset = sports_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]
    ).select(range(min(10, len(sports_dataset))))

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
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    in_domain_sentence = "The basketball game was intense and exciting."

    in_domain_uncertainty_before = bert_gradient(
        in_domain_sentence, in_domain_sentence, model, tokenizer, device
    )
    print("Uncertainty before fine-tuning:", in_domain_uncertainty_before)

    trainer.train()

    in_domain_uncertainty_after = bert_gradient(
        in_domain_sentence, in_domain_sentence, model, tokenizer, device
    )
    print("Uncertainty after fine-tuning:", in_domain_uncertainty_after)


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
        help="Number of examples to sample from the dataset (0 = use full dataset)",
    )

    args = parser.parse_args()

    main(args)
