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

hf_token = os.getenv("HF_TOKEN")

dataset = load_dataset("fancyzhx/ag_news", split="train")

label_names = dataset.features["label"].names
sports_label = label_names.index("Sports")

sports_dataset = dataset.filter(lambda x: x["label"] == sports_label)

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForMaskedLM.from_pretrained(model_name, token=hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

print(device)


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
).select(range(10))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)


training_args = TrainingArguments(
    output_dir="/tmp/no_save",
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
