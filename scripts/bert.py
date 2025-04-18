import os

from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    TrainingArguments,
    Trainer,
)
import torch

hf_token = os.getenv("HF_TOKEN")

dataset = load_dataset("fancyzhx/ag_news", split="train")

sports_dataset = dataset.filter(lambda x: x["label"] == 1)

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name, token=hf_token)
model = BertForMaskedLM.from_pretrained(model_name, token=hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)


def tokenize_function(example):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=128
    ).to(device)


tokenized_dataset = sports_dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="/tmp/ignore_this",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
