import os

from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

hf_token = os.getenv("HF_TOKEN")

dataset = load_dataset("m-a-p/FineFineWeb", split="train", streaming=True)

medical_stream = dataset.filter(lambda x: x["domain"] == "medicine")

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name, token=hf_token)
model = BertForMaskedLM.from_pretrained(model_name, token=hf_token)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_special_tokens_mask=True,
    )


tokenized_stream = medical_stream.map(tokenize)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    max_steps=1,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_stream,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
