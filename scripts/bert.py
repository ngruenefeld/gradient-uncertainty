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


def get_embedding(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
        device
    )
    print("Input Shape", inputs.input_ids.shape)
    with torch.no_grad():
        outputs = model.bert(**inputs)
        last_hidden = outputs.last_hidden_state
        print("Last Hidden Shape", last_hidden.shape)
        pooled = last_hidden.mean(dim=-1)
        print("Pooled Shape", pooled.shape)
    return pooled.squeeze()


sentence = "The basketball game was intense and exciting."

embedding_before = get_embedding(sentence, model)
print("Embedding before fine-tuning:", embedding_before[:5])


# trainer.train()


embedding_after = get_embedding(sentence, model)
print("Embedding after fine-tuning:", embedding_after[:5])


inputs = tokenizer("This is a test", return_tensors="pt")
print("Input Shape", inputs.input_ids.shape)
print(inputs.input_ids)
labels = torch.ones([1, 6], dtype=torch.long)

outputs = model(**inputs, labels=labels)
loss = outputs.loss

loss.backward()

grads = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grads.append(param.grad.flatten())

uncertainty = torch.norm(torch.cat(grads))

print(uncertainty)
