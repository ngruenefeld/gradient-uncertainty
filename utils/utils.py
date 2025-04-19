from datasets import load_dataset, DatasetDict, Dataset
import torch
import gc


def get_response(prompt, model, tokenizer, device):
    try:
        model.eval()

        # Use no_grad to prevent gradient storage during inference
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_text.startswith(prompt):
            completion = generated_text[len(prompt) :].strip()
        else:
            completion = generated_text

        # Free memory
        del inputs, outputs
        torch.cuda.empty_cache()

        return completion
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        torch.cuda.empty_cache()
        return {"error": str(e)}


def arithmetic_mean_change(param_values, param_grads):
    new_param_values = param_values + param_grads
    denominator = 0.5 * (param_values + new_param_values)

    # Small epsilon to avoid division by zero
    epsilon = 1e-8

    # Calculate symmetric percent change
    normalized_grads = torch.where(
        denominator.abs() > epsilon,
        param_grads / denominator,
        torch.zeros_like(param_grads),
    )

    return normalized_grads


def completion_gradient(
    prompt, completion, model, tokenizer, device, response_only=True, normalize=False
):
    try:
        model.train()

        full_text = prompt + completion

        # Get the encodings for both prompt and full text
        full_encodings = tokenizer(full_text, return_tensors="pt")
        input_ids = full_encodings.input_ids.to(device)

        prompt_encodings = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encodings.input_ids.shape[1]

        # Normal processing
        labels = input_ids.clone()
        if response_only:
            labels[0, :prompt_len] = -100  # Ignore loss for prompt tokens

        # Calculate loss and backprop
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        model.zero_grad()
        loss.backward()

        # Calculate gradient norm
        num_params = sum(p.numel() for _, p in model.named_parameters())
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if normalize == False:
                    param_norm = param.grad.detach().norm(2)
                else:
                    param_values = param.detach()
                    param_grads = param.grad.detach()

                    normalized_grads = arithmetic_mean_change(param_values, param_grads)
                    param_norm = normalized_grads.detach().norm(2)

                total_norm += param_norm.item() ** 2

        uncertainty = torch.tensor(total_norm**0.5)
        print("8", uncertainty)

        # Calculate completion length
        completion_length = (
            full_encodings.input_ids.shape[1] - prompt_encodings.input_ids.shape[1]
        )

        # Free memory before returning
        del outputs, loss, input_ids, labels, full_encodings, prompt_encodings
        gc.collect()
        torch.cuda.empty_cache()

        return uncertainty, completion_length
    except Exception as e:
        print(f"Error in completion_gradient: {str(e)}")
        # Make sure to free memory
        torch.cuda.empty_cache()
        gc.collect()
        return {"error": str(e)}


def bert_gradient(inputs, labels, model, normalize=False):
    try:
        model.train()

        model.zero_grad()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()

        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if normalize == False:
                    param_norm = param.grad.detach().norm(2)
                else:
                    param_values = param.detach()
                    param_grads = param.grad.detach()

                    normalized_grads = arithmetic_mean_change(param_values, param_grads)
                    param_norm = normalized_grads.detach().norm(2)

                total_norm += param_norm.item() ** 2

        uncertainty = torch.tensor(total_norm**0.5)

        del outputs, loss, inputs, labels
        gc.collect()
        torch.cuda.empty_cache()

        return uncertainty
    except Exception as e:
        print(f"Error in bert_gradient: {str(e)}")
        # Make sure to free memory
        torch.cuda.empty_cache()
        gc.collect()
        return {"error": str(e)}


def load_bert_dataset_dicts(choice="ag_news"):
    if choice == "ag_news":
        dataset = load_dataset("fancyzhx/ag_news")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        label_names = train_dataset.features["label"].names

        train_labels = [label_names[label] for label in train_dataset["label"]]
        test_labels = [label_names[label] for label in test_dataset["label"]]

        train_data = {
            "text": train_dataset["text"],
            "origin": ["ag_news"] * len(train_dataset["text"]),
            "label": train_labels,
        }

        test_data = {
            "text": test_dataset["text"],
            "origin": ["ag_news"] * len(test_dataset["text"]),
            "label": test_labels,
        }

        return train_data, test_data

    elif choice == "pubmed":
        dataset = load_dataset("MedRAG/pubmed", streaming=True)["train"]

        sample_size = 10000

        content_samples = []
        count = 0

        for example in dataset:
            if count >= sample_size:
                break
            content_samples.append(example["content"])
            count += 1

        data = {
            "text": content_samples,
            "origin": ["pubmed"] * len(content_samples),
            "label": ["Medicine"] * len(content_samples),
        }

        return data

    if choice == "mmlu":
        cs_dataset = load_dataset("tasksource/mmlu", "computer_security")
        phil_dataset = load_dataset("tasksource/mmlu", "philosophy")

        cs_dataset_val = cs_dataset["validation"]
        cs_dataset_test = cs_dataset["validation"]
        phil_dataset = phil_dataset["test"]

        cs_data_val = {
            "text": cs_dataset_val["question"],
            "origin": ["mmlu"] * len(cs_dataset_val["question"]),
            "label": ["Computer Security"] * len(cs_dataset_val["question"]),
        }

        cs_data_test = {
            "text": cs_dataset_test["question"],
            "origin": ["mmlu"] * len(cs_dataset_test["question"]),
            "label": ["Computer Security"] * len(cs_dataset_test["question"]),
        }

        phil_data = {
            "text": phil_dataset["question"],
            "origin": ["mmlu"] * len(phil_dataset["question"]),
            "label": ["Philosophy"] * len(phil_dataset["question"]),
        }

        return cs_data_val, cs_data_test, phil_data

    else:
        raise ValueError(f"Dataset {choice} not supported.")


def load_bert_datasets(choice="ag_news"):
    if choice == "ag_news":
        train_data, test_data = load_bert_dataset_dicts("ag_news")

        indices_to_remove = {
            i for i, v in enumerate(train_data["label"]) if v != "Sports"
        }

        filtered_train_data = {
            key: [v for i, v in enumerate(vals) if i not in indices_to_remove]
            for key, vals in train_data.items()
        }

        return Dataset.from_dict(filtered_train_data), Dataset.from_dict(test_data)

    elif choice == "ag-pubmed":
        ag_train_data, ag_test_data = load_bert_dataset_dicts("ag_news")
        pubmed_data = load_bert_dataset_dicts("pubmed")

        combined_test = {
            "text": ag_test_data["text"] + pubmed_data["text"],
            "origin": ag_test_data["origin"] + pubmed_data["origin"],
            "label": ag_test_data["label"] + pubmed_data["label"],
        }

        return Dataset.from_dict(ag_train_data), Dataset.from_dict(combined_test)

    elif choice == "mmlu":
        cs_data_val, cs_data_test, phil_data = load_bert_dataset_dicts("mmlu")

        combined_test = {
            "text": cs_data_test["text"] + phil_data["text"],
            "origin": cs_data_test["origin"] + phil_data["origin"],
            "label": cs_data_test["label"] + phil_data["label"],
        }

        return Dataset.from_dict(cs_data_val), Dataset.from_dict(combined_test)

    else:
        raise ValueError(f"Dataset {choice} not supported.")

        raise ValueError(f"Dataset {choice} not supported.")
