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

                    # Using Symmetric Percentage Change
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


def bert_gradient(sample, labels, model, tokenizer, device, normalize=False):
    # try:
    model.train()

    inputs = tokenizer(sample, return_tensors="pt").to(device)
    labels = inputs.input_ids.clone().to(device)

    print(inputs)
    print(inputs.input_ids.shape)
    print(labels.shape)

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

                # Using Symmetric Percentage Change
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

                param_norm = normalized_grads.detach().norm(2)
            total_norm += param_norm.item() ** 2

    uncertainty = torch.tensor(total_norm**0.5)

    del outputs, loss, input_ids, labels
    gc.collect()
    torch.cuda.empty_cache()

    return uncertainty
    # except Exception as e:
    #     print(f"Error in bert_gradient: {str(e)}")
    #     # Make sure to free memory
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     return {"error": str(e)}
