import torch


def get_response(prompt, model, tokenizer, device):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt) :].strip()
    else:
        completion = generated_text

    return completion


def completion_gradient(prompt, completion, model, tokenizer, device):
    model.train()

    full_text = prompt + completion

    full_encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = full_encodings.input_ids.to(device)

    prompt_encodings = tokenizer(prompt, return_tensors="pt")
    prompt_len = prompt_encodings.input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    model.zero_grad()
    loss.backward()

    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads.append(param.grad.flatten())

    # uncertainty = torch.norm(torch.cat(grads))
    # return uncertainty.cpu().item()

    uncertainty = torch.cat(grads)
    return (
        uncertainty.cpu(),
        full_encodings.input_ids.shape[1] - prompt_encodings.input_ids.shape[1],
    )
