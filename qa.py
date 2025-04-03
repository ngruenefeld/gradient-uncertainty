import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

import openai
from openai import OpenAI
import json
import os

import pandas as pd

with open(os.path.expanduser(".api_key"), "r") as f:
    api_key = f.read().strip()

oai_client = OpenAI(api_key=api_key)
gpt_model = "o3-mini-2025-01-31"


def rephrase_text(text_to_rephrase, client, model):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant that paraphrases text.",
            },
            {
                "role": "user",
                "content": f"Generate three paraphrases of the following text: {text_to_rephrase}",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "rephrasings_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "original_input": {
                            "type": "string",
                            "description": "The original input that needs to be rephrased.",
                        },
                        "rephrasings": {
                            "type": "array",
                            "description": "A list of rephrased versions of the original input.",
                            "items": {
                                "type": "string",
                                "description": "A single rephrased sentence.",
                            },
                        },
                    },
                    "required": ["original_input", "rephrasings"],
                    "additionalProperties": False,
                },
            }
        },
    )
    event = json.loads(response.output_text)
    return event


def evaluate_answers(question, answer, reference_answers, client, model):
    reference_answers_formatted = "\n".join(reference_answers)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates question-answer pairs.",
            },
            {
                "role": "user",
                "content": f"Your task is to evaluate whether a generated answer candidate to a given question is correct or not, given a set of correct reference answers.\nThe question is: {question}\nThe generated answer candidate is: {answer}\nThe correct reference answers are:\n{reference_answers_formatted}\nIs the answer candidate correct or not? The answer might be overly verbose, try to extract what is meant.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "answer_verification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question posed to which the answer is to be verified.",
                        },
                        "given_answer": {
                            "type": "string",
                            "description": "The answer provided that needs verification.",
                        },
                        "correct_answers": {
                            "type": "array",
                            "description": "A list of correct answers against which the given answer is verified.",
                            "items": {
                                "type": "string",
                                "description": "A correct reference answer.",
                            },
                        },
                        "is_correct": {
                            "type": "boolean",
                            "description": "Indicates whether the given answer matches any of the correct answers.",
                        },
                    },
                    "required": [
                        "question",
                        "given_answer",
                        "correct_answers",
                        "is_correct",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    )
    event = json.loads(response.output_text)
    return event


model_name = "TheBloke/Llama-2-7B-Chat-AWQ"
# model_name = "QuantFactory/NVIDIA-Llama3-ChatQA-1.5-8B-GGUF"
# model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

d = "trivia"
if d == "truthful":
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
elif d == "natural":
    dataset = load_dataset("google-research-datasets/natural_questions", "dev")
elif d == "trivia":
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")

data = dataset["validation"]

# for i in random.sample(range(len(data)), 5):
#     if d == "natural":
#         print(data[i]['question']['text'])
#         [print(a['text']) for a in data[i]['annotations']['short_answers']]
#         print(data[i]['annotations']['yes_no_answer'])
#         print()
#     elif d == "truthful":
#         print(data[i]['question'])
#         [print(a) for a in data[i]['correct_answers']]
#         print()
#     elif d == "trivia":
#         print(data[i]['question'])
#         [print(a) for a in data[i]['answer']['aliases']]
#         print()


def get_response(prompt, model, tokenizer):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt) :].strip()
    else:
        completion = generated_text

    return completion


def completion_gradient(prompt, completion, model, tokenizer):
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

    uncertainty = torch.norm(torch.cat(grads))

    return uncertainty.cpu().item()


results = []

# for i in tqdm(range(len(data))):
for i in random.sample(range(len(data)), 10):
    # print()
    if d == "natural":
        prompt = data[i]["question"]["text"]
        answers = [a["text"] for a in data[i]["annotations"]["short_answers"]]
    elif d == "truthful":
        prompt = data[i]["question"]
        answers = data[i]["correct_answers"]
    elif d == "trivia":
        prompt = data[i]["question"]
        answers = data[i]["answer"]["aliases"]

    completion = get_response(prompt, model, tokenizer)

    # print(f"Question: {prompt}")
    # print()
    # print(f"Completion: {completion}")
    # print()
    # print("Correct Answers")
    # [print(a) for a in answers]
    # print()

    evaluation = evaluate_answers(prompt, completion, answers, oai_client, gpt_model)
    # print(evaluation["is_correct"])

    gradient = completion_gradient(prompt, completion, model, tokenizer)
    # print()
    # print("Gradient")
    # print(gradient)

    rephrasings = rephrase_text(completion, oai_client, gpt_model)["rephrasings"]
    # print()
    # print("Rephrasings")

    rephrasing_gradients = []

    for phrasing in rephrasings:
        rephrasing_gradient = completion_gradient(prompt, phrasing, model, tokenizer)
        rephrasing_gradients.append(rephrasing_gradient)
        # print(phrasing)
        # print(rephrasing_gradient)

    results.append(
        {
            "question": prompt,
            "completion": completion,
            "correct_answers": answers,
            "evaluation": evaluation["is_correct"],
            "completion_gradient": gradient,
            "rephrased_completions": rephrasings,
            "rephrased_gradients": rephrasing_gradients,
        }
    )

    # print()
    # print("--------------------------------")

df = pd.DataFrame(results)
df.to_csv("data/results.csv", index=False)
