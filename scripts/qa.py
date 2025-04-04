import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

from openai import OpenAI
import os

import pandas as pd

from utils.utils import get_response, completion_gradient
from utils.gpt import evaluate_answers, rephrase_text


with open(os.path.expanduser(".api_key"), "r") as f:
    api_key = f.read().strip()

oai_client = OpenAI(api_key=api_key)
gpt_model = "o3-mini-2025-01-31"


# model_name = "TheBloke/Llama-2-7B-Chat-AWQ"
# model_name = "QuantFactory/NVIDIA-Llama3-ChatQA-1.5-8B-GGUF"
model_name = "gpt2"

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

results = []

# for i in tqdm(range(len(data))):
for i in random.sample(range(len(data)), 1):
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

    completion = get_response(prompt, model, tokenizer, device)

    # print(f"Question: {prompt}")
    # print()
    # print(f"Completion: {completion}")
    # print()
    # print("Correct Answers")
    # [print(a) for a in answers]
    # print()

    evaluation = evaluate_answers(prompt, completion, answers, oai_client, gpt_model)
    # print(evaluation["is_correct"])

    gradient = completion_gradient(prompt, completion, model, tokenizer, device)
    # print()
    # print("Gradient")
    # print(gradient)

    rephrasings = rephrase_text(completion, oai_client, gpt_model)["rephrasings"]
    # print()
    # print("Rephrasings")

    rephrasing_gradients = []
    rephrasing_lengths = []
    rephrasing_gradient_norms = []

    for phrasing in rephrasings:
        rephrasing_gradient, rephrasing_length = completion_gradient(
            prompt, phrasing, model, tokenizer, device
        )
        rephrasing_gradients.append(rephrasing_gradient)
        rephrasing_lengths.append(rephrasing_length)
        rephrasing_gradient_norms.append(torch.norm(rephrasing_gradient))
        # print(phrasing)
        # print(rephrasing_gradient)

    rephrasing_gradient_std = torch.sum(
        torch.std(torch.stack(rephrasing_gradients), dim=0)
    )

    results.append(
        {
            "question": prompt,
            "completion": completion,
            "correct_answers": answers,
            "evaluation": evaluation["is_correct"],
            "completion_gradient": gradient,
            "rephrased_completions": rephrasings,
            "rephrased_completion_lengths": rephrasing_lengths,
            "rephrased_gradients": rephrasing_gradients,
            "rephrased_gradient_std": rephrasing_gradient_std,
        }
    )

    # print()
    # print("--------------------------------")

df = pd.DataFrame(results)
df.to_csv("../data/results_test.csv", index=False)
