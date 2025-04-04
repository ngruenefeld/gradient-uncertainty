import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

from openai import OpenAI
import os

import pandas as pd

from utils.utils import get_response, completion_gradient
from utils.gpt import evaluate_answers, rephrase_text


def main(args):
    job_number = args.job_number
    dataset_name = args.dataset
    model_name = args.model
    gpt_model = args.gpt_model
    key_mode = args.key_mode

    if key_mode == "keyfile":
        with open(os.path.expanduser(".api_key"), "r") as f:
            api_key = f.read().strip()
    elif key_mode == "env":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not found. Please set the OPENAI_API_KEY environment variable."
            )

    oai_client = OpenAI(api_key=api_key)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if dataset_name == "truthful":
        dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    elif dataset_name == "natural":
        dataset = load_dataset("google-research-datasets/natural_questions", "dev")
    elif dataset_name == "trivia":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")

    data = dataset["validation"]

    results = []

    for i in range(len(data)):
        if dataset_name == "natural":
            prompt = data[i]["question"]["text"]
            answers = [a["text"] for a in data[i]["annotations"]["short_answers"]]
        elif dataset_name == "truthful":
            prompt = data[i]["question"]
            answers = data[i]["correct_answers"]
        elif dataset_name == "trivia":
            prompt = data[i]["question"]
            answers = data[i]["answer"]["aliases"]

        completion = get_response(prompt, model, tokenizer, device)

        evaluation = evaluate_answers(
            prompt, completion, answers, oai_client, gpt_model
        )

        gradient, completion_length = completion_gradient(
            prompt, completion, model, tokenizer, device
        )
        gradient = torch.norm(gradient).item()

        rephrasings = rephrase_text(completion, oai_client, gpt_model)["rephrasings"]

        rephrasing_gradients = []
        rephrasing_lengths = []
        rephrasing_gradient_norms = []

        for phrasing in rephrasings:
            rephrasing_gradient, rephrasing_length = completion_gradient(
                prompt, phrasing, model, tokenizer, device
            )
            rephrasing_gradients.append(rephrasing_gradient)
            rephrasing_lengths.append(rephrasing_length)
            rephrasing_gradient_norms.append(torch.norm(rephrasing_gradient).item())

        rephrasing_gradient_std = torch.sum(
            torch.std(torch.stack(rephrasing_gradients), dim=0)
        ).item()

        results.append(
            {
                "question": prompt,
                "completion": completion,
                "completion_length": completion_length,
                "correct_answers": answers,
                "evaluation": evaluation["is_correct"],
                "completion_gradient": gradient,
                "rephrased_completions": rephrasings,
                "rephrased_completion_lengths": rephrasing_lengths,
                "rephrased_gradients": rephrasing_gradient_norms,
                "rephrased_gradient_std": rephrasing_gradient_std,
            }
        )
        break

    df = pd.DataFrame(results)
    df.to_pickle(f"data/results_{job_number}_{model_name}_{dataset_name}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument("job_number")
    parser.add_argument(
        "--dataset",
        type=str,
        default="truthful",
        help="Dataset to use: truthful, natural, trivia",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model to use, including the path to the model",
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="o3-mini-2025-01-31",
        help="GPT model to use for OpenAI API",
    )
    parser.add_argument(
        "--key_mode",
        type=str,
        default="keyfile",
        help="Whether to read the OpenAI API key from a file or use an environment variable",
    )

    args = parser.parse_args()

    main(args)
