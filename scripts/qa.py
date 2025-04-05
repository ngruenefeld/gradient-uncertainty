import argparse
import os
import random

import pandas as pd
import torch
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.gpt import rephrase_text, evaluate_answers
from utils.utils import get_response, completion_gradient


def main(args):
    job_number = args.job_number
    dataset_name = args.dataset
    model_name = args.model
    gpt_model = args.gpt_model
    key_mode = args.key_mode
    sample_size = args.sample_size

    if model_name == "gpt2":
        model_path = "gpt2"
    elif model_name == "llama-awq":
        model_path = "TheBloke/Llama-2-7B-Chat-AWQ"

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

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if dataset_name == "truthful":
        dataset = load_dataset("truthfulqa/truthful_qa", "generation")
        data = dataset["validation"]
    elif dataset_name == "natural":
        dataset = load_dataset("google-research-datasets/natural_questions", "dev")
        data = dataset["validation"]
    elif dataset_name == "trivia":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
        data = dataset["validation"]

    results = []

    if sample_size > 0:
        indices = random.sample(range(len(data)), sample_size)
    else:
        indices = range(len(data))

    for i in indices:
        if dataset_name == "natural":
            prompt = data[i]["question"]["text"]
            answers = [
                text
                for sublist in [
                    a["text"] for a in data[i]["annotations"]["short_answers"]
                ]
                for text in sublist
            ]
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
        if "error" in evaluation:
            print(
                f"Skipping sample {i} due to evaluate_answers error: {evaluation['error']}"
            )
            continue

        gradient, completion_length = completion_gradient(
            prompt, completion, model, tokenizer, device
        )
        gradient = torch.norm(gradient).item()

        rephrasings_result = rephrase_text(completion, oai_client, gpt_model)
        if "error" in rephrasings_result:
            print(
                f"Skipping sample {i} due to rephrase_text error: {rephrasings_result['error']}"
            )
            continue

        rephrasings = rephrasings_result["rephrasings"]

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
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Set if you want to sample a specific number of examples from the dataset",
    )

    args = parser.parse_args()

    main(args)
