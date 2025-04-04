import argparse
from utils.utils import get_response


def main(args):
    args = parser.parse_args()

    print(f"Job Number: {args.job_number}")
    print(f"Dataset: {args.dataset}")

    # Example usage of get_response
    print(get_response)


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
