import argparse
from utils.utils import get_response


def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--name", type=str, help="Your name", required=True)
    parser.add_argument("--age", type=int, help="Your age", required=False)

    args = parser.parse_args()

    print(f"Name: {args.name}")
    if args.age:
        print(f"Age: {args.age}")

    # Example usage of get_response
    print(get_response)


if __name__ == "__main__":
    main()
