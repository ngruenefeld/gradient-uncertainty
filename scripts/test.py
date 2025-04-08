from utils.utils import get_response
import torch


def main():
    print("Test")
    print(get_response)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


if __name__ == "__main__":
    main()
