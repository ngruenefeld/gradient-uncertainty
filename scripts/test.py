from utils.utils import get_response
import torch


def main():
    print("Test")
    print(get_response)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved

        print("Total memory:", t)
        print("Reserved memory:", r)
        print("Allocated memory:", a)
        print("Free memory:", f)


if __name__ == "__main__":
    main()
