import argparse
import torch
from mlagility import benchmark_model

torch.manual_seed(0)

# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 1000
output_size = 500
pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Benchmark a PyTorch model on a specified device and backend."
    )

    # Add the arguments
    parser.add_argument(
        "--device",
        type=str,
        choices=["x86", "nvidia"],
        default="x86",
        help="The device to benchmark on (x86 or nvidia)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "remote"],
        default="local",
        help="The backend to use (local or remote)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Instantiate model and generate inputs
    torch.manual_seed(0)
    input_size = 1000
    output_size = 500
    pytorch_model = SmallModel(input_size, output_size)
    inputs = {"x": torch.rand(input_size)}

    # Benchmark the model on the specified device and backend
    print(f"Benchmarking on {args.device} {args.backend}...")
    benchmark_model(pytorch_model, inputs, device=args.device, backend=args.backend)


if __name__ == "__main__":
    main()
