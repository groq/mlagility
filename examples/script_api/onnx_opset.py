"""
Example script that demonstrates how to set a custom ONNX opset for a benchmarking run

You can run this script in your mlagility Conda environment with:
    python onnx_opset.py --onnx-opset YOUR_OPSET

And then you can observe the ONNX opset in the resulting ONNX files with:
   benchit cache stats BUILD_NAME
"""

import pathlib
import argparse
from mlagility import benchmark_script


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Benchmark a PyTorch model with a specified ONNX opset."
    )

    # Add the arguments
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=16,
        help="ONNX opset to use when creating ONNX files",
    )

    # Parse the arguments
    args = parser.parse_args()

    path_to_hello_world_script = str(
        pathlib.Path(__file__).parent.resolve() / "scripts" / "hello_world.py"
    )

    benchmark_script(
        input_scripts=[path_to_hello_world_script], onnx_opset=args.onnx_opset
    )


if __name__ == "__main__":
    main()
