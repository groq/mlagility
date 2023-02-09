import argparse
from dataclasses import dataclass
from typing import TypeVar, List, Union


@dataclass
class Arg:
    """
    Class for tracking command-line arg configurations
    """

    name: str
    default: int
    type: TypeVar

    def add_arg(self, parser_obj):
        parser_obj.add_argument(
            "--" + self.name, default=self.default, type=self.type, nargs="?"
        )


def parse(valid_args: List[str]) -> List[Union[int, float]]:
    """
    The parse() function receives a list of strings with the desired arguments and
    returns a list containing the values of those arguments.

    The parse function uses argparse, allowing users to parse arguments through the CLI
    Using any argument that is not part of the list of valid args will result in a ValueError.
    """
    # List valid mlagility args
    mla_args = {
        # General args
        # Batch size for the input to the model that will be used for benchmarking
        "batch_size": Arg("batch_size", default=1, type=int),
        # Maximum sequence length for the model's input;
        # also the input sequence length that will be used for benchmarking
        "max_seq_length": Arg("max_seq_length", default=128, type=int),
        # Maximum sequence length for the model's audio input;
        # also the input sequence length that will be used for benchmarking
        "max_audio_seq_length": Arg("max_audio_seq_length", default=25600, type=int),
        # Height of the input image that will be used for benchmarking
        "height": Arg("height", default=224, type=int),
        # Number of channels in the input image that will be used for benchmarking
        "num_channels": Arg("num_channels", default=3, type=int),
        # Width of the input image that will be used for benchmarking
        "width": Arg("width", default=224, type=int),
        # Args for Graph Neural Networks
        "k": Arg("k", default=8, type=int),
        "alpha": Arg("alpha", default=2.2, type=float),
        "out_channels": Arg("out_channels", default=16, type=int),
        "num_layers": Arg("num_layers", default=8, type=int),
        "in_channels": Arg("in_channels", default=1433, type=int),
    }

    # Create parser that accepts only the args received as part of valid_args
    parser = argparse.ArgumentParser()
    for arg_name in valid_args:
        if arg_name not in mla_args.keys():
            raise KeyError(
                f"{arg_name} is not a valid mlagility arg. Valid args are {str(mla_args.keys())}."
            )
        mla_args[arg_name].add_arg(parser)

    # Parse arg and return args as a list
    args = vars(parser.parse_args())
    parsed_args = [args[arg] for arg in valid_args]
    return parsed_args
