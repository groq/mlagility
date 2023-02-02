import argparse
from dataclasses import dataclass
from typing import TypeVar, List


@dataclass
class Arg:
    """
    Object for tracking command-line arg configurations.
    """

    name: str
    default: int
    type: TypeVar
    nargs: str

    def add_arg(self, parser_obj):
        parser_obj.add_argument(
            "--" + self.name, default=self.default, type=self.type, nargs=self.nargs
        )


def parse(valid_args: List):

    # List valid ml-agility args
    mla_args = {
        "batch_size": Arg("batch_size", default=1, type=int, nargs="?"),
        "max_seq_length": Arg("max_seq_length", default=128, type=int, nargs="?"),
        "max_audio_seq_length": Arg(
            "max_audio_seq_length", default=25600, type=int, nargs="?"
        ),
        "height": Arg("height", default=224, type=int, nargs="?"),
        "num_channels": Arg("num_channels", default=3, type=int, nargs="?"),
        "width": Arg("width", default=224, type=int, nargs="?"),
        # For Graph Neural Networks
        "k": Arg("k", default=8, type=int, nargs="?"),
        "alpha": Arg("alpha", default=2.2, type=float, nargs="?"),
        "out_channels": Arg("out_channels", default=16, type=int, nargs="?"),
        "num_layers": Arg("num_layers", default=8, type=int, nargs="?"),
        "in_channels": Arg("in_channels", default=1433, type=int, nargs="?"),
    }

    # Create parser that accepts only the args received as part of valid_args
    parser = argparse.ArgumentParser()
    for arg_name in valid_args:
        if arg_name not in mla_args.keys():
            raise KeyError(
                f"{arg_name} is not a valid arg. Valid args are {str(mla_args.keys())}."
            )
        mla_args[arg_name].add_arg(parser)

    # Parse arg and return args as a list
    args = vars(parser.parse_args())
    parsed_args = [args[arg] for arg in valid_args]
    return parsed_args
