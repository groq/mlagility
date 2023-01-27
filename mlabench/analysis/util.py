import os
import sys
import shutil
from dataclasses import dataclass
from typing import Callable, List, Union
import inspect
import torch
import groqflow.justgroqit.export as export
import groqflow.justgroqit.stage as stage
from groqflow.common import printing
import groqflow.common.build as build


@dataclass
class ModelInfo:
    model: torch.nn.Module
    name: str
    file: str = ""
    line: int = 0
    params: int = 0
    depth: int = 0
    hash: Union[str, None] = None
    parent_hash: Union[str, None] = None
    inputs: Union[dict, None] = None
    executed: int = 0
    exec_time: float = 0.0
    old_forward: Union[Callable, None] = None
    status_message: str = ""
    status_message_color: printing.Colors = printing.Colors.ENDC
    is_target: bool = False
    check_ops: bool = False
    build_model: bool = False
    model_type: build.ModelType = build.ModelType.PYTORCH

    def __post_init__(self):
        self.params = count_parameters(self.model, self.model_type)


check_ops_pytorch = stage.Sequence(
    "default_pytorch_check_op_sequence",
    "Checking Ops For PyTorch Model",
    [
        export.ExportPytorchModel(),
        export.OptimizeOnnxModel(),
        export.CheckOnnxCompatibility(),
    ],
    enable_model_validation=True,
)

check_ops_keras = stage.Sequence(
    "default_keras_check_op_sequence",
    "Checking Ops For Keras Model",
    [
        export.ExportKerasModel(),
        export.OptimizeOnnxModel(),
        export.CheckOnnxCompatibility(),
    ],
    enable_model_validation=True,
)


def clean_output_dir(output_dir: str = "~/.cache/groqflow") -> None:
    """
    Delete all elements of the output directory that are not text files
    """
    output_dir = os.path.expanduser(output_dir)
    ext_list = [".txt", ".out", ".yaml", ".json"]
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path) and not any(
            [ext in file_path for ext in ext_list]
        ):
            os.remove(file_path)
        elif file_path == os.path.join(output_dir, "compile"):
            clean_output_dir(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def count_parameters(model: torch.nn.Module, model_type: build.ModelType) -> int:
    """
    Returns the number of parameters of a given model
    """
    if model_type == build.ModelType.PYTORCH:
        total_params = 0
        for _, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_params += params
    elif build.ModelType.KERAS:
        total_params = model.count_params()
    return total_params


def stop_stdout_forward() -> None:
    """
    Stop forwarding stdout to file
    """
    if hasattr(sys.stdout, "terminal"):
        sys.stdout = sys.stdout.terminal


def get_classes(module) -> List[str]:
    """
    Returns all classes within a module
    """
    return [y for x, y in inspect.getmembers(module, inspect.isclass)]
