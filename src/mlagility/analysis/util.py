import os
import sys
import shutil
from dataclasses import dataclass
import glob
from typing import Callable, List, Union
import inspect
import torch
import groqflow.justgroqit.export as export
import groqflow.justgroqit.stage as stage
from groqflow.common import printing
import groqflow.common.build as build
import mlagility.common.filesystem as filesystem
from mlagility.api.performance import MeasuredPerformance


@dataclass
class ModelInfo:
    model: torch.nn.Module
    name: str
    script_name: str
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
    traceback_message_color: printing.Colors = printing.Colors.FAIL
    is_target: bool = False
    build_model: bool = False
    model_type: build.ModelType = build.ModelType.PYTORCH
    performance: MeasuredPerformance = None
    traceback: List[str] = None

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


def clean_output_dir(output_dir: str = filesystem.DEFAULT_CACHE_DIR) -> None:
    """
    Delete all elements of the output directory that are not human readable
    """
    output_dir = os.path.expanduser(output_dir)

    # Remove files that do not have an allowed extension
    allowed_extensions = (".txt", ".out", ".yaml", ".json")
    all_paths = glob.glob(f"{output_dir}/**/*", recursive=True)
    for path in all_paths:
        if os.path.isfile(path) and not path.endswith(allowed_extensions):
            os.remove(path)

    # Remove all empty folders
    for path in all_paths:
        if os.path.isdir(path):
            if len(os.listdir(path)) == 0:
                shutil.rmtree(path)


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
