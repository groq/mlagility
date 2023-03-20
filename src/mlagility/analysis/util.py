import sys
import re
from dataclasses import dataclass
from typing import Callable, List, Union, Dict
import inspect
import torch
import onnx
import groqflow.justgroqit.export as export
import groqflow.justgroqit.stage as stage
from groqflow.common import printing
import groqflow.common.build as build
from mlagility.api.performance import MeasuredPerformance


class AnalysisException(Exception):
    """
    Indicates a failure during analysis
    """


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


def count_parameters(model: torch.nn.Module, model_type: build.ModelType) -> int:
    """
    Returns the number of parameters of a given model
    """
    if model_type == build.ModelType.PYTORCH:
        return sum([parameter.numel() for _, parameter in model.named_parameters()])
    elif build.ModelType.KERAS:
        return model.count_params()

    # Raise exception if an unsupported model type is provided
    raise AnalysisException(f"model_type {model_type} is not supported")

def get_onnx_ops_list(onnx_model) -> Dict:
    """
    List unique ops found in the onnx model 
    """
    onnx_ops_counter = {}
    model = onnx.load(onnx_model)
    assert model is not None
    for node in model.graph.node:
        if node.op_type not in onnx_ops_counter:
            onnx_ops_counter[node.op_type] = 1
        else:
            onnx_ops_counter[node.op_type] += 1
    return onnx_ops_counter

def populate_onnx_model_info(onnx_model) -> Dict:
    """
    Read the model metadata to populate IR, Opset and model size
    """
    model_metadata = {}
    model = onnx.load(onnx_model)
    model_metadata["ir_version"] = model.ir_version
    opset_str = str(model.opset_import)
    model_metadata["opset"] = int(re.search(r"\d+", opset_str).group())
    model_metadata["size on disk(KiB)"] = int(model.ByteSize())/1024
    return model_metadata

def onnx_input_dimensions(onnx_model) -> Dict:
    """
    Read model input dimensions
    """
    model = onnx.load(onnx_model)
    input_shape = {}
    for inp in model.graph.input:
        shape = str(inp.type.tensor_type.shape.dim)
        input_shape[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
    return input_shape

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
