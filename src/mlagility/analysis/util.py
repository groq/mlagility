import sys
from dataclasses import dataclass
from typing import Callable, List, Union, Dict
import inspect
import torch
import onnx
from onnxflow.common import printing
import onnxflow.common.build as build
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
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        return onnx_ops_counter
    for node in model.graph.node:  # pylint: disable=E1101
        onnx_ops_counter[node.op_type] = onnx_ops_counter.get(node.op_type, 0) + 1
    return onnx_ops_counter


def populate_onnx_model_info(onnx_model) -> Dict:
    """
    Read the model metadata to populate IR, Opset and model size
    """
    result_dict = {
        "ir_version": None,
        "opset": None,
        "size on disk (KiB)": None,
    }
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        result_dict.update({"error": "ONNX model analysis failed"})
        return result_dict
    result_dict.update(
        {
            "ir_version": getattr(model, "ir_version", None),
            "opset": getattr(
                model.opset_import[-1], "version", None
            ),  # pylint: disable=E1101
            "size on disk (KiB)": round(
                model.SerializeToString().__sizeof__() / 1024, 4
            ),
        }
    )
    return result_dict


def onnx_input_dimensions(onnx_model) -> Dict:
    """
    Read model input dimensions
    """
    input_shape = {}
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        return input_shape
    for inp in model.graph.input:  # pylint: disable=E1101
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
