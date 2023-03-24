"""
Helper functions for dealing with ONNX files and ONNX models
"""

from typing import Tuple
import numpy as np
import onnx
import onnxflow.common.exceptions as exp


def parameter_count(model):
    weights = model.graph.initializer
    parameter_count = 0

    for w in weights:
        weight = onnx.numpy_helper.to_array(w)
        parameter_count += np.prod(weight.shape)
    return parameter_count


def io_bytes(onnx_path: str) -> Tuple[int, int]:
    """Return the number of bytes of each of the inputs and outputs"""
    # pylint: disable = no-member

    def elem_type_to_bytes(elem_type) -> int:
        """
        Convert ONNX's elem_type to the number of bytes used by
        hardware to send that specific datatype through PCIe
        """
        if (
            elem_type == onnx.TensorProto.DataType.UINT8
            or elem_type == onnx.TensorProto.DataType.INT8
            or elem_type == onnx.TensorProto.DataType.BOOL
        ):
            # Each bool requires an entire byte
            return 1
        elif (
            elem_type == onnx.TensorProto.DataType.UINT16
            or elem_type == onnx.TensorProto.DataType.INT16
            or elem_type == onnx.TensorProto.DataType.FLOAT16
        ):
            return 2
        if (
            elem_type == onnx.TensorProto.DataType.FLOAT
            or elem_type == onnx.TensorProto.DataType.INT32
            or elem_type == onnx.TensorProto.DataType.INT64
            or elem_type == onnx.TensorProto.DataType.DOUBLE
            or elem_type == onnx.TensorProto.DataType.UINT64
        ):
            # 64 bit ints are treated as 32 bits everywhere
            # Doubles are treated as floats
            return 4
        elif (
            elem_type == onnx.TensorProto.DataType.COMPLEX64
            or elem_type == onnx.TensorProto.DataType.COMPLEX128
            or elem_type == onnx.TensorProto.DataType.STRING
            or elem_type == onnx.TensorProto.DataType.UNDEFINED
        ):
            raise exp.Error("Unsupported data type")
        else:
            raise exp.Error("Unsupported data type (unknown to ONNX)")

    def get_nodes_bytes(nodes):
        nodes_bytes = {}
        for node in nodes:

            # Get the number of the data type
            dtype_bytes = elem_type_to_bytes(node.type.tensor_type.elem_type)

            # Calculate the total number of elements based on the shape
            shape = str(node.type.tensor_type.shape.dim)
            num_elements = np.prod([int(s) for s in shape.split() if s.isdigit()])

            # Assign a total number of bytes to each node
            nodes_bytes[node.name] = num_elements * dtype_bytes

        return nodes_bytes

    # Get the number of bytes of each of the inputs and outputs
    model = onnx.load(onnx_path)
    onnx_input_bytes = get_nodes_bytes(model.graph.input)
    onnx_output_bytes = get_nodes_bytes(model.graph.output)

    return int(sum(onnx_input_bytes.values())), int(sum(onnx_output_bytes.values()))
