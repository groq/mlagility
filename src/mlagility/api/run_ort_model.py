import argparse
import re
import math
from timeit import default_timer as timer
import numpy as np
import onnxruntime as ort

# NOTE: functionality in this module is replicated in onnxflow.common.onnx_helpers
# to help make it available to all onnxflow users. This code could be de-duplicated
# if mlagility allows this module to import onnxflow.


def run_ort_profile(onnx_file, num_iterations=100):
    # Run the provided onnx model using onnxruntime and measure average latency

    per_iteration_latency = []
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = ort.InferenceSession(onnx_file, sess_options)
    sess_input = onnx_session.get_inputs()
    input_feed = _dummy_inputs(sess_input)
    output_name = onnx_session.get_outputs()[0].name

    for _ in range(num_iterations):
        start = timer()
        onnx_session.run([output_name], input_feed)
        end = timer()
        iteration_latency = end - start
        per_iteration_latency.append(iteration_latency)

    print(per_iteration_latency)


def _dummy_inputs(sess_input) -> dict:
    # Generate dummy inputs of the expected shape and type for the input model
    input_stats = []
    for _idx, input_ in enumerate(range(len(sess_input))):
        input_name = sess_input[input_].name
        input_shape = sess_input[input_].shape

        # TODO: Use onnx update_inputs_outputs_dims to automatically freeze models
        for dim in input_shape:
            if isinstance(dim, str) is True or math.isnan(dim) is True:
                raise AssertionError(
                    "Error: Model has dynamic inputs. Freeze the graph and try again"
                )

        input_type = sess_input[input_].type
        input_stats.append([input_name, input_shape, input_type])

    input_feed = {}
    for stat in input_stats:
        dtype_str = re.search(r"\((.*)\)", stat[2])
        assert dtype_str is not None
        datatype = dtype_ort2str(dtype_str.group(1))
        input_feed[stat[0]] = np.random.rand(*stat[1]).astype(datatype)
    return input_feed


def dtype_ort2str(dtype_str: str):
    if dtype_str == "float16":
        datatype = "float16"
    elif dtype_str == "float":
        datatype = "float32"
    elif dtype_str == "double":
        datatype = "float64"
    elif dtype_str == "long":
        datatype = "int64"
    else:
        datatype = dtype_str
    return datatype


if __name__ == "__main__":
    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models using onnxruntime")
    parser.add_argument(
        "--onnx-file",
        required=True,
        help="Path where the ONNX file is located",
    )
    parser.add_argument(
        "--iterations",
        required=True,
        type=int,
        help="Number of times to execute the received onnx model",
    )
    args = parser.parse_args()

    run_ort_profile(
        onnx_file=args.onnx_file,
        num_iterations=args.iterations,
    )
