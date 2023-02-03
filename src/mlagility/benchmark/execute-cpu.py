"""
The following script is used to get the latency and outputs of a given run on the x86 CPUs.
This script doesn't depend on GroqFlow to be executed.
"""
# pylint: disable = no-name-in-module
# pylint: disable = import-error
import argparse
import subprocess
import json
import re
import math
from statistics import mean
from timeit import default_timer as timer
import numpy as np
import onnxruntime as ort

BATCHSIZE = 1


def run_ort_profile(source_onnx, num_iterations=100) -> str:
    # Run the provided onnx model using onnxruntime and measure average latency

    per_iteration_latency = []
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )
    onnx_session = ort.InferenceSession(source_onnx, sess_options)
    sess_input = onnx_session.get_inputs()
    input_feed = _dummy_inputs(sess_input)
    # input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    for _ in range(num_iterations):
        start = timer()
        onnx_session.run([output_name], input_feed)
        end = timer()
        iteration_latency = end - start
        per_iteration_latency.append(iteration_latency)

    return per_iteration_latency


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
    if dtype_str == "float":
        datatype = "float32"
    if dtype_str == "double":
        datatype = "float64"
    if dtype_str == "long":
        datatype = "int64"
    else:
        datatype = dtype_str
    return datatype


def run(
    output_dir: str,
    num_iterations: int,
):

    # TODO: Handle exception for failure
    onnx_model = f"{output_dir}/model.onnx"
    perf_result = run_ort_profile(onnx_model, num_iterations)
    # assert perf_result is list of int or float
    save_ort_results(perf_result, num_iterations)


def save_ort_results(perf_result, num_iterations):

    # Get CPU spec from lscpu
    cpu_info_command = "lscpu"
    cpu_info = subprocess.Popen(cpu_info_command.split(), stdout=subprocess.PIPE)
    cpu_info_output, cpu_info_error = cpu_info.communicate()
    decoded_info = bytes(str(cpu_info_output), "utf-8").decode("unicode_escape")

    with open("cpu_info.txt", "w", encoding="utf-8") as f:
        f.writelines(decoded_info)

    info_file = open("cpu_info.txt", "r", encoding="utf-8")

    field_mapping = {
        "Architecture": "CPU Architecture",
        "Vendor ID": "CPU Vendor",
        "Model name": "CPU Name",
        "CPU family": "CPU Family",
        "Model": "CPU Model",
        "CPU MHz": "CPU Max Frequency (MHz)",
        "CPU(s)": "CPU Core Count",
    }

    def format_field(line: str) -> str:
        return line.split(":")[-1].strip()

    cpu_performance = {}
    for line in info_file:
        for field, key in field_mapping.items():
            if field in line:
                cpu_performance[key] = format_field(line)
                break

    cpu_performance["OnnxRuntime Version"] = str(ort.__version__)
    cpu_performance["Mean Latency(ms)"] = str(mean(perf_result) * 1000 / num_iterations)
    cpu_performance["Throughput"] = str(
        BATCHSIZE / mean(perf_result) * 1000 / num_iterations
    )
    cpu_performance["Min Latency(ms)"] = str(min(perf_result) * 1000 / num_iterations)
    cpu_performance["Max Latency(ms)"] = str(max(perf_result) * 1000 / num_iterations)

    info_file.close()
    with open(args["outputs_file"], "w", encoding="utf-8") as out_file:
        json.dump(cpu_performance, out_file, ensure_ascii=False, indent=4)

    with open(args["errors_file"], "w", encoding="utf-8") as e:
        e.writelines(str(cpu_info_error))


if __name__ == "__main__":
    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models built by GroqFlow")
    parser.add_argument("output_dir", help="Path where the build files are stored")
    parser.add_argument("outputs_file", help="File in which the outputs will be saved")
    parser.add_argument("errors_file", help="File in which the outputs will be saved")
    parser.add_argument(
        "iterations",
        type=int,
        help="Number of times to execute the received onnx model",
    )
    args = vars(parser.parse_args())

    run(
        args["output_dir"],
        args["iterations"],
    )
