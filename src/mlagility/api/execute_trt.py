"""
The following script is used to get the latency and outputs of a given run on the NVIDIA GPU.
This script doesn't depend on GroqFlow to be executed.
"""
# pylint: disable = no-name-in-module
# pylint: disable = import-error
import argparse
import subprocess
import json


def run(
    output_dir: str,
    onnx_model: str,
    outputs_file: str,
    errors_file: str,
    repetitions: int,
):
    # Latest docker image can be found here:
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags
    # GroqFlow maintainers to keep this up to date with the latest version of release container
    latest_trt_docker = "nvcr.io/nvidia/tensorrt:22.12-py3"
    docker_name = "tensorrt22.12"

    # docker run args:
    # "--gpus all" - use all gpus available
    # "-v <path>" -  mount the home dir to access the model inside the docker
    # "-itd" - start the docker in interactive mode in the background
    # "--rm" - remove the container automatically upon stopping
    docker_run_args = f"--gpus all -v {output_dir}:/app  -itd --rm"

    # docker exec args:
    # "--onnx=<path>" - path to the onnx model in the mounted file
    # "--fp16" - enable execution in fp16 mode on tensorrt
    # "--iterations=<int>" - number of iterations to run on tensorrt
    docker_exec_args = f"trtexec --onnx={onnx_model} --fp16 --iterations={repetitions}"

    run_command = (
        f"sudo docker run --name {docker_name} {docker_run_args} {latest_trt_docker}"
    )
    exec_command = f"sudo docker exec {docker_name} {docker_exec_args}"
    stop_command = f"sudo docker stop {docker_name}"

    # Run TensorRT docker in interactive model
    run_docker = subprocess.Popen(run_command.split(), stdout=subprocess.PIPE)
    run_docker.communicate()

    # Execute the onnx model user trtexec inside the container
    run_trtexec = subprocess.Popen(exec_command.split(), stdout=subprocess.PIPE)
    trtexec_output, trtexec_error = run_trtexec.communicate()

    # Stop the container
    stop_docker = subprocess.Popen(stop_command.split(), stdout=subprocess.PIPE)
    stop_docker.communicate()

    output = str(trtexec_output)
    error = str(trtexec_error)

    decoded_output = bytes(output, "utf-8").decode("unicode_escape")
    decoded_error = bytes(error, "utf-8").decode("unicode_escape")

    field_mapping = {
        "Selected Device": "Selected Device",
        "TensorRT version": "TensorRT version",
        "H2D Latency:": "Host to Device Latency",
        "GPU Compute Time:": "GPU Compute Time:",
        "D2H Latency:": "Device to Host Latency",
        "Throughput": "Throughput",
        "Latency:": "Total Latency",
    }

    def format_field(line: str):
        all_metrics = line.split(":")[-1].strip()
        if "," not in all_metrics:
            return all_metrics
        metrics = {
            k: v for k, v in (m.strip().split("=") for m in all_metrics.split(","))
        }
        return metrics

    gpu_performance = {}
    for line in decoded_output.splitlines():
        for field, key in field_mapping.items():
            if field in line:
                gpu_performance[key] = format_field(line)
                break

    with open(outputs_file, "w", encoding="utf-8") as out_file:
        json.dump(gpu_performance, out_file, ensure_ascii=False, indent=4)

    with open(errors_file, "w", encoding="utf-8") as e:
        e.writelines(decoded_error)


if __name__ == "__main__":
    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models built by GroqFlow")
    parser.add_argument(
        "output_dir",
        help="Path where the build files are stored",
    )
    parser.add_argument(
        "onnx_file",
        help="Path where the ONNX file is located",
    )
    parser.add_argument(
        "outputs_file",
        help="File in which the outputs will be saved",
    )
    parser.add_argument(
        "errors_file",
        help="File in which the outputs will be saved",
    )
    parser.add_argument(
        "iterations",
        type=int,
        help="Number of times to execute the received onnx model",
    )
    args = parser.parse_args()

    run(
        args.output_dir,
        args.onnx_file,
        args.outputs_file,
        args.errors_file,
        args.iterations,
    )
