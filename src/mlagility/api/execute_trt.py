"""
The following script is used to get the latency and outputs of a given run on the NVIDIA GPU.
This script doesn't depend on GroqFlow to be executed.
"""
# pylint: disable = no-name-in-module
# pylint: disable = import-error
import argparse
import subprocess
import logging
import json
import re
import sys
import time
import threading

# Set a 15 minutes timeout for all docker commands
TIMEOUT = 900

TRT_VERSION = "23.03-py3"


def run(
    output_dir: str,
    onnx_file: str,
    outputs_file: str,
    errors_file: str,
    num_iterations: int,
):
    # Latest docker image can be found here:
    # https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags
    # GroqFlow maintainers to keep this up to date with the latest version of release container
    latest_trt_docker = f"nvcr.io/nvidia/tensorrt:{TRT_VERSION}"
    docker_name = f"tensorrt{TRT_VERSION}"

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
    docker_exec_args = (
        f"trtexec --onnx={onnx_file} --fp16 --iterations={num_iterations}"
    )

    run_command = (
        f"sudo docker run --name {docker_name} {docker_run_args} {latest_trt_docker}"
    )
    exec_command = f"sudo docker exec {docker_name} {docker_exec_args}"
    stop_command = f"sudo docker stop {docker_name}"

    # Run TensorRT docker in interactive model
    run_docker = subprocess.Popen(run_command.split(), stdout=subprocess.PIPE)
    try:
        run_docker.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        logging.error(f"{run_command} timed out!")
    except subprocess.CalledProcessError as e:
        logging.error(f"{run_command} failed with error code {e.returncode}: {e}")

    # Execute the onnx model user trtexec inside the container
    run_trtexec = subprocess.Popen(exec_command.split(), stdout=subprocess.PIPE)
    try:
        trtexec_output, trtexec_error = run_trtexec.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        logging.error(f"{exec_command} timed out!")
    except subprocess.CalledProcessError as e:
        logging.error(f"{exec_command}  failed with error code {e.returncode}: {e}")

    # Stop the container
    stop_docker = subprocess.Popen(stop_command.split(), stdout=subprocess.PIPE)
    try:
        stop_docker.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        logging.error(f"{stop_command} timed out!")
    except subprocess.CalledProcessError as e:
        logging.error(f"{stop_command} failed with error code {e.returncode}: {e}")

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

def get_gpu_power():
    try:
        output = subprocess.check_output("nvidia-smi -q -d POWER".split())
        output = output.decode("utf-8")
        power_pattern = re.compile(r"Power Draw\s+:\s+(\d+\.\d+)\s+W")
        power_draw = float(power_pattern.search(output).group(1))
        return power_draw
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def measure_power(sample_rate=0.01, duration=10):
    start_time = time.time()
    power_readings = []
    while time.time() - start_time < duration:
        power_draw = get_gpu_power()
        power_readings.append(power_draw)
        time.sleep(sample_rate)

    average_power = sum(power_readings) / len(power_readings)
    return power_readings, average_power

if __name__ == "__main__":
    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models built by benchit")
    parser.add_argument(
        "--output-dir",
        help="Path where the build files are stored",
    )
    parser.add_argument(
        "--onnx-file",
        required=True,
        help="Path where the ONNX file is located",
    )
    parser.add_argument(
        "--outputs-file",
        required=True,
        help="File in which the outputs will be saved",
    )
    parser.add_argument(
        "--errors-file",
        required=True,
        help="File in which the outputs will be saved",
    )
    parser.add_argument(
        "--iterations",
        required=True,
        type=int,
        help="Number of times to execute the received onnx model",
    )
    args = parser.parse_args()

    power_readings = []
    average_power = 0

    # Start power measurement in a separate thread
    power_thread = threading.Thread(target=measure_power, args=(0.01, TIMEOUT))
    power_thread.start()

    run(
        output_dir=args.output_dir,
        onnx_file=args.onnx_file,
        outputs_file=args.outputs_file,
        errors_file=args.errors_file,
        num_iterations=args.iterations,
    )

    # Wait for power measurement to finish
    power_thread.join()

    # Calculate the average power consumption
    average_power = sum(power_readings) / len(power_readings)

    # Load existing GPU performance data
    with open(args.outputs_file, "r", encoding="utf-8") as out_file:
        gpu_performance = json.load(out_file)

    # Add power readings and average power consumption to the dictionary
    gpu_performance["Power readings (W)"] = power_readings
    gpu_performance["Average power consumption (W)"] = round(average_power, 2)

    # Save the updated GPU performance data
    with open(args.outputs_file, "w", encoding="utf-8") as out_file:
        json.dump(gpu_performance, out_file, ensure_ascii=False, indent=4)
