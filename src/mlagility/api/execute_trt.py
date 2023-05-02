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
    start_event: threading.Event,
    stop_event: threading.Event,
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

    # Start power measurement
    start_event.set()

    # Execute the onnx model user trtexec inside the container
    run_trtexec = subprocess.Popen(exec_command.split(), stdout=subprocess.PIPE)
    try:
        trtexec_output, trtexec_error = run_trtexec.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        logging.error(f"{exec_command} timed out!")
    except subprocess.CalledProcessError as e:
        logging.error(f"{exec_command}  failed with error code {e.returncode}: {e}")

    # Stop power measurement
    stop_event.set()

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


class NvidiaSmiError(Exception):
    """An exception class for errors related to nvidia-smi."""
    pass

"""
In the average_power_and_utilization function, we eliminate the values below
the idle_threshold at the beginning and end of the list because we want to 
exclude the idle periods before and after the actual workload execution. By
doing this, we can get a more accurate representation of the average power 
consumption and GPU utilization during the execution of the workload.
We set the Idle threshold to 3% utilization based on heuristics.
"""
IDLE_THRESHOLD = 3

def average_power_and_utilization(power_readings):

    if not power_readings:
        return 0, 0

    # Remove readings below the threshold from the beginning of the list
    while power_readings and power_readings[0][0] <= IDLE_THRESHOLD:
        power_readings.pop(0)

    # Remove readings below the threshold from the end of the list
    while power_readings and power_readings[-1][0] <= IDLE_THRESHOLD:
        power_readings.pop()

    if not power_readings:
        return 0, 0

    average_power = sum(power_draw for _, power_draw in power_readings) / len(power_readings)
    average_utilization = sum(utilization for utilization, _ in power_readings) / len(power_readings)

    return average_power, average_utilization


def get_gpu_power_and_utilization():
    try:
        query = "nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits"
        output = subprocess.check_output(query.split())
        output = output.decode("utf-8").strip()

        utilization, power_draw = output.split(", ")
        return int(utilization), float(power_draw)

    except subprocess.CalledProcessError as e:
        raise NvidiaSmiError(f"nvidia-smi failed with return code {e.returncode}")


def measure_power(start_event, stop_event, power_readings, sample_rate=0.01):
    start_event.wait()  # Wait for the start event to be set

    while not stop_event.is_set():
        gpu_utilization, power_draw = get_gpu_power_and_utilization()
        power_readings.append((gpu_utilization, power_draw))
        time.sleep(sample_rate)


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
    start_event = threading.Event()
    stop_event = threading.Event()
    power_thread = threading.Thread(
        target=measure_power, args=(start_event, stop_event, power_readings, 0.01)
    )
    power_thread.start()

    run(
        output_dir=args.output_dir,
        onnx_file=args.onnx_file,
        outputs_file=args.outputs_file,
        errors_file=args.errors_file,
        num_iterations=args.iterations,
        start_event=start_event,
        stop_event=stop_event,
    )

    # Wait for power measurement to finish
    power_thread.join()

    # Calculate the average power consumption, average utilization, and peak power consumption
    average_power, average_utilization = average_power_and_utilization(power_readings)
    peak_power = (
        max([reading[1] for reading in power_readings]) if power_readings else None
    )

    # Load existing GPU performance data
    with open(args.outputs_file, "r", encoding="utf-8") as out_file:
        gpu_performance = json.load(out_file)

    # Add average power consumption, average utilization, and peak power consumption to the dictionary
    gpu_performance["Average power consumption (W)"] = (
        round(average_power, 2) if average_power is not None else None
    )
    gpu_performance["Peak power consumption (W)"] = (
        round(peak_power, 2) if peak_power is not None else None
    )
    gpu_performance["Average GPU utilization (%)"] = (
        round(average_utilization, 2) if average_utilization is not None else None
    )

    # Save the updated GPU performance data
    with open(args.outputs_file, "w", encoding="utf-8") as out_file:
        json.dump(gpu_performance, out_file, ensure_ascii=False, indent=4)
