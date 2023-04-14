"""
The following script is used to get the latency and outputs of a given run on the x86 CPUs.
This script doesn't depend on GroqFlow to be executed.
"""
# pylint: disable = no-name-in-module
# pylint: disable = import-error
import argparse
import subprocess
import json
import logging
from statistics import mean
from typing import Dict

BATCHSIZE = 1

# Set a 15 minutes timeout for all docker commands
TIMEOUT = 900


class SubprocessError(Exception):
    pass


def run_subprocess(cmd):
    """Run a subprocess with the given command and log the output."""
    logging.info(f"Running subprocess with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=TIMEOUT)
        logging.info(f"Subprocess finished with command: {' '.join(cmd)}")
    except subprocess.TimeoutExpired:
        logging.error(f"{' '.join(cmd)} timed out after {TIMEOUT} seconds")
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Subprocess failed with command: {' '.join(cmd)} and error message: {e.stderr}"
        )
        raise SubprocessError("CalledProcessError")
    except (OSError, ValueError) as e:
        logging.error(
            f"Subprocess failed with command: {' '.join(cmd)} and error message: {str(e)}"
        )
        raise SubprocessError(str(e))


def build_docker_image(output_dir, docker_image):
    """Build a docker image with the given name."""
    cmd = ["docker", "build", "-t", docker_image, output_dir]
    run_subprocess(cmd)


def run_docker_container(output_dir, docker_name, docker_image):
    """Run a docker container with the given name and image."""
    # docker run args:
    # "-v <path>" -  mount the home dir to access the model inside the docker
    # "-d" - start the docker in detached mode in the background
    # "--rm" - remove the container automatically upon stopping
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "-v",
        f"{output_dir}:/app",
        "--name",
        docker_name,
        docker_image,
    ]
    try:
        output = subprocess.check_output(cmd, text=True).strip()
        logging.info(f"Container {docker_name} started with ID: {output}")
        return output
    except subprocess.CalledProcessError as e:
        logging.error(
            (
                f"Failed to start Docker container with command: "
                f"{' '.join(cmd)} and error message: {e.stderr}"
            )
        )
        raise


def execute_benchmark(onnx_file, docker_name, num_iterations):
    """Execute the benchmark script in a docker container and retrieve the output."""
    cmd = (
        "docker exec "
        f"{docker_name} "
        "/usr/bin/python3 /app/run_ort_model.py "
        f"--onnx-file {onnx_file} "
        f"--iterations {num_iterations}"
    )

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    ) as proc:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0 or not stdout:
            raise ValueError(f"Execution of command {cmd} failed with stderr: {stderr}")
        try:
            output_list = json.loads(stdout.decode("utf-8").strip())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse the output {stdout.decode('utf-8').strip()} as a list: {e}"
            )
        return output_list


def stop_docker_container(container_id):
    """Stop and remove the docker container with the given container ID."""
    cmd = ["docker", "stop", container_id]
    run_subprocess(cmd)


def get_ort_version(docker_name):
    """Stop and remove the docker container with the given name."""
    cmd = (
        "docker exec "
        f"{docker_name} "
        '/usr/bin/python3 -c "import onnxruntime as ort; print(ort.__version__)"'
    )

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    ) as proc:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0 or not stdout:
            raise ValueError(f"Execution of command {cmd} failed with stderr: {stderr}")
    version = stdout.decode("utf-8").strip()
    return version


def run(
    output_dir: str,
    onnx_file: str,
    outputs_file: str,
    num_iterations: int,
):
    docker_image = "mlagility-onnxruntime-image"
    docker_name = "mlagility-onnxruntime-mlas-ep"

    try:
        # Build the docker image
        build_docker_image(output_dir, docker_image)
    except Exception as e:
        raise ValueError(f"Docker image build failed with exception: {e}")

    container_id = None
    try:
        # Run the docker container
        container_id = run_docker_container(output_dir, docker_name, docker_image)

        try:
            # Execute the benchmark script
            perf_result = execute_benchmark(onnx_file, docker_name, num_iterations)
        except Exception as e:
            raise ValueError(f"Benchmark execution failed with exception: {e}")

        try:
            ort_version = get_ort_version(docker_name)
        except Exception as e:
            raise ValueError(f"Get ort version failed with exception: {e}")

    except Exception as e:
        raise ValueError(f"Docker container run failed with exception: {e}")

    # Make sure the container is stopped even if there is a failure
    finally:
        if container_id:
            try:
                logging.info(f"Stopping docker container with ID: {container_id}")
                stop_docker_container(container_id)
            except SubprocessError as e:
                logging.error(f"Failed to stop Docker container with exception: {e}")

    cpu_performance = get_cpu_specs()
    cpu_performance["OnnxRuntime Version"] = str(ort_version)
    cpu_performance["Mean Latency(ms)"] = str(mean(perf_result) * 1000)
    cpu_performance["Throughput"] = str(BATCHSIZE / mean(perf_result))
    cpu_performance["Min Latency(ms)"] = str(min(perf_result) * 1000)
    cpu_performance["Max Latency(ms)"] = str(max(perf_result) * 1000)

    with open(outputs_file, "w", encoding="utf-8") as out_file:
        json.dump(cpu_performance, out_file, ensure_ascii=False, indent=4)


def get_cpu_specs() -> Dict:
    # Get CPU spec from lscpu
    cpu_info_command = "lscpu"
    cpu_info = subprocess.Popen(cpu_info_command.split(), stdout=subprocess.PIPE)
    cpu_info_output, _ = cpu_info.communicate()
    decoded_info = bytes(str(cpu_info_output), "utf-8").decode("unicode_escape")

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

    cpu_spec = {}
    for line in decoded_info.split("\n"):
        for field, key in field_mapping.items():
            if field in line:
                cpu_spec[key] = format_field(line)
                break

    return cpu_spec


if __name__ == "__main__":
    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models built by benchit")
    parser.add_argument(
        "--output-dir",
        required=True,
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
        "--iterations",
        required=True,
        type=int,
        help="Number of times to execute the received onnx model",
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        onnx_file=args.onnx_file,
        outputs_file=args.outputs_file,
        num_iterations=args.iterations,
    )
