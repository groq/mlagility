import os
import sys
import subprocess
from typing import Tuple, Union, Dict, Any
from stat import S_ISDIR
import getpass
import shutil
import yaml
import paramiko
import groqflow.common.exceptions as exp
import groqflow.common.build as build
import groqflow.common.sdk_helpers as sdk
import groqflow.groqmodel.groqmodel as groqmodel


class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target) -> None:
        # Removes previous directory before transferring
        for item in os.listdir(source):
            if ".aa" in item or ".onnx" in item or ".json" in item or ".npy" in item:
                continue
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), "%s/%s" % (target, item))
            else:
                self.mkdir("%s/%s" % (target, item))
                self.put_dir(os.path.join(source, item), "%s/%s" % (target, item))

    def is_dir(self, path) -> bool:
        try:
            return S_ISDIR(self.stat(path).st_mode)
        except IOError:
            return False

    def rm_dir(self, path) -> None:
        files = self.listdir(path)
        for f in files:
            filepath = os.path.join(path, f)
            if self.is_dir(filepath):
                self.rm_dir(filepath)
            else:
                self.remove(filepath)

    def mkdir(self, path, mode=511) -> None:
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            self.rm_dir(path)


def load_remote_config(accelerator: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = f"{dir_path}/config.yaml"

    # Create a configuration file if one doesn't exist already
    if not os.path.exists(config_file_path):
        conf: Dict[str, Any] = {
            "remote_machine_gpu": {"ip": None, "username": None},
            "remote_machine_cpu": {"ip": None, "username": None},
            "remote_machine_groqchip": {"ip": None, "username": None},
        }
        with open(config_file_path, "w", encoding="utf8") as outfile:
            yaml.dump(conf, outfile)

    # Return the contents of the configuration file
    config_file = open(config_file_path, encoding="utf8")
    conf = yaml.load(config_file, Loader=yaml.FullLoader)
    return (
        conf[f"remote_machine_{accelerator}"]["ip"],
        conf[f"remote_machine_{accelerator}"]["username"],
    )


def save_remote_config(ip, username, accelerator) -> None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file = open(f"{dir_path}/config.yaml", encoding="utf8")
    conf = yaml.load(config_file, Loader=yaml.FullLoader)
    conf[f"remote_machine_{accelerator}"]["ip"] = ip
    conf[f"remote_machine_{accelerator}"]["username"] = username
    with open(f"{dir_path}/config.yaml", "w", encoding="utf8") as outfile:
        yaml.dump(conf, outfile)


def connect_to_host(ip, username) -> paramiko.SSHClient:
    print(f"Connecting to {username}@{ip}")

    class AllowAllKeys(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            return

    ssh_config = paramiko.SSHConfig.from_path("/etc/ssh/ssh_config")

    # Set SFT_AUTH_SOCK env var to enable users to access gcloud instances
    auth_sock_lines = subprocess.check_output(
        [
            "find",
            f"/var/run/sftd/client_trust_forwarding/{username}/",
            "-type",
            "s",
        ]
    )
    auth_sock = auth_sock_lines.split()[0]
    unicode_auth_sock = auth_sock.decode("utf-8")
    os.environ["SFT_AUTH_SOCK"] = unicode_auth_sock

    host_conf = ssh_config.lookup(ip)
    sock = paramiko.ProxyCommand(host_conf["proxycommand"])

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
    client.set_missing_host_key_policy(AllowAllKeys())
    client.connect(ip, username=username, sock=sock)

    return client


def exec_command(client, command, ignore_error=False) -> Tuple[str, str]:
    _, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    stdout = stdout.read().decode("ascii").strip("\n")
    stderr = str(stderr.read(), "utf-8")
    if not ignore_error:
        print(stderr)

    return stdout, exit_code


def configure_remote(accelerator: str) -> Tuple[str, str]:
    # Load stored values
    ip, username = load_remote_config(accelerator)

    if not all((ip, username)):
        if accelerator == "groqchip":
            print(
                (
                    "User is responsible for ensuring the remote server has the Groq "
                    "SDK and a miniconda environment named 'groqflow' installed."
                )
            )
        else:
            print(
                (
                    "User is responsible for ensuring the remote server has python>=3.8"
                    "and docker>=20.10 installed"
                )
            )

        print("Provide your instance IP and hostname below:")

        ip = ip or input(f"{accelerator} instance ASA name (Do not use IP): ")
        username = username or input(f"Username for {ip}: ")

        if not username or not ip:
            raise exp.GroqModelRuntimeError("Username and hostname are required")

        # Store information on yaml file
        save_remote_config(ip, username, accelerator)

    return ip, username

def setup_groqchip_host(client) -> None:
    # Make sure at least one GroqChip Processor is available remotely
    stdout, exit_code = exec_command(client, "/usr/bin/lspci -n")
    if stdout == "" or exit_code == 1:
        msg = "Failed to run lspci to get GroqChip Processors available"
        raise exp.GroqModelRuntimeError(msg)
    num_chips_available = sdk.get_num_chips_available(stdout.split("\n"))
    if num_chips_available < 1:
        raise exp.GroqModelRuntimeError("No GroqChip Processor(s) found")
    print(f"{num_chips_available} GroqChip Processor(s) found")

    # Transfer common files to host
    exec_command(client, "mkdir groqflow_remote_cache", ignore_error=True)
    dir_path = os.path.dirname(groqmodel.__file__)
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.put(f"{dir_path}/execute.py", "groqflow_remote_cache/execute.py")

def setup_remote_host(client, device_type: str, output_dir: str) -> None:
    if device_type == "gpu":
        # Check if at least one NVIDIA GPU is available remotely
        stdout, exit_code = exec_command(client, "lspci | grep -i nvidia")
        if stdout == "" or exit_code == 1:
            msg = "No NVIDIA GPUs available on the remote machine"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = ["execute-gpu.py"]
    elif device_type == "cpu":
        # Check if x86_64 CPU is available remotely
        stdout, exit_code = exec_command(client, "uname -i")
        if stdout != "x86_64" or exit_code == 1:
            msg = "Only x86_64 CPUs are supported at this time for competitive benchmarking"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = ["execute-cpu.py", "setup_ort_env.sh"]
    else:
        raise ValueError(f"Only 'cpu' and 'gpu' are supported. But received {device_type}")

    # Transfer common files to host
    exec_command(client, f"mkdir {output_dir}", ignore_error=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with MySFTPClient.from_transport(client.get_transport()) as s:
        for file in files_to_transfer:
            s.put(f"{dir_path}/{file}", f"{output_dir}/{file}")

def setup_local_host(device_type: str, output_dir: str) -> None:
    if device_type == "cpu":
        # Check if x86_64 CPU is available locally
        check_device = subprocess.run(
            ["uname", "-i"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        stdout = check_device.stdout.decode().strip()
        if stdout != "x86_64" or check_device.returncode == 1:
            msg = "Only x86_64 CPUs are supported at this time for competitive benchmarking"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = ["execute-cpu.py", "setup_ort_env.sh"]

    elif device_type == "gpu":
        # Check if at least one NVIDIA GPU is available locally
        result = subprocess.run(
            ["lspci"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=False
        )
        print (result)
        if "NVIDIA" not in result.stdout or result.returncode == 1:
            msg = "No NVIDIA GPUs available on the local machine"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = ["execute-gpu.py"]

    else:
        raise ValueError(f"Invalid device type: {device_type}")

    # Transfer files to host
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    subprocess.run(["mkdir", f"{output_dir}"], check=False)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for file in files_to_transfer:
        shutil.copy(f"{dir_path}/{file}", f"{output_dir}/{file}")

def setup_connection(device_type: str, output_dir: str = None) -> paramiko.SSHClient:
    # Setup authentication scheme if needed
    ip, username = configure_remote(device_type)

    # Connect to host
    client = connect_to_host(ip, username)

    if device_type == "groqchip":
        # Check for GroqChips and transfer common files
        setup_groqchip_host(client)
    elif device_type == "cpu" or device_type == "gpu":
        setup_remote_host(client, device_type=device_type, output_dir=output_dir)
    else:
        raise ValueError(
            f"Only 'cpu' and 'gpu' are supported, but received {device_type}"
        )

    return client


def execute_groqchip_remotely(
    bringup_topology: bool,
    repetitions: int,
    state: build.State,
    log_execute_path: str,
) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _, hostname = configure_remote("groqchip")

    # Redirect all stdout to log_file
    sys.stdout = build.Logger(log_execute_path)

    # Connect to remote machine and transfer common files
    client = setup_connection("groqchip")

    # Transfer iop and inputs file
    print("Transferring model and inputs...")
    if not os.path.exists(state.execution_inputs_file):
        msg = "Model input file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir("groqflow_remote_cache/compile")
        s.put_dir(state.compile_dir, "groqflow_remote_cache/compile")
        s.put(state.execution_inputs_file, "groqflow_remote_cache/inputs.npy")

    python_cmd = (
        "export PYTHONPATH='/opt/groq/runtime/site-packages:$PYTHONPATH' && "
        f"/home/{hostname}/miniconda3/envs/groqflow/bin/python"
    )

    # Run benchmarking script
    output_dir = "groqflow_remote_cache"
    remote_outputs_file = "groqflow_remote_cache/outputs.npy"
    remote_latency_file = "groqflow_remote_cache/latency.npy"
    print("Running benchmarking script...")

    bringup_topology_arg = "" if bringup_topology else "--bringup_topology"
    _, exit_code = exec_command(
        client,
        (
            f"{python_cmd} groqflow_remote_cache/execute.py "
            f"{state.num_chips_used} {output_dir} {remote_outputs_file} "
            f"{remote_latency_file} {state.topology} {repetitions} "
            f"{bringup_topology_arg}"
        ),
    )
    if exit_code == 1:
        msg = f"""
        Failed to execute GroqChip Processor(s) remotely.
        Look at **{log_execute_path}** for details.
        """
        raise exp.GroqModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.get(remote_outputs_file, state.outputs_file)
        s.get(remote_latency_file, state.latency_file)
        s.remove(remote_outputs_file)
        s.remove(remote_latency_file)
    # Stop redirecting stdout
    sys.stdout = sys.stdout.terminal


def execute_gpu_remotely(
    state: build.State, log_execute_path: str, iterations: int
) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _ip, username = configure_remote("gpu")

    # Redirect all stdout to log_file
    sys.stdout = build.Logger(log_execute_path)

    # Setup remote execution folders to save outputs/ errors
    output_dir = f"/home/{username}/mlagility_remote_cache"
    remote_outputs_file = f"{output_dir}/outputs.txt"
    remote_errors_file = f"{output_dir}/errors.txt"

    # Connect to remote machine and transfer common files
    client = setup_connection(device_type="gpu", output_dir=output_dir)

    print("Transferring model file...")
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir(f"{output_dir}/onnxmodel")
        s.put(state.converted_onnx_file, f"{output_dir}/onnxmodel/model.onnx")

    # Run benchmarking script
    print("Running benchmarking script...")
    _, exit_code = exec_command(
        client,
        (
            f"/usr/bin/python3 {output_dir}/execute-gpu.py "
            f"{output_dir} {remote_outputs_file} {remote_errors_file} {iterations} {username}"
        ),
    )
    if exit_code == 1:
        msg = f"""
        Failed to execute GPU(s) remotely.
        Look at **{log_execute_path}** for details.
        """
        raise exp.GroqModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        try:
            s.get(
                remote_outputs_file,
                os.path.join(
                    state.cache_dir, state.config.build_name, "gpu_performance.json"
                ),
            )
            s.get(
                remote_errors_file,
                os.path.join(state.cache_dir, state.config.build_name, "gpu_error.npy"),
            )
            s.remove(remote_outputs_file)
            s.remove(remote_errors_file)
        except FileNotFoundError:
            print(
                "Output/ error files not found! Please make sure your remote GPU machine is"
                "turned ON and has all the required dependencies installed"
            )
    # Stop redirecting stdout
    sys.stdout = sys.stdout.terminal


def execute_gpu_locally(
    state: build.State, log_execute_path: str, iterations: int
) -> None:
    """
    Execute Model on the local GPU
    """

    # Redirect all stdout to log_file
    sys.stdout = build.Logger(log_execute_path)

    # Setup local execution folders to save outputs/ errors
    username = getpass.getuser()
    output_dir = f"/home/{username}/mlagility_local_cache"
    outputs_file = f"{output_dir}/outputs.txt"
    errors_file = f"{output_dir}/errors.txt"

    setup_local_host(device_type="gpu", output_dir=output_dir)

    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    os.makedirs(f"{output_dir}/onnxmodel")
    shutil.copy(state.converted_onnx_file, f"{output_dir}/onnxmodel/model.onnx")

    # Check if docker is installed
    docker_location = shutil.which("docker")
    if not docker_location:
        raise ValueError("'docker' installation not found. Please install docker>=20.10")

    # Check if python is installed
    python_location = shutil.which("python")
    if not python_location:
        raise ValueError("'python' installation not found. Please install python>=3.8")

    print("Running benchmarking script...")

    run_benchmark = subprocess.Popen(
        [
            python_location,
            f"{output_dir}/execute-gpu.py",
            f"{output_dir}",
            f"{outputs_file}",
            f"{errors_file}",
            f"{iterations}",
            f"{username}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = run_benchmark.communicate()
    if run_benchmark.returncode == 0:
        print(f"Success: Running model using onnxrutime - {stdout.decode().strip()}")
    else:
        print(
            f"Error: Failure to run model using onnxruntime - {stderr.decode().strip()}"
        )

    # Move output files back to the build cache
    shutil.move(
        outputs_file,
        os.path.join(state.cache_dir, state.config.build_name, "gpu_performance.json"),
    )
    shutil.move(
        errors_file,
        os.path.join(state.cache_dir, state.config.build_name, "gpu_error.npy"),
    )

    # Delete the local cache folder created
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # Stop redirecting stdout
    sys.stdout = sys.stdout.terminal


def execute_cpu_remotely(
    state: build.State, log_execute_path: str, iterations: int
) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _ip, username = configure_remote("cpu")

    # Redirect all stdout to log_file
    sys.stdout = build.Logger(log_execute_path)

    # Setup remote execution folders to save outputs/ errors
    output_dir = f"/home/{username}/mlagility_remote_cache"
    remote_outputs_file = f"{output_dir}/outputs.txt"
    remote_errors_file = f"{output_dir}/errors.txt"

    # Connect to remote machine and transfer common files
    client = setup_connection(device_type="cpu", output_dir=output_dir)

    print("Transferring model file...")
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir(f"{output_dir}/onnxmodel")
        s.put(state.converted_onnx_file, f"{output_dir}/onnxmodel/model.onnx")

    # Check if conda is installed on the remote machine
    conda_location, exit_code = exec_command(client, f"ls /home/{username}")
    if "miniconda3" not in conda_location:
        raise ValueError(
                f"conda installation not found in /home/{username}. Please install miniconda3"
            )
    # TODO: Remove requirement that conda has to be installed on the /home/user
    conda_src = f"/home/{username}"

    # Run benchmarking script
    env_name = "onnxruntime_env"
    exec_command(
        client, f"bash {output_dir}/setup_ort_env.sh {env_name} {conda_src}", ignore_error=True
    )

    print("Running benchmarking script...")
    _, exit_code = exec_command(
        client,
        (
            f"/home/{username}/miniconda3/envs/{env_name}/bin/python {output_dir}/"
            "execute-cpu.py "
            f"{output_dir} {remote_outputs_file} {remote_errors_file} {iterations}"
        ),
    )
    if exit_code == 1:
        msg = f"""
        Failed to execute CPU(s) remotely.
        Look at **{log_execute_path}** for details.
        """
        raise exp.GroqModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        try:
            s.get(
                remote_outputs_file,
                os.path.join(
                    state.cache_dir, state.config.build_name, "cpu_performance.json"
                ),
            )
            s.get(
                remote_errors_file,
                os.path.join(state.cache_dir, state.config.build_name, "cpu_error.npy"),
            )
            s.remove(remote_outputs_file)
            s.remove(remote_errors_file)
        except FileNotFoundError:
            print(
                "Output/ error files not found! Please make sure your remote CPU machine is"
                "turned ON and has all the required dependencies installed"
            )
    # Stop redirecting stdout
    sys.stdout = sys.stdout.terminal


def execute_cpu_locally(
    state: build.State, log_execute_path: str, iterations: int
) -> None:
    """
    Execute Model on the local CPU
    """

    # Redirect all stdout to log_file
    sys.stdout = build.Logger(log_execute_path)

    # Setup local execution folders to save outputs/ errors
    username = getpass.getuser()
    output_dir = f"/home/{username}/mlagility_local_cache"
    outputs_file = f"{output_dir}/outputs.txt"
    errors_file = f"{output_dir}/errors.txt"

    setup_local_host(device_type="cpu", output_dir=output_dir)

    # Check if ONNX file has been generated
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    os.makedirs(f"{output_dir}/onnxmodel")
    shutil.copy(state.converted_onnx_file, f"{output_dir}/onnxmodel/model.onnx")

    # Check if conda is installed
    conda_location = shutil.which("conda")
    if not conda_location:
        raise ValueError("conda installation not found.")
    conda_src = conda_location.split("miniconda3")[0]

    # Create/ update local conda environment for CPU benchmarking
    print("Creating environment...")
    env_name = "onnxruntime-env"
    setup_env = subprocess.Popen(
        [
            "bash",
            f"{output_dir}/setup_ort_env.sh",
            f"{env_name}",
            f"{conda_src}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = setup_env.communicate()
    if setup_env.returncode == 0:
        print(f"Success: Setup/ updated conda environment - {stdout.decode().strip()}")
    else:
        print(f"Error: Failure to setup conda environment - {stderr.decode().strip()}")

    # Run the benchmark
    print("Running benchmarking script...")
    run_benchmark = subprocess.Popen(
        [
            f"{conda_src}miniconda3/envs/{env_name}/bin/python",
            f"{output_dir}/execute-cpu.py",
            f"{output_dir}",
            f"{outputs_file}",
            f"{errors_file}",
            f"{iterations}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = run_benchmark.communicate()
    if run_benchmark.returncode == 0:
        print(f"Success: Running model using onnxrutime - {stdout.decode().strip()}")
    else:
        print(
            f"Error: Failure to run model using onnxruntime - {stderr.decode().strip()}"
        )

    # Move output files back to the build cache
    shutil.move(
        outputs_file,
        os.path.join(state.cache_dir, state.config.build_name, "cpu_performance.json"),
    )
    shutil.move(
        errors_file,
        os.path.join(state.cache_dir, state.config.build_name, "cpu_error.npy"),
    )

    # Delete the local cache folder created
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # Stop redirecting stdout
    sys.stdout = sys.stdout.terminal
