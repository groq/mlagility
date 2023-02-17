import os
import subprocess
from typing import Tuple, Union, Dict, Any
from stat import S_ISDIR
import shutil
import yaml
import paramiko
import groqflow.common.exceptions as exp
import groqflow.common.build as build
import groqflow.common.sdk_helpers as sdk

ORT_BENCHMARKING_SCRIPT = "execute_ort.py"
TRT_BENCHMARKING_SCRIPT = "execute_trt.py"


class BenchmarkPaths:
    def __init__(self, state, device, backend, username=None):
        self.state = state
        self.device = device
        self.backend = backend
        self.username = username

        if backend == "remote" and username is None:
            raise ValueError("username must be set when backend==remote")

    @property
    def output_dir(self):
        if self.backend == "local":
            return os.path.join(
                build.output_dir(self.state.cache_dir, self.state.config.build_name),
                f"{self.device}_benchmark",
            )
        elif self.backend == "remote":
            return os.path.join("/home", self.username, "mlagility_remote_cache")
        elif self.backend == "docker":
            return "/app"
        else:
            raise ValueError(
                f"Got backend {self.backend}, which is not allowed. "
                "The allowed backends are: local, remote, docker"
            )

    @property
    def onnx_dir(self):
        return os.path.join(self.output_dir, "onnxmodel")

    @property
    def onnx_file(self):
        return os.path.join(self.onnx_dir, "model.onnx")

    @property
    def outputs_file(self):
        return os.path.join(self.output_dir, "outputs.json")

    @property
    def errors_file(self):
        return os.path.join(self.output_dir, "errors.npy")


class BenchmarkException(Exception):
    """
    Indicates a failure during benchmarking
    """


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


def load_remote_config(device: str) -> Union[Tuple[str, str], Tuple[None, None]]:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = f"{dir_path}/config.yaml"

    # Create a configuration file if one doesn't exist already
    if not os.path.exists(config_file_path):
        conf: Dict[str, Any] = {
            "remote_machine_nvidia": {"ip": None, "username": None},
            "remote_machine_x86": {"ip": None, "username": None},
            "remote_machine_groqchip": {"ip": None, "username": None},
        }
        with open(config_file_path, "w", encoding="utf8") as outfile:
            yaml.dump(conf, outfile)

    # Return the contents of the configuration file
    config_file = open(config_file_path, encoding="utf8")
    conf = yaml.load(config_file, Loader=yaml.FullLoader)
    return (
        conf[f"remote_machine_{device}"]["ip"],
        conf[f"remote_machine_{device}"]["username"],
    )


def save_remote_config(ip, username, device) -> None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file = open(f"{dir_path}/config.yaml", encoding="utf8")
    conf = yaml.load(config_file, Loader=yaml.FullLoader)
    conf[f"remote_machine_{device}"]["ip"] = ip
    conf[f"remote_machine_{device}"]["username"] = username
    with open(f"{dir_path}/config.yaml", "w", encoding="utf8") as outfile:
        yaml.dump(conf, outfile)


def connect_to_host(ip, username) -> paramiko.SSHClient:
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
        raise BenchmarkException(stderr)

    return stdout, exit_code


def configure_remote(device: str) -> Tuple[str, str]:
    # Load stored values
    ip, username = load_remote_config(device)

    if not all((ip, username)):
        if device == "groqchip":
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

        ip = ip or input(f"{device} instance ASA name (Do not use IP): ")
        username = username or input(f"Username for {ip}: ")

        if not username or not ip:
            raise exp.GroqModelRuntimeError("Username and hostname are required")

        # Store information on yaml file
        save_remote_config(ip, username, device)

    return ip, username


def setup_remote_host(client, device_type: str, output_dir: str) -> None:
    if device_type == "nvidia":
        # Check if at least one NVIDIA GPU is available remotely
        stdout, exit_code = exec_command(client, "lspci | grep -i nvidia")
        if stdout == "" or exit_code == 1:
            msg = "No NVIDIA GPUs available on the remote machine"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = [TRT_BENCHMARKING_SCRIPT]
    elif device_type == "x86":
        # Check if x86_64 CPU is available remotely
        stdout, exit_code = exec_command(client, "uname -i")
        if stdout != "x86_64" or exit_code == 1:
            msg = "Only x86_64 CPUs are supported at this time for benchmarking"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = [ORT_BENCHMARKING_SCRIPT, "setup_ort_env.sh"]
    elif device_type == "groqchip":
        # Check if at least one GroqChip Processor is available remotely
        stdout, exit_code = exec_command(client, "/usr/bin/lspci -n")
        if stdout == "" or exit_code == 1:
            msg = "Failed to run lspci to get GroqChip Processors available"
            raise exp.GroqModelRuntimeError(msg)
        num_chips_available = sdk.get_num_chips_available(stdout.split("\n"))
        if num_chips_available < 1:
            raise exp.GroqModelRuntimeError("No GroqChip Processor(s) found")
        files_to_transfer = ["execute.py"]
    else:
        raise ValueError(
            "Only 'nvidia', 'x86' and 'groqchip' are supported."
            f"But received {device_type}"
        )

    # Transfer common files to host
    exec_command(client, f"mkdir {output_dir}", ignore_error=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with MySFTPClient.from_transport(client.get_transport()) as s:
        for file in files_to_transfer:
            s.put(f"{dir_path}/{file}", f"{output_dir}/{file}")


def setup_local_host(device_type: str, output_dir: str) -> None:
    if device_type == "x86":
        # Check if x86_64 CPU is available locally
        check_device = subprocess.run(
            ["uname", "-i"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        stdout = check_device.stdout.decode().strip()
        if stdout != "x86_64" or check_device.returncode == 1:
            msg = "Only x86_64 CPUs are supported at this time for competitive benchmarking"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = [ORT_BENCHMARKING_SCRIPT, "setup_ort_env.sh"]

    elif device_type == "nvidia":
        # Check if at least one NVIDIA GPU is available locally
        result = subprocess.run(
            ["lspci"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=False,
        )

        if "NVIDIA" not in result.stdout or result.returncode == 1:
            msg = "No NVIDIA GPUs available on the local machine"
            raise exp.GroqModelRuntimeError(msg)
        files_to_transfer = [TRT_BENCHMARKING_SCRIPT]

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
    setup_remote_host(client, device_type=device_type, output_dir=output_dir)

    return client


def execute_groqchip_remotely(
    bringup_topology: bool,
    repetitions: int,
    state: build.State,
) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _, hostname = configure_remote("groqchip")

    # Connect to remote machine and transfer common files
    client = setup_connection("groqchip")

    # Transfer iop and inputs file
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
        msg = """
        Failed to execute GroqChip Processor(s) remotely.
        """
        raise exp.GroqModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.get(remote_outputs_file, state.outputs_file)
        s.get(remote_latency_file, state.latency_file)
        s.remove(remote_outputs_file)
        s.remove(remote_latency_file)


def execute_gpu_remotely(state: build.State, device: str, iterations: int) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _ip, username = configure_remote(device)

    # Setup remote execution folders to save outputs/ errors
    remote_paths = BenchmarkPaths(state, device, "remote", username)
    local_paths = BenchmarkPaths(state, device, "local")
    docker_paths = BenchmarkPaths(state, device, "docker")
    os.makedirs(local_paths.output_dir, exist_ok=True)

    # Connect to remote machine and transfer common files
    client = setup_connection(device_type=device, output_dir=remote_paths.output_dir)

    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir(remote_paths.onnx_dir)
        s.put(state.converted_onnx_file, remote_paths.onnx_file)

    # Run benchmarking script
    _, exit_code = exec_command(
        client,
        f"/usr/bin/python3 {remote_paths.output_dir}/{TRT_BENCHMARKING_SCRIPT} "
        f"{remote_paths.output_dir} {docker_paths.onnx_file} {remote_paths.outputs_file} "
        f"{remote_paths.errors_file} {iterations}",
    )
    if exit_code == 1:
        msg = """
        Failed to execute GPU(s) remotely.
        """
        raise exp.GroqModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        try:
            s.get(
                remote_paths.outputs_file,
                local_paths.outputs_file,
            )
            s.get(
                remote_paths.errors_file,
                local_paths.errors_file,
            )
            s.remove(remote_paths.outputs_file)
            s.remove(remote_paths.errors_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Output/ error files not found! Please make sure your remote GPU machine is"
                "turned ON and has all the required dependencies installed"
                f"Full exception: {e}"
            )

    if not os.path.isfile(local_paths.outputs_file):
        raise BenchmarkException(
            "No benchmarking outputs file found after benchmarking run."
            "Sorry we don't have more information."
        )


def execute_gpu_locally(state: build.State, device: str, iterations: int) -> None:
    """
    Execute Model on the local GPU
    """

    # Setup local execution folders to save outputs/ errors
    local_paths = BenchmarkPaths(state, device, "local")
    docker_paths = BenchmarkPaths(state, device, "docker")

    setup_local_host(device_type="nvidia", output_dir=local_paths.output_dir)

    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    os.makedirs(local_paths.onnx_dir)
    shutil.copy(state.converted_onnx_file, local_paths.onnx_file)

    # Check if docker is installed
    docker_location = shutil.which("docker")
    if not docker_location:
        raise ValueError(
            "'docker' installation not found. Please install docker>=20.10"
        )

    # Check if python is installed
    python_location = shutil.which("python")
    if not python_location:
        raise ValueError("'python' installation not found. Please install python>=3.8")

    run_benchmark = subprocess.Popen(
        [
            python_location,
            os.path.join(local_paths.output_dir, TRT_BENCHMARKING_SCRIPT),
            local_paths.output_dir,
            docker_paths.onnx_file,
            local_paths.outputs_file,
            local_paths.errors_file,
            str(iterations),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = run_benchmark.communicate()
    if run_benchmark.returncode != 0:
        raise BenchmarkException(
            "Error: Failure to run model using TensorRT - " f"{stderr.decode().strip()}"
        )

    if not os.path.isfile(local_paths.outputs_file):
        raise BenchmarkException(
            "No benchmarking outputs file found after benchmarking run."
            "Sorry we don't have more information."
        )


def execute_cpu_remotely(state: build.State, device: str, iterations: int) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _ip, username = configure_remote(device)

    # Setup remote execution folders to save outputs/ errors
    remote_paths = BenchmarkPaths(state, device, "remote", username)
    local_paths = BenchmarkPaths(state, device, "local")
    os.makedirs(local_paths.output_dir, exist_ok=True)

    # Connect to remote machine and transfer common files
    client = setup_connection(device_type=device, output_dir=remote_paths.output_dir)

    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir(remote_paths.onnx_dir)
        s.put(state.converted_onnx_file, remote_paths.onnx_file)

    # Check if conda is installed on the remote machine
    conda_location, exit_code = exec_command(client, f"ls /home/{username}")
    if "miniconda3" not in conda_location:
        raise ValueError(
            f"conda installation not found in /home/{username}. Please install miniconda3"
        )
    # TODO: Remove requirement that conda has to be installed on the /home/user
    conda_src = f"/home/{username}"

    # Run benchmarking script
    env_name = "mlagility-onnxruntime-env"
    exec_command(
        client,
        f"bash {remote_paths.output_dir}/setup_ort_env.sh {env_name} {conda_src}",
        ignore_error=True,
    )

    _, exit_code = exec_command(
        client,
        f"/home/{username}/miniconda3/envs/{env_name}/bin/python "
        f"{os.path.join(remote_paths.output_dir, ORT_BENCHMARKING_SCRIPT)} "
        f"{remote_paths.output_dir} {remote_paths.onnx_file} {remote_paths.outputs_file} "
        f"{remote_paths.errors_file} {iterations}",
    )
    if exit_code == 1:
        msg = """
        Failed to execute CPU(s) remotely.
        """
        raise exp.GroqModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        try:
            s.get(remote_paths.outputs_file, local_paths.outputs_file)
            s.get(remote_paths.errors_file, local_paths.errors_file)
            s.remove(remote_paths.outputs_file)
            s.remove(remote_paths.errors_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Output/ error files not found! Please make sure your remote CPU machine is"
                "turned ON and has all the required dependencies installed"
                f"Full exception: {e}"
            )

    if not os.path.isfile(local_paths.outputs_file):
        raise BenchmarkException(
            "No benchmarking outputs file found after benchmarking run."
            "Sorry we don't have more information."
        )


def execute_cpu_locally(state: build.State, device: str, iterations: int) -> None:
    """
    Execute Model on the local CPU
    """

    # Setup local execution folders to save outputs/ errors
    local_paths = BenchmarkPaths(state, device, "local")

    setup_local_host(device_type=device, output_dir=local_paths.output_dir)

    # Check if ONNX file has been generated
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    os.makedirs(local_paths.onnx_dir)
    shutil.copy(state.converted_onnx_file, local_paths.onnx_file)

    # Check if conda is installed
    conda_location = shutil.which("conda")
    if not conda_location:
        raise ValueError("conda installation not found.")
    conda_src = conda_location.split("condabin")[0]

    # Create/ update local conda environment for CPU benchmarking
    env_name = "mlagility-onnxruntime-env"

    def available_conda_envs():
        result = subprocess.run(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.stdout.decode()

    if env_name not in available_conda_envs():
        print(
            "Creating a new conda environment to benchmark with CPU. This takes a few seconds..."
        )

        setup_env = subprocess.Popen(
            [
                "bash",
                os.path.join(local_paths.output_dir, "setup_ort_env.sh"),
                env_name,
                conda_src,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, stderr = setup_env.communicate()
        if setup_env.returncode != 0:
            print(
                f"Error: Failure to setup conda environment - {stderr.decode().strip()}"
            )

    # Run the benchmark
    run_benchmark = subprocess.Popen(
        [
            f"{conda_src}envs/{env_name}/bin/python",
            os.path.join(local_paths.output_dir, ORT_BENCHMARKING_SCRIPT),
            local_paths.output_dir,
            local_paths.onnx_file,
            local_paths.outputs_file,
            local_paths.errors_file,
            str(iterations),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = run_benchmark.communicate()
    if run_benchmark.returncode != 0:
        raise BenchmarkException(
            f"Error: Failure to run model using onnxruntime - {stderr.decode().strip()}"
        )

    if not os.path.isfile(local_paths.outputs_file):
        raise BenchmarkException(
            "No benchmarking outputs file found after benchmarking run."
            "Sorry we don't have more information."
        )
