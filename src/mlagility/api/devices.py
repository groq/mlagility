import os
import sys
import subprocess
from typing import Tuple, Union, Dict, Any
from stat import S_ISDIR
import shutil
import yaml
import paramiko
import onnxflow.common.exceptions as exp
import onnxflow.common.build as build

ORT_BENCHMARKING_SCRIPT = "setup_ort.py"
ORT_EXECUTION_SCRIPT = "run_ort_model.py"
TRT_BENCHMARKING_SCRIPT = "execute_trt.py"
SUPPORTED_DEVICES = {
    "x86": ["ort", "torch-eager", "torch-compiled"],
    "groq": ["groq"],
    "nvidia": ["trt"],
}
DEFAULT_RUNTIME = 0


class BenchmarkPaths:
    def __init__(self, cache_dir, build_name, device, backend, username=None):
        self.cache_dir = cache_dir
        self.build_name = build_name
        self.device = device
        self.backend = backend
        self.username = username

        if backend == "remote" and username is None:
            raise ValueError("username must be set when backend==remote")

    @property
    def output_dir(self):
        if self.backend == "local":
            return os.path.join(
                build.output_dir(self.cache_dir, self.build_name),
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
    if exit_code != 0 and not ignore_error:
        raise BenchmarkException(stderr)

    return stdout, exit_code


def configure_remote(device: str) -> Tuple[str, str]:
    # Load stored values
    ip, username = load_remote_config(device)

    if not all((ip, username)):
        if device == "groq":
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
            raise exp.ModelRuntimeError("Username and hostname are required")

        # Store information on yaml file
        save_remote_config(ip, username, device)

    return ip, username


def setup_remote_host(client, device_type: str, output_dir: str) -> None:
    if device_type == "nvidia":
        # Check if at least one NVIDIA GPU is available remotely
        stdout, exit_code = exec_command(client, "lspci | grep -i nvidia")
        if stdout == "" or exit_code == 1:
            msg = "No NVIDIA GPUs available on the remote machine"
            raise exp.ModelRuntimeError(msg)
        files_to_transfer = [TRT_BENCHMARKING_SCRIPT]
    elif device_type == "x86":
        # Check if x86_64 CPU is available remotely
        stdout, exit_code = exec_command(client, "uname -m")
        if stdout != "x86_64" or exit_code == 1:
            msg = "Only x86_64 CPUs are supported at this time for benchmarking"
            raise exp.ModelRuntimeError(msg)
        files_to_transfer = [
            ORT_BENCHMARKING_SCRIPT,
            "Dockerfile",
            ORT_EXECUTION_SCRIPT,
        ]
    elif device_type == "groq":
        # pylint: disable=import-error
        import groqflow.common.sdk_helpers as sdk
        import groqflow.common.exceptions as groq_exp

        # Check if at least one GroqChip Processor is available remotely
        stdout, exit_code = exec_command(client, "/usr/bin/lspci -n")
        if stdout == "" or exit_code == 1:
            msg = "Failed to run lspci to get GroqChip Processors available"
            raise groq_exp.GroqModelRuntimeError(msg)
        num_chips_available = sdk.get_num_chips_available(stdout.split("\n"))
        if num_chips_available < 1:
            raise groq_exp.GroqModelRuntimeError("No GroqChip Processor(s) found")
        files_to_transfer = ["execute.py"]
    else:
        raise ValueError(
            "Only 'nvidia', 'x86', and 'groq' are supported."
            f"But received {device_type}"
        )

    # Transfer common files to host
    exec_command(client, f"mkdir {output_dir}", ignore_error=True)
    if device_type == "groq":
        # pylint: disable=import-error
        from groqflow.groqmodel import groqmodel

        dir_path = os.path.dirname(os.path.realpath(groqmodel.__file__))
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
    with MySFTPClient.from_transport(client.get_transport()) as s:
        for file in files_to_transfer:
            s.put(f"{dir_path}/{file}", f"{output_dir}/{file}")


def setup_local_host(device_type: str, output_dir: str) -> None:
    if device_type == "x86":
        # Check if x86_64 CPU is available locally
        check_device = subprocess.run(
            ["uname", "-m"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
        stdout = check_device.stdout.decode().strip()
        if stdout != "x86_64" or check_device.returncode == 1:
            msg = "Only x86_64 CPUs are supported at this time for competitive benchmarking"
            raise exp.ModelRuntimeError(msg)
        files_to_transfer = [
            ORT_BENCHMARKING_SCRIPT,
            "Dockerfile",
            ORT_EXECUTION_SCRIPT,
        ]

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
            raise exp.ModelRuntimeError(msg)
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
    _, hostname = configure_remote("groq")

    # Connect to remote machine and transfer common files
    client = setup_connection("groq")

    # Transfer iop and inputs file
    if not os.path.exists(state.execution_inputs_file):
        msg = "Model input file not found"
        raise exp.ModelRuntimeError(msg)

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
        raise exp.ModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.get(remote_outputs_file, state.outputs_file)
        s.get(remote_latency_file, state.latency_file)
        s.remove(remote_outputs_file)
        s.remove(remote_latency_file)


def execute_trt_remotely(
    cache_dir: str, build_name: str, device: str, iterations: int
) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _ip, username = configure_remote(device)

    # Setup remote execution folders to save outputs/ errors
    cache_dir = os.path.abspath(cache_dir)
    remote_paths = BenchmarkPaths(cache_dir, build_name, device, "remote", username)
    local_paths = BenchmarkPaths(cache_dir, build_name, device, "local")
    docker_paths = BenchmarkPaths(cache_dir, build_name, device, "docker")
    os.makedirs(local_paths.output_dir, exist_ok=True)

    # Connect to remote machine and transfer common files
    client = setup_connection(device_type=device, output_dir=remote_paths.output_dir)

    state = build.load_state(cache_dir, build_name)
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.ModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir(remote_paths.onnx_dir)
        s.put(state.converted_onnx_file, remote_paths.onnx_file)

    # Run benchmarking script
    _, exit_code = exec_command(
        client,
        f"/usr/bin/python3 {remote_paths.output_dir}/{TRT_BENCHMARKING_SCRIPT} "
        f"--output-dir {remote_paths.output_dir} --onnx-file {docker_paths.onnx_file} "
        f"--outputs-file {remote_paths.outputs_file} "
        f"--errors-file {remote_paths.errors_file} --iterations {iterations}",
    )
    if exit_code == 1:
        msg = """
        Failed to execute GPU(s) remotely.
        """
        raise exp.ModelRuntimeError(msg)

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
            "No benchmarking outputs file found after benchmarking run. "
            "Sorry we don't have more information."
        )


def execute_trt_locally(
    cache_dir: str, build_name: str, device: str, iterations: int
) -> None:
    """
    Execute Model on the local GPU
    """

    # Setup local execution folders to save outputs/ errors
    cache_dir = os.path.abspath(cache_dir)
    local_paths = BenchmarkPaths(cache_dir, build_name, device, "local")
    docker_paths = BenchmarkPaths(cache_dir, build_name, device, "docker")

    setup_local_host(device_type="nvidia", output_dir=local_paths.output_dir)

    state = build.load_state(cache_dir, build_name)
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.ModelRuntimeError(msg)

    os.makedirs(local_paths.onnx_dir)
    shutil.copy(state.converted_onnx_file, local_paths.onnx_file)

    # Check if docker is installed
    docker_location = shutil.which("docker")
    if not docker_location:
        raise ValueError(
            "'docker' installation not found. Please install docker>=20.10"
        )

    # Check if python is installed
    python_location = sys.executable
    if not python_location:
        raise ValueError("'python' installation not found. Please install python>=3.8")

    run_benchmark = subprocess.Popen(
        [
            python_location,
            os.path.join(local_paths.output_dir, TRT_BENCHMARKING_SCRIPT),
            "--output-dir",
            local_paths.output_dir,
            "--onnx-file",
            docker_paths.onnx_file,
            "--outputs-file",
            local_paths.outputs_file,
            "--errors-file",
            local_paths.errors_file,
            "--iterations",
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
            "No benchmarking outputs file found after benchmarking run. "
            "Sorry we don't have more information."
        )


def execute_ort_remotely(
    cache_dir: str, build_name: str, device: str, iterations: int
) -> None:
    """
    Execute Model on the remote machine
    """

    # Ask the user for credentials if needed
    _ip, username = configure_remote(device)

    # Setup remote execution folders to save outputs/ errors
    cache_dir = os.path.abspath(cache_dir)
    remote_paths = BenchmarkPaths(cache_dir, build_name, device, "remote", username)
    local_paths = BenchmarkPaths(cache_dir, build_name, device, "local")
    docker_paths = BenchmarkPaths(cache_dir, build_name, device, "docker")
    os.makedirs(local_paths.output_dir, exist_ok=True)

    # Connect to remote machine and transfer common files
    client = setup_connection(device_type=device, output_dir=remote_paths.output_dir)

    state = build.load_state(cache_dir, build_name)
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.ModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir(remote_paths.onnx_dir)
        s.put(state.converted_onnx_file, remote_paths.onnx_file)

    _, exit_code = exec_command(
        client,
        f"/usr/bin/python3 "
        f"{os.path.join(remote_paths.output_dir, ORT_BENCHMARKING_SCRIPT)} "
        f"--output-dir {remote_paths.output_dir} "
        f"--onnx-file {docker_paths.onnx_file} "
        f"--outputs-file {remote_paths.outputs_file} "
        f"--iterations {iterations}",
    )

    if exit_code == 1:
        msg = """
        Failed to execute model on ORT container.
        """
        raise exp.ModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        try:
            s.get(remote_paths.outputs_file, local_paths.outputs_file)
            s.remove(remote_paths.outputs_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Output/ error files not found! Please make sure your remote CPU machine is"
                "turned ON and has all the required dependencies installed"
                f"Full exception: {e}"
            )

    if not os.path.isfile(local_paths.outputs_file):
        raise BenchmarkException(
            "No benchmarking outputs file found after benchmarking run. "
            "Sorry we don't have more information."
        )


def execute_ort_locally(
    cache_dir: str, build_name: str, device: str, iterations: int
) -> None:
    """
    Execute Model on the local ORT
    """

    # Setup local execution folders to save outputs/ errors
    cache_dir = os.path.abspath(cache_dir)
    local_paths = BenchmarkPaths(cache_dir, build_name, device, "local")
    docker_paths = BenchmarkPaths(cache_dir, build_name, device, "docker")

    setup_local_host(device_type=device, output_dir=local_paths.output_dir)

    # Check if ONNX file has been generated
    state = build.load_state(cache_dir, build_name)
    if not os.path.exists(state.intermediate_results[0]):
        msg = "Model file not found"
        raise exp.ModelRuntimeError(msg)

    os.makedirs(local_paths.onnx_dir)
    shutil.copy(state.intermediate_results[0], local_paths.onnx_file)

    # Check if docker and python are installed on the local machine
    docker_location = shutil.which("docker")
    if not docker_location:
        raise ValueError("Docker installation not found. Please install Docker>=20.10")

    python_location = sys.executable
    if not python_location:
        raise ValueError("'python' installation not found. Please install python>=3.8")

    run_benchmark = subprocess.Popen(
        [
            python_location,
            os.path.join(local_paths.output_dir, ORT_BENCHMARKING_SCRIPT),
            "--output-dir",
            local_paths.output_dir,
            "--onnx-file",
            docker_paths.onnx_file,
            "--outputs-file",
            local_paths.outputs_file,
            "--iterations",
            str(iterations),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = run_benchmark.communicate()
    if run_benchmark.returncode != 0:
        raise BenchmarkException(
            "Error: Failure to run model using ORT - " f"{stderr.decode().strip()}"
        )

    if not os.path.isfile(local_paths.outputs_file):
        raise BenchmarkException(
            "No benchmarking outputs file found after benchmarking run. "
            "Sorry we don't have more information."
        )
