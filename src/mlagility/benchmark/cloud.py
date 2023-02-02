import os
import sys
import subprocess
from typing import Tuple, Union, Dict, Any
from stat import S_ISDIR
import yaml
import paramiko
import groqflow.common.exceptions as exp
import groqflow.common.build as build


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
        # TODO (ramkrishna2910): Enabling localhost execution will be handled
        # in a separate MR (Issue #5)
        print(
            "User is responsible for ensuring the remote server has python>=3.8 \
            and docker>=20.10 installed"
        )
        print("Provide your instance IP and hostname below:")

        ip = ip or input(f"{accelerator} instance ASA name (Do not use IP): ")
        username = username or input(f"Username for {ip}: ")

        if not username or not ip:
            raise exp.GroqModelRuntimeError("Username and hostname are required")

        # Store information on yaml file
        save_remote_config(ip, username, accelerator)

    return ip, username


def setup_gpu_host(client) -> None:
    # Check if at least one NVIDIA GPU is available remotely
    stdout, exit_code = exec_command(client, "lspci | grep -i nvidia")
    if stdout == "" or exit_code == 1:
        msg = "No NVIDIA GPUs available on the remote machine"
        raise exp.GroqModelRuntimeError(msg)

    # Transfer common files to host
    exec_command(client, "mkdir mlagility_remote_cache", ignore_error=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.put(f"{dir_path}/execute-gpu.py", "mlagility_remote_cache/execute-gpu.py")


def setup_cpu_host(client) -> None:
    # Check if x86_64 CPU is available remotely
    stdout, exit_code = exec_command(client, "uname -i")
    if stdout != "x86_64" or exit_code == 1:
        msg = "Only x86_64 CPUs are supported at this time for competitive benchmarking"
        raise exp.GroqModelRuntimeError(msg)

    # Transfer common files to host
    exec_command(client, "mkdir mlagility_remote_cache", ignore_error=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.put(f"{dir_path}/execute-cpu.py", "mlagility_remote_cache/execute-cpu.py")
        s.put(f"{dir_path}/setup_ort_env.sh", "mlagility_remote_cache/setup_ort_env.sh")


def setup_connection(accelerator: str) -> paramiko.SSHClient:
    # Setup authentication scheme if needed
    ip, username = configure_remote(accelerator)

    # Connect to host
    client = connect_to_host(ip, username)

    if accelerator == "gpu":
        # Check for GPU and transfer files
        setup_gpu_host(client)
    elif accelerator == "cpu":
        # Check for CPU and transfer files
        setup_cpu_host(client)
    else:
        raise ValueError(
            f"Only 'cpu' and 'gpu' are supported, but received {accelerator}"
        )

    return client


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

    # Connect to remote machine and transfer common files
    client = setup_connection("gpu")

    print("Transferring model file...")
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir("mlagility_remote_cache/onnxmodel")
        s.put(state.converted_onnx_file, "mlagility_remote_cache/onnxmodel/model.onnx")

    # Run benchmarking script
    output_dir = "mlagility_remote_cache"
    remote_outputs_file = "mlagility_remote_cache/outputs.txt"
    remote_errors_file = "mlagility_remote_cache/errors.txt"
    print("Running benchmarking script...")
    _, exit_code = exec_command(
        client,
        (
            f"/usr/bin/python3 mlagility_remote_cache/execute-gpu.py "
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

    # Connect to remote machine and transfer common files
    client = setup_connection("cpu")

    print("Transferring model file...")
    if not os.path.exists(state.converted_onnx_file):
        msg = "Model file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir("mlagility_remote_cache/onnxmodel")
        s.put(state.converted_onnx_file, "mlagility_remote_cache/onnxmodel/model.onnx")

    # Run benchmarking script
    output_dir = "mlagility_remote_cache"
    remote_outputs_file = "mlagility_remote_cache/outputs.txt"
    remote_errors_file = "mlagility_remote_cache/errors.txt"
    env_name = "ort_env"
    exec_command(
        client, "bash mlagility_remote_cache/setup_ort_env.sh", ignore_error=True
    )

    print("Running benchmarking script...")
    _, exit_code = exec_command(
        client,
        (
            f"/home/{username}/miniconda3/envs/{env_name}/bin/python mlagility_remote_cache/"
            "execute-cpu.py "
            f"{output_dir} {remote_outputs_file} {remote_errors_file} {iterations} {username}"
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
    # Stop redirecting stdout
    sys.stdout = sys.stdout.terminal
