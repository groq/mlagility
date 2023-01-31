import os
import subprocess
import pathlib
import time
import getpass
from typing import List, Optional


def jobs_in_queue(job_name=None) -> List[str]:
    """Return the set of slurm jobs that are currently pending/running"""
    user = getpass.getuser()
    if job_name is None:
        output = subprocess.check_output(["squeue", "-u", user])
    else:
        output = subprocess.check_output(["squeue", "-u", user, "--name", job_name])
    output = output.split(b"\n")
    output = [s.decode("utf").split() for s in output]

    # Remove headers
    output.pop(0)

    # Remove empty line at the end
    output.pop(-1)

    # Get just the job names
    if len(output) > 0:
        name_index_in_squeue = 2
        output = [s[name_index_in_squeue] for s in output]

    return output


def job_name(build_name) -> str:
    return f"{build_name}_groqit"


def run(
    groqit_script: str,
    args: str,
    build_name: str,
    cache_dir: str,
):
    """
    Run a GroqFlow job on Slurm

    args:
        groqit_script: an executable script that knows how to take the following arguments.
            Most of the time this will be `groqit-util`, however you can provide your own
            custom executable.
        args: command line arguments passed to the groqit_script
        build_name: name of the build
        cache_dir: location of the GroqFlow build cache

    """

    max_jobs = 30

    slurm_log_file = os.path.join(cache_dir, f"{build_name}_slurm_log.txt")

    while len(jobs_in_queue()) >= max_jobs:
        print(
            f"Waiting: Your number of jobs running ({len(jobs_in_queue())}) "
            "matches or exceeds the maximum "
            f"concurrent jobs allowed ({max_jobs}). The jobs in queue are: {jobs_in_queue()}"
        )
        time.sleep(5)

    job = job_name(build_name)

    shell_script = os.path.join(pathlib.Path(__file__).parent.resolve(), "run_slurm.sh")

    slurm_command = [
        "sbatch",
        "-c",
        "1",
        "--mem=128000",
        f"--output={slurm_log_file}",
        f"--job-name={job}",
        shell_script,
        groqit_script,
        args,
    ]

    print(f"Submitting build {build_name} to Slurm")

    subprocess.check_call(slurm_command)


def list_arg(values, flag):
    if values is not None:
        result = " ".join(values)
        result = f" {flag} {result}"
    else:
        result = ""

    return result


def value_arg(value, flag):
    if value is not None:
        result = f" {flag} {value}"
    else:
        result = ""

    return result


def bool_arg(value, flag):
    if value:
        result = f" {flag}"
    else:
        result = ""

    return result


def run_autogroq(
    op: str,
    search_dir: str,
    script: str,
    cache_dir: str,
    rebuild: str,
    compiler_flags: Optional[List[str]],
    assembler_flags: Optional[List[str]],
    num_chips: int,
    groqview: bool,
    devices: Optional[List[str]],
    runtimes: Optional[List[str]],
    ip: str,
    max_depth: int,
    analyze_only: bool,
    build_only: bool,
):

    compiler_flags_str = list_arg(compiler_flags, "--compiler-flags")
    assembler_flags_str = list_arg(assembler_flags, "--assembler-flags")

    num_chips_str = value_arg(num_chips, "--num-chips")
    groqview_str = bool_arg(groqview, "--groqview")

    devices_str = list_arg(devices, "--devices")
    runtimes_str = list_arg(runtimes, "--runtimes")

    ip_str = value_arg(ip, "--ip")
    max_depth_str = value_arg(max_depth, "--max-depth")
    analyze_only_str = bool_arg(analyze_only, "--analyze-only")
    build_only_str = bool_arg(build_only, "--build-only")

    args = (
        f"{op} -s {search_dir} -i {script} -d {cache_dir} "
        f"--rebuild {rebuild}"
        f"{compiler_flags_str}{assembler_flags_str}{num_chips_str}{groqview_str}"
        f"{devices_str}{runtimes_str}{ip_str}{max_depth_str}{analyze_only_str}{build_only_str}"
    )

    # Remove the .py extension from the build name
    build_name = script.split(".")[0]

    run(
        groqit_script="groqit",
        args=args,
        build_name=build_name,
        cache_dir=cache_dir,
    )
