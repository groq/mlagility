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


def run_benchit(
    op: str,
    script: str,
    cache_dir: str,
    search_dir: Optional[str] = None,
    rebuild: Optional[str] = None,
    compiler_flags: Optional[List[str]] = None,
    assembler_flags: Optional[List[str]] = None,
    num_chips: Optional[int] = None,
    groqview: Optional[bool] = None,
    devices: Optional[List[str]] = None,
    ip: Optional[str] = None,
    max_depth: Optional[int] = None,
    analyze_only: Optional[bool] = None,
    build_only: Optional[bool] = None,
    lean_cache: Optional[bool] = None,
    working_dir: str = os.getcwd(),
    ml_cache_dir: Optional[str] = None,
    max_jobs: int = 50,
):

    # Convert args to strings
    cache_dir_str = value_arg(cache_dir, "--cache-dir")
    search_dir_str = value_arg(search_dir, "--search-dir")
    rebuild_str = value_arg(rebuild, "--rebuild")
    compiler_flags_str = list_arg(compiler_flags, "--compiler-flags")
    assembler_flags_str = list_arg(assembler_flags, "--assembler-flags")
    num_chips_str = value_arg(num_chips, "--num-chips")
    groqview_str = bool_arg(groqview, "--groqview")
    devices_str = list_arg(devices, "--devices")
    ip_str = value_arg(ip, "--ip")
    max_depth_str = value_arg(max_depth, "--max-depth")
    analyze_only_str = bool_arg(analyze_only, "--analyze-only")
    build_only_str = bool_arg(build_only, "--build-only")
    lean_cache_str = bool_arg(lean_cache, "--lean_cache")

    args = (
        f"{op} {script} {cache_dir_str}{search_dir_str}{rebuild_str}"
        f"{compiler_flags_str}{assembler_flags_str}{num_chips_str}{groqview_str}"
        f"{devices_str}{ip_str}{max_depth_str}{analyze_only_str}"
        f"{build_only_str}{lean_cache_str}"
    )

    # Remove the .py extension from the build name
    job_name = script.split("/")[-1].split(".")[0]

    while len(jobs_in_queue()) >= max_jobs:
        print(
            f"Waiting: Your number of jobs running ({len(jobs_in_queue())}) "
            "matches or exceeds the maximum "
            f"concurrent jobs allowed ({max_jobs}). The jobs in queue are: {jobs_in_queue()}"
        )
        time.sleep(5)

    shell_script = os.path.join(pathlib.Path(__file__).parent.resolve(), "run_slurm.sh")

    slurm_command = [
        "sbatch",
        "-c",
        "1",
        "--mem=128000",
        "--time=00-02:00:00",  # days-hh:mm:ss"
        f"--job-name={job_name}",
        shell_script,
        "benchit",
        args,
        working_dir,
    ]
    if ml_cache_dir is not None:
        slurm_command.append(ml_cache_dir)

    print(f"Submitting job {job_name} to Slurm")
    subprocess.check_call(slurm_command)
