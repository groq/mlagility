import os
import subprocess
import pathlib
import time
import getpass
from typing import List, Optional
import mlagility.common.filesystem as filesystem


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
    device: str,
    rebuild: Optional[str] = None,
    groq_compiler_flags: Optional[List[str]] = None,
    groq_assembler_flags: Optional[List[str]] = None,
    groq_num_chips: Optional[int] = None,
    groqview: Optional[bool] = None,
    runtimes: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    analyze_only: Optional[bool] = None,
    build_only: Optional[bool] = None,
    lean_cache: Optional[bool] = None,
    working_dir: str = os.getcwd(),
    ml_cache_dir: Optional[str] = os.environ.get("SLURM_ML_CACHE"),
    max_jobs: int = 50,
):

    # Convert args to strings
    cache_dir_str = value_arg(cache_dir, "--cache-dir")
    rebuild_str = value_arg(rebuild, "--rebuild")
    compiler_flags_str = list_arg(groq_compiler_flags, "--groq-compiler-flags")
    assembler_flags_str = list_arg(groq_assembler_flags, "--groq-assembler-flags")
    num_chips_str = value_arg(groq_num_chips, "--groq-num-chips")
    groqview_str = bool_arg(groqview, "--groqview")
    runtimes_str = list_arg(runtimes, "--runtimes")
    max_depth_str = value_arg(max_depth, "--max-depth")
    analyze_only_str = bool_arg(analyze_only, "--analyze-only")
    build_only_str = bool_arg(build_only, "--build-only")
    lean_cache_str = bool_arg(lean_cache, "--lean-cache")

    args = (
        f"{op} {script} --device {device} {cache_dir_str}{rebuild_str}"
        f"{compiler_flags_str}{assembler_flags_str}{num_chips_str}{groqview_str}"
        f"{runtimes_str}{max_depth_str}{analyze_only_str}"
        f"{build_only_str}{lean_cache_str}"
    )

    # Remove the .py extension from the build name
    job_name = filesystem.clean_script_name(script)

    while len(jobs_in_queue()) >= max_jobs:
        print(
            f"Waiting: Your number of jobs running ({len(jobs_in_queue())}) "
            "matches or exceeds the maximum "
            f"concurrent jobs allowed ({max_jobs}). The jobs in queue are: {jobs_in_queue()}"
        )
        time.sleep(5)

    shell_script = os.path.join(pathlib.Path(__file__).parent.resolve(), "run_slurm.sh")

    slurm_command = ["sbatch", "-c", "1"]
    if os.environ.get("MLAGILITY_SLURM_USE_DEFAULT_MEMORY") != "True":
        slurm_command.append("--mem=128000")
    slurm_command.extend(
        [
            "--time=00-02:00:00",  # days-hh:mm:ss"
            f"--job-name={job_name}",
            shell_script,
            "benchit",
            args,
            working_dir,
        ]
    )
    if ml_cache_dir is not None:
        slurm_command.append(ml_cache_dir)

    print(f"Submitting job {job_name} to Slurm")
    subprocess.check_call(slurm_command)


def update_database_builds(cache_dir, input_scripts):
    # In the parent process, add all builds to the cache database
    if os.environ.get("USING_SLURM") != "TRUE":
        db = filesystem.CacheDatabase(cache_dir)
        for script in input_scripts:
            builds, script_name = filesystem.get_builds_from_script(cache_dir, script)
            for build in builds:
                db.add_build(script_name, build)
