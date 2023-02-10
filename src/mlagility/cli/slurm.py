import os
import subprocess
import pathlib
import time
import getpass
from typing import List


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


def run(
    groqit_script: str,
    args: str,
    job_name: str,
    working_directory: str,
    conda_directory: str,
):
    """
    Run a GroqFlow job on Slurm

    args:
        groqit_script: an executable script that knows how to take the following arguments.
            Most of the time this will be `groqit-util`, however you can provide your own
            custom executable.
        args: command line arguments passed to the groqit_script
        build_name: name of the build

    """

    max_jobs = 50

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
        groqit_script,
        args,
        working_directory,
        conda_directory,
    ]

    print(f"Submitting job {job_name} to Slurm")
    subprocess.check_call(slurm_command)
