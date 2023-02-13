import time
import os
import types
import importlib.machinery
from typing import Tuple, List, Dict, Optional, Union
import groqflow.common.printing as printing
import groqflow.common.exceptions as exceptions
from groqflow.justgroqit.stage import Sequence
import mlagility.cli.slurm as slurm
import mlagility.common.filesystem as filesystem
from mlagility.analysis.analysis import evaluate_script, TracerArgs, Action
from mlagility.analysis.util import ModelInfo


def decode_script_name(input: str) -> Tuple[str, List[str]]:
    # Parse the targets out of the script name
    # Targets use the format:
    #   script_name.py::target0,target1,...,targetN
    decoded_name = input.split("::")
    script_name = decoded_name[0]

    if len(decoded_name) == 2:
        targets = decoded_name[1].split(",")
    elif len(decoded_name) == 1:
        targets = []
    else:
        raise ValueError(
            "Each script input to benchit should have either 0 or 1 '::' in it."
            f"However, {script_name} was received."
        )

    return script_name, targets


def benchmark_script(
    search_dir: str = os.getcwd(),
    input_script: str = None,
    benchmark_all: bool = False,
    use_slurm: bool = False,
    lean_cache: bool = False,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    rebuild: Optional[str] = None,
    devices: List[str] = None,
    backend: str = "local",
    analyze_only: bool = False,
    build_only: bool = False,
    script_args: str = "",
    max_depth: int = 0,
    sequence: Union[str, Sequence] = None,
    groq_compiler_flags: Optional[List[str]] = None,
    groq_assembler_flags: Optional[List[str]] = None,
    groq_num_chips: Optional[int] = None,
    groqview: bool = False,
):

    # Import the sequence file to get a custom sequence, if the user provided
    # one
    if sequence is not None:
        if use_slurm:
            # The slurm node will need to load a sequence file
            if not isinstance(sequence, str):
                raise ValueError(
                    "The 'sequence' arg must be a str (path to a sequence file) "
                    "when use_slurm=True."
                )
            custom_sequence = sequence
        elif isinstance(sequence, str):
            loader = importlib.machinery.SourceFileLoader("a_b", sequence)
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            # pylint: disable = no-member
            custom_sequence = mod.get_sequence()
        elif isinstance(sequence, Sequence):
            custom_sequence = sequence
        else:
            raise ValueError(
                "The 'sequence' arg must be a str (path to a sequence file) "
                "or an instance of the Sequence class."
            )
    else:
        custom_sequence = None

    if devices is None:
        devices = ["x86"]

    # Force the user to specify a legal cache dir in NFS if they are using slurm
    if not os.path.expanduser(cache_dir).startswith("/net") and use_slurm:
        raise ValueError(
            "You must specify a --cache-dir in `/net` when using groqit-util with Slurm, "
            f"however your current --cache-dir is set to {os.path.expanduser(cache_dir)}"
        )

    # Get a specific list of models to process
    available_scripts = filesystem.get_available_scripts(search_dir)

    # Filter based on the model names provided by the user
    if benchmark_all:
        scripts = [os.path.join(search_dir, script) for script in available_scripts]
    else:
        user_script_path = os.path.join(search_dir, input_script)

        # Ignore everything after the ':' symbol, if there is one
        clean_user_script_path = user_script_path.split(":")[0]

        # Validate that the script exists
        if os.path.exists(clean_user_script_path):
            scripts = [user_script_path]
        else:
            raise exceptions.GroqitArgError(
                f"Script could not be found: {user_script_path}"
            )

    # Get absolute path of scripts
    scripts = [os.path.abspath(s) for s in scripts]

    # Decode benchit args into TracerArgs flags
    if analyze_only:
        actions = [
            Action.ANALYZE,
        ]
    elif build_only:
        actions = [
            Action.ANALYZE,
            Action.BUILD,
        ]
    else:
        actions = [
            Action.ANALYZE,
            Action.BUILD,
            Action.BENCHMARK,
        ]

    if use_slurm:
        if backend != "local":
            raise ValueError(
                "Slurm only works with local benchmarking, set the `backend` "
                "argument to 'local'."
            )
        jobs = slurm.jobs_in_queue()
        if len(jobs) > 0:
            printing.log_warning(f"There are already slurm jobs in your queue: {jobs}")
            printing.log_info(
                "Suggest quitting benchit, running 'scancel -u $USER' and trying again."
            )

    # Use this data structure to keep a running index of all models
    # found across all scripts when running in --all mode
    models_found: Dict[str, ModelInfo] = {}

    for script in scripts:
        for device in devices:
            if use_slurm:
                slurm.run_benchit(
                    op="benchmark",
                    script=script,
                    search_dir=search_dir,
                    cache_dir=cache_dir,
                    rebuild=rebuild,
                    groq_compiler_flags=groq_compiler_flags,
                    groq_assembler_flags=groq_assembler_flags,
                    groq_num_chips=groq_num_chips,
                    groqview=groqview,
                    devices=[device],
                    max_depth=max_depth,
                    analyze_only=analyze_only,
                    build_only=build_only,
                    lean_cache=lean_cache,
                )

            else:

                # Parse the targets out of the script name
                script_name, targets = decode_script_name(script)

                # Instantiate an object that holds all of the arguments
                # for analysis, build, and benchmarking
                tracer_args = TracerArgs(
                    input=script_name,
                    lean_cache=lean_cache,
                    targets=targets,
                    max_depth=max_depth,
                    cache_dir=cache_dir,
                    rebuild=rebuild,
                    groq_compiler_flags=groq_compiler_flags,
                    groq_assembler_flags=groq_assembler_flags,
                    groq_num_chips=groq_num_chips,
                    groqview=groqview,
                    device=device,
                    backend=backend,
                    actions=actions,
                    models_found=models_found,
                    sequence=custom_sequence,
                )

                # Run analysis, build, and benchmarking on every model
                # in the script
                models_found = evaluate_script(tracer_args, script_args)

    # Wait until all the Slurm jobs are done
    if use_slurm:
        while len(slurm.jobs_in_queue()) != 0:
            print(
                f"Waiting: {len(slurm.jobs_in_queue())} "
                f"jobs left in queue: {slurm.jobs_in_queue()}"
            )
            time.sleep(5)

    printing.log_success(
        "The 'benchmark' command is complete. Use the 'report' command to get a .csv "
        "file that summarizes results across all builds in the cache."
    )
