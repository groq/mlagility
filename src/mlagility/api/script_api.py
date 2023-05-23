import time
import os
import types
import importlib.machinery
from typing import Tuple, List, Dict, Optional, Union
import onnxflow.common.printing as printing
import onnxflow.common.build as build
import onnxflow.common.exceptions as exceptions
import onnxflow.justbuildit.stage as stage
import onnxflow.justbuildit.export as export
import mlagility.cli.spawn as spawn
import mlagility.common.filesystem as filesystem
import mlagility.common.labels as labels_library
from mlagility.api.model_api import benchmark_model
from mlagility.api.devices import SUPPORTED_DEVICES, DEFAULT_RUNTIME
from mlagility.analysis.analysis import (
    evaluate_script,
    TracerArgs,
    Action,
)
from mlagility.analysis.util import ModelInfo


def decode_input_script(input: str) -> Tuple[str, List[str], str]:
    # Parse the targets out of the script name
    # Targets use the format:
    #   script_path.py::target0,target1,...,targetN
    decoded_input = input.split("::")
    script_path = os.path.abspath(decoded_input[0])

    if len(decoded_input) == 2:
        targets = decoded_input[1].split(",")
        encoded_input = script_path + "::" + decoded_input[1]
    elif len(decoded_input) == 1:
        targets = []
        encoded_input = script_path
    else:
        raise ValueError(
            "Each script input to benchit should have either 0 or 1 '::' in it."
            f"However, {script_path} was received."
        )

    return script_path, targets, encoded_input


def load_sequence_from_file(
    sequence: Union[str, stage.Sequence],
    use_slurm: bool,
    process_isolation: bool,
):
    """
    Import the sequence file to get a custom sequence, if the user provided
    one. Sequence instances are passed through this function as long as
    the user is not going to spawn a new process (indicated by `use_slurm` or `process_isolation`).
    """

    if sequence is not None:
        if use_slurm or process_isolation:
            # The spawned process will need to load a sequence file
            if not isinstance(sequence, str):
                raise ValueError(
                    "The 'sequence' arg must be a str (path to a sequence file) "
                    "when use_slurm=True or process_isolation=True."
                )
            custom_sequence = sequence
        elif isinstance(sequence, str):
            loader = importlib.machinery.SourceFileLoader("a_b", sequence)
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            # pylint: disable = no-member
            custom_sequence = mod.get_sequence()
        elif isinstance(sequence, stage.Sequence):
            custom_sequence = sequence
        else:
            raise ValueError(
                "The 'sequence' arg must be a str (path to a sequence file) "
                "or an instance of the Sequence class."
            )
    else:
        custom_sequence = None

    return custom_sequence


def benchmark_script(
    input_scripts: List[str],
    use_slurm: bool = False,
    process_isolation: bool = False,
    lean_cache: bool = False,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    labels: List[str] = None,
    rebuild: Optional[str] = None,
    device: str = None,
    backend: str = "local",
    runtimes: List[str] = None,
    analyze_only: bool = False,
    build_only: bool = False,
    export_only: bool = False,
    resume: bool = False,
    script_args: Optional[str] = None,
    max_depth: int = 0,
    onnx_opset: int = build.DEFAULT_ONNX_OPSET,
    sequence: Union[str, stage.Sequence] = None,
    groq_compiler_flags: Optional[List[str]] = None,
    groq_assembler_flags: Optional[List[str]] = None,
    groq_num_chips: Optional[int] = None,
    groqview: bool = False,
):

    # Make sure the cache directory exists
    filesystem.make_cache_dir(cache_dir)

    custom_sequence = load_sequence_from_file(sequence, use_slurm, process_isolation)

    if device is None:
        device = "x86"
    if runtimes is None:
        runtimes = [SUPPORTED_DEVICES[device][DEFAULT_RUNTIME]]

    # Force the user to specify a legal cache dir in NFS if they are using slurm
    if cache_dir == filesystem.DEFAULT_CACHE_DIR and use_slurm:
        printing.log_warning(
            "Using the default cache directory when using Slurm will cause your cached "
            "files to only be available at the Slurm node. If this is not the behavior "
            "you desired, please se a --cache-dir that is accessible by both the slurm "
            "node and your local machine."
        )

    # Get list containing only script names
    clean_scripts = [decode_input_script(script)[0] for script in input_scripts]

    # Validate that the script exists
    for script in clean_scripts:
        if os.path.isdir(script):
            raise exceptions.ArgError(
                f'"{script}" is a directory. Do you mean "{script}/*.py" ?'
            )
        if not os.path.isfile(script):
            raise exceptions.ArgError(
                (
                    f"{script} could not be found. If this corresponds to a "
                    "regular expression, the regular expression did not match "
                    "any file(s)."
                )
            )
        if not script.endswith(".py"):
            raise exceptions.ArgError(f"Script must end with .py (got {script})")

    # Decode benchit args into TracerArgs flags
    if analyze_only:
        actions = [
            Action.ANALYZE,
        ]
    elif export_only:
        actions = [
            Action.ANALYZE,
            Action.EXPORT,
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
        jobs = spawn.slurm_jobs_in_queue()
        if len(jobs) > 0:
            printing.log_warning(f"There are already slurm jobs in your queue: {jobs}")
            printing.log_info(
                "Suggest quitting benchit, running 'scancel -u $USER' and trying again."
            )

    # Use this data structure to keep a running index of all models
    models_found: Dict[str, ModelInfo] = {}

    for script in input_scripts:
        script_path, targets, encoded_input = decode_input_script(script)

        # Skip a script if the required_labels are not a subset of the script_labels.
        if labels:
            required_labels = labels_library.to_dict(labels)
            script_labels = labels_library.load_from_file(encoded_input)
            if not labels_library.is_subset(required_labels, script_labels):
                continue

        # Resume mode will skip any scripts that have already been evaluated.
        # We keep track of that using the cache database
        db = filesystem.CacheDatabase(cache_dir)
        if db.exists():
            if resume and db.script_in_database(filesystem.clean_script_name(script)):
                continue

        # Add the script to the database
        # Skip this if we are in Slurm mode; it has already been done in the main process
        if os.environ.get("USING_SLURM") != "TRUE":
            db.add_script(filesystem.clean_script_name(script))

        for runtime in runtimes:
            if use_slurm or process_isolation:
                # Decode args into spawn.Target
                if use_slurm and process_isolation:
                    raise ValueError(
                        "use_slurm and process_isolation are mutually exclusive, but both are True"
                    )
                elif use_slurm:
                    target = spawn.Target.SLURM
                elif process_isolation:
                    target = spawn.Target.LOCAL_PROCESS
                else:
                    raise ValueError(
                        "This code path requires use_slurm or use_process to be True, "
                        "but both are False"
                    )

                spawn.run_benchit(
                    op="benchmark",
                    script=encoded_input,
                    cache_dir=cache_dir,
                    rebuild=rebuild,
                    target=target,
                    groq_compiler_flags=groq_compiler_flags,
                    groq_assembler_flags=groq_assembler_flags,
                    groq_num_chips=groq_num_chips,
                    groqview=groqview,
                    device=device,
                    runtimes=[runtime],
                    max_depth=max_depth,
                    onnx_opset=onnx_opset,
                    analyze_only=analyze_only,
                    build_only=build_only,
                    export_only=export_only,
                    lean_cache=lean_cache,
                )

            else:

                # Instantiate an object that holds all of the arguments
                # for analysis, build, and benchmarking
                tracer_args = TracerArgs(
                    input=script_path,
                    lean_cache=lean_cache,
                    targets=targets,
                    max_depth=max_depth,
                    onnx_opset=onnx_opset,
                    cache_dir=cache_dir,
                    rebuild=rebuild,
                    groq_compiler_flags=groq_compiler_flags,
                    groq_assembler_flags=groq_assembler_flags,
                    groq_num_chips=groq_num_chips,
                    groqview=groqview,
                    device=device,
                    backend=backend,
                    runtime=runtime,
                    actions=actions,
                    models_found=models_found,
                    sequence=custom_sequence,
                )

                # Run analysis, build, and benchmarking on every model
                # in the script
                models_found = evaluate_script(tracer_args, script_args)

    # Wait until all the Slurm jobs are done
    if use_slurm:
        while len(spawn.slurm_jobs_in_queue()) != 0:
            print(
                f"Waiting: {len(spawn.slurm_jobs_in_queue())} "
                f"jobs left in queue: {spawn.slurm_jobs_in_queue()}"
            )
            time.sleep(5)

        spawn.update_database_builds(cache_dir, input_scripts)

    printing.log_success("The 'benchmark' command is complete.")


def benchmark_files(
    input_files: str = None,
    use_slurm: bool = False,
    process_isolation: bool = False,
    lean_cache: bool = False,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    labels: List[str] = None,
    rebuild: Optional[str] = None,
    device: str = None,
    backend: str = "local",
    runtimes: List[str] = None,
    analyze_only: bool = False,
    build_only: bool = False,
    export_only: bool = False,
    resume: bool = False,
    script_args: Optional[str] = None,
    max_depth: int = 0,
    onnx_opset: int = build.DEFAULT_ONNX_OPSET,
    sequence: Union[str, stage.Sequence] = None,
    groq_compiler_flags: Optional[List[str]] = None,
    groq_assembler_flags: Optional[List[str]] = None,
    groq_num_chips: Optional[int] = None,
    groqview: bool = False,
):

    """
    Inspect the input_files and sort them into .py and .onnx files.
    Pass .py files into benchmark_script() and .onnx files into benchmark_model().
    """

    python_scripts = []
    onnx_files = []

    for file in input_files:
        if ".py" in file:
            python_scripts.append(file)
        elif file.endswith(".onnx"):
            onnx_files.append(file)

    if len(python_scripts):
        # Pass the args straight into benchmark_script(), which knows how
        # to iterate over python scripts
        benchmark_script(
            input_scripts=python_scripts,
            use_slurm=use_slurm,
            process_isolation=process_isolation,
            lean_cache=lean_cache,
            cache_dir=cache_dir,
            labels=labels,
            rebuild=rebuild,
            device=device,
            backend=backend,
            runtimes=runtimes,
            analyze_only=analyze_only,
            build_only=build_only,
            export_only=export_only,
            resume=resume,
            script_args=script_args,
            max_depth=max_depth,
            onnx_opset=onnx_opset,
            sequence=sequence,
            groq_compiler_flags=groq_compiler_flags,
            groq_assembler_flags=groq_assembler_flags,
            groq_num_chips=groq_num_chips,
            groqview=groqview,
        )

    # Iterate and pass each ONNX file into benchmark_model() one at a time
    for onnx_file in onnx_files:
        build_name = filesystem.clean_script_name(onnx_file)

        # Sequence that just passes the onnx file into the cache
        if sequence is None:
            onnx_sequence = stage.Sequence(
                unique_name="onnx_passthrough",
                monitor_message="Pass through ONNX file",
                stages=[export.ReceiveOnnxModel(), export.SuccessStage()],
                enable_model_validation=True,
            )
        else:
            onnx_sequence = load_sequence_from_file(
                sequence, use_slurm, process_isolation
            )

        for runtime in runtimes:
            benchmark_model(
                model=onnx_file,
                inputs=None,
                build_name=build_name,
                cache_dir=cache_dir,
                device=device,
                backend=backend,
                runtime=runtime,
                build_only=build_only,
                export_only=export_only,
                lean_cache=lean_cache,
                rebuild=rebuild,
                onnx_opset=onnx_opset,
                groq_compiler_flags=groq_compiler_flags,
                groq_assembler_flags=groq_assembler_flags,
                groq_num_chips=groq_num_chips,
                groqview=groqview,
                sequence=onnx_sequence,
            )
