import sys
import os
from timeit import default_timer as timer
from statistics import mean
from typing import Any, Dict, Optional, List
import torch
from packaging import version
import numpy as np
from onnxflow import build_model
from onnxflow.justbuildit.stage import Sequence
import onnxflow.common.exceptions as exp
import onnxflow.common.printing as printing
import onnxflow.common.build as build
import onnxflow.justbuildit.ignition as ignition
from mlagility.api import trtmodel, ortmodel
from mlagility.api.setup_ort import get_cpu_specs
import mlagility.common.filesystem as filesystem
from mlagility.api.performance import MeasuredPerformance
from mlagility.api.devices import SUPPORTED_DEVICES, BenchmarkException


MLAGILITY_DEFAULT_REBUILD_POLICY = "if_needed"


def benchmark_model(
    model: Any,
    inputs: Dict[str, Any],
    build_name: str,
    script_name: str = None,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    device: str = "x86",
    runtime: str = "ort",
    backend: str = "local",
    build_only: bool = False,
    export_only: bool = False,
    lean_cache: bool = False,
    rebuild: str = MLAGILITY_DEFAULT_REBUILD_POLICY,
    onnx_opset: int = build.DEFAULT_ONNX_OPSET,
    groq_compiler_flags: Optional[List[str]] = None,
    groq_assembler_flags: Optional[List[str]] = None,
    groq_num_chips: Optional[int] = None,
    groqview: bool = False,
    sequence: Sequence = None,
) -> MeasuredPerformance:
    """
    Benchmark a model against some inputs on target hardware
    """
    # Make sure the cache exists, and populate the cache database
    # with this script and build.
    # Skip this if we are in Slurm mode; it will be done in the main process
    if os.environ.get("USING_SLURM") != "TRUE":
        filesystem.make_cache_dir(cache_dir)
        db = filesystem.CacheDatabase(cache_dir)

        if script_name is None:
            db_script_name = filesystem.clean_script_name(sys.argv[0])
        else:
            db_script_name = script_name

        db.add_build(db_script_name, build_name)

    # Build and benchmark the model
    try:

        if device == "groq":
            # pylint: disable=import-error
            from groqflow import groqit
            import onnxflow.justbuildit.export as export
            import onnxflow.justbuildit.stage as stage
            import groqflow.justgroqit.export as gf_export
            import groqflow.justgroqit.compile as gf_compile
            import groqflow.common.build as gf_build

            if onnx_opset != gf_build.DEFAULT_ONNX_OPSET:
                raise ValueError(
                    "ONNX opset for Groq builds must match GroqFlow's ONNX opset, "
                    f"{gf_build.DEFAULT_ONNX_OPSET}, however onnx_opset is set to {onnx_opset}"
                )

            # Set the GroqFlow sequence to execute Stages in the same order
            # as build_model()

            if sequence is None:
                groqflow_sequence = stage.Sequence(
                    "groqflow_sequence",
                    "GroqFlow build",
                    [
                        export.ExportPytorchModel(),
                        export.OptimizeOnnxModel(),
                        export.ConvertOnnxToFp16(),
                        gf_export.CheckOnnxCompatibility(),
                        gf_compile.CompileOnnx(),
                        gf_compile.Assemble(),
                    ],
                    enable_model_validation=True,
                )
            else:
                groqflow_sequence = sequence

            gmodel = groqit(
                model=model,
                inputs=inputs,
                build_name=build_name,
                cache_dir=cache_dir,
                rebuild=rebuild,
                compiler_flags=groq_compiler_flags,
                assembler_flags=groq_assembler_flags,
                num_chips=groq_num_chips,
                groqview=groqview,
                sequence=groqflow_sequence,
            )

            if not build_only:
                printing.log_info(f"Benchmarking on {backend} {device}...")
                groq_perf = gmodel.benchmark()

                # Map GroqFlow's GroqMeasuredPerformance into the MeasuredPerformance
                # class used by the MLAgility project
                perf = MeasuredPerformance(
                    throughput=groq_perf.throughput,
                    mean_latency=groq_perf.latency,
                    device="GroqChip1",
                    device_type="groq",
                    runtime="groq",
                    runtime_version=gmodel.state.groqflow_version,
                    build_name=gmodel.state.config.build_name,
                )

        elif device == "nvidia":
            omodel = build_model(
                model=model,
                inputs=inputs,
                build_name=build_name,
                cache_dir=cache_dir,
                rebuild=rebuild,
                sequence=sequence,
                onnx_opset=onnx_opset,
                export_only=export_only,
            )

            if not build_only:
                printing.log_info(f"Benchmarking on {backend} {device}...")
                gpu_model = trtmodel.TRTModel(
                    cache_dir=omodel.state.cache_dir,
                    build_name=omodel.state.config.build_name,
                )
                perf = gpu_model.benchmark(backend=backend)

        elif device == "x86" and runtime == "ort":
            omodel = build_model(
                model=model,
                inputs=inputs,
                build_name=build_name,
                cache_dir=cache_dir,
                rebuild=rebuild,
                sequence=sequence,
                onnx_opset=onnx_opset,
                export_only=export_only,
            )

            if not build_only:
                printing.log_info(f"Benchmarking on {backend} {device}...")
                cpu_model = ortmodel.ORTModel(
                    build_name=omodel.state.config.build_name,
                    cache_dir=omodel.state.cache_dir,
                )
                perf = cpu_model.benchmark(backend=backend)

        elif device == "x86" and runtime in ["torch-eager", "torch-compiled"]:

            # Ensure we have the correct model type
            model_type = ignition.identify_model_type(model)
            if model_type != build.ModelType.PYTORCH:
                raise exp.IntakeError(
                    f"Only Pytorch models are valid when runtime is {runtime}"
                )

            # Benchmarking using `torch-eager` and `torch-compiled` does not require
            # converting the model to ONNX. Here, we simply create a state in order to
            # have a place to store our results.
            if not os.path.isfile(build.state_file(cache_dir, build_name)):
                config = ignition.lock_config(
                    model=model,
                    build_name=build_name,
                    sequence=sequence,
                )
                state = build.State(
                    model=model,
                    inputs=inputs,
                    rebuild=rebuild,
                    cache_dir=cache_dir,
                    config=config,
                    model_type=model_type,
                )
                state.save()

            # Ensure we have the required version of Pytorch
            torch_version = str(torch.__version__)
            if runtime == "torch-compiled":
                clean_torch_version = torch_version.split("+")[0]
                if version.parse(clean_torch_version) < version.parse("2.0.0"):
                    BenchmarkException(
                        (
                            f"{runtime} can only be used with Pytorch 2.0.0 or above. "
                            f"However, version {torch_version} was found."
                        )
                    )

            # Compile model if needed
            selected_model = (
                torch.compile(model) if runtime == "torch-compiled" else model
            )

            if not build_only:
                num_iterations = 100
                per_iteration_latency = [0] * num_iterations
                for idx in range(num_iterations):
                    start_time = timer()
                    selected_model(**inputs)
                    end_time = timer()
                    per_iteration_latency[idx] = end_time - start_time

                # Calculate perf from per_iteration_latency
                mean_latency_ms = mean(per_iteration_latency) * 1000
                throughput_ips = float(
                    1 / (np.sum(per_iteration_latency) / num_iterations)
                )

                return MeasuredPerformance(
                    mean_latency=mean_latency_ms,
                    throughput=throughput_ips,
                    device=get_cpu_specs()["CPU Name"],
                    device_type=device,
                    runtime=runtime,
                    runtime_version=torch_version,
                    build_name=build_name,
                )

        else:
            raise ValueError(
                (
                    f"Got device '{device}' and runtime '{runtime}'. "
                    f"However, only the following combinations are allowed: {SUPPORTED_DEVICES}"
                )
            )

    finally:
        # Make sure the build and cache dirs exist and have the proper marker files
        # NOTE: We would do this at the top of the file, however
        # there are conditions where groqit() will wipe the build dir,
        # which would eliminate our marker file
        filesystem.make_build_dir(cache_dir, build_name)

        # Clean cache if needed
        if lean_cache:
            printing.log_info("Removing build artifacts...")
            filesystem.clean_output_dir(cache_dir, build_name)

    if not build_only:
        perf.print()
        return perf
    else:
        return None
