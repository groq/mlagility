import sys
import os
from typing import Any, Dict, Optional, List
from groqflow import groqit
import groqflow.common.build as build
from groqflow.justgroqit.stage import Sequence
from groqflow.justgroqit.ignition import identify_model_type
import groqflow.justgroqit.export as export
import groqflow.justgroqit.hummingbird as hummingbird
import groqflow.common.printing as printing
from groqflow.groqmodel import GroqModel
from mlagility.api import trtmodel, ortmodel
import mlagility.common.filesystem as filesystem
from mlagility.api.performance import MeasuredPerformance
import mlagility.common.groqflow_helpers as groqflow_helpers

MLAGILITY_DEFAULT_REBUILD_POLICY = "if_needed"


model_type_to_export_sequence = {
    build.ModelType.PYTORCH: Sequence(
        unique_name="pytorch_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            export.ExportPytorchModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            groqflow_helpers.SuccessStage(),
        ],
        enable_model_validation=True,
    ),
    build.ModelType.KERAS: Sequence(
        unique_name="keras_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            export.ExportKerasModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            groqflow_helpers.SuccessStage(),
        ],
        enable_model_validation=True,
    ),
    build.ModelType.ONNX_FILE: Sequence(
        unique_name="onnx_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            export.ReceiveOnnxModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            groqflow_helpers.SuccessStage(),
        ],
        enable_model_validation=True,
    ),
    build.ModelType.HUMMINGBIRD: Sequence(
        unique_name="pytorch_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            hummingbird.ConvertHummingbirdModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            groqflow_helpers.SuccessStage(),
        ],
        enable_model_validation=True,
    ),
}


def exportit(
    model: Any,
    inputs: Dict[str, Any],
    build_name: Optional[str] = None,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    rebuild: str = MLAGILITY_DEFAULT_REBUILD_POLICY,
    sequence: Sequence = None,
) -> GroqModel:
    """
    Export a model to ONNX and save it to the cache
    """

    model_type = identify_model_type(model)

    if sequence is None:
        sequence_arg = model_type_to_export_sequence[model_type]
    else:
        sequence_arg = sequence

    gmodel = groqit(
        model=model,
        inputs=inputs,
        build_name=build_name,
        cache_dir=cache_dir,
        sequence=sequence_arg,
        rebuild=rebuild,
    )

    return gmodel


def benchmark_model(
    model: Any,
    inputs: Dict[str, Any],
    build_name: str,
    script_name: str = None,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    device: str = "x86",
    backend: str = "local",
    build_only: bool = False,
    lean_cache: bool = False,
    rebuild: str = MLAGILITY_DEFAULT_REBUILD_POLICY,
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
                sequence=sequence,
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
                    build_name=gmodel.state.config.build_name,
                )

        elif device == "nvidia":
            gmodel = exportit(
                model=model,
                inputs=inputs,
                build_name=build_name,
                cache_dir=cache_dir,
                rebuild=rebuild,
                sequence=sequence,
            )

            if not build_only:
                printing.log_info(f"Benchmarking on {backend} {device}...")
                gpu_model = trtmodel.TRTModel(
                    cache_dir=gmodel.state.cache_dir,
                    build_name=gmodel.state.config.build_name,
                )
                perf = gpu_model.benchmark(backend=backend)

        elif device == "x86":
            gmodel = exportit(
                model=model,
                inputs=inputs,
                build_name=build_name,
                cache_dir=cache_dir,
                rebuild=rebuild,
                sequence=sequence,
            )

            if not build_only:
                printing.log_info(f"Benchmarking on {backend} {device}...")
                cpu_model = ortmodel.ORTModel(
                    build_name=gmodel.state.config.build_name,
                    cache_dir=gmodel.state.cache_dir,
                )
                perf = cpu_model.benchmark(backend=backend)

        else:
            raise ValueError(
                "Only groq, x86, or nvidia are allowed values for device type, "
                f"but got {device}"
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
