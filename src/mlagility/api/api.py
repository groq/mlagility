from typing import Any, Dict, Optional, List
import os
from groqflow import groqit
import groqflow.common.build as build
import groqflow.justgroqit.stage as stage
from groqflow.justgroqit.ignition import identify_model_type
import groqflow.justgroqit.export as export
import groqflow.justgroqit.hummingbird as hummingbird
from mlagility.api import gpumodel, cpumodel
import mlagility.common.filesystem as filesystem
import mlagility.analysis.util as util

MLAGILITY_DEFAULT_REBUILD_POLICY = "if_needed"


class SuccessStage(stage.GroqitStage):
    """
    Stage that sets state.build_status = build.Status.SUCCESSFUL_BUILD,
    indicating to groqit() that the build can be used for benchmarking
    CPUs and GPUs.
    """

    def __init__(self):
        super().__init__(
            unique_name="set_success",
            monitor_message="Finishing up",
        )

    def fire(self, state: build.State):
        state.build_status = build.Status.SUCCESSFUL_BUILD

        return state


model_type_to_export_sequence = {
    build.ModelType.PYTORCH: stage.Sequence(
        unique_name="pytorch_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            export.ExportPytorchModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            SuccessStage(),
        ],
        enable_model_validation=True,
    ),
    build.ModelType.KERAS: stage.Sequence(
        unique_name="keras_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            export.ExportKerasModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            SuccessStage(),
        ],
        enable_model_validation=True,
    ),
    build.ModelType.ONNX_FILE: stage.Sequence(
        unique_name="onnx_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            export.ReceiveOnnxModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            SuccessStage(),
        ],
        enable_model_validation=True,
    ),
    build.ModelType.HUMMINGBIRD: stage.Sequence(
        unique_name="pytorch_bench",
        monitor_message="Benchmark sequence for PyTorch",
        stages=[
            hummingbird.ConvertHummingbirdModel(),
            export.OptimizeOnnxModel(),
            export.ConvertOnnxToFp16(),
            SuccessStage(),
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
):
    """
    Export a model to ONNX and save it to the cache
    """

    model_type = identify_model_type(model)

    gmodel = groqit(
        model=model,
        inputs=inputs,
        build_name=build_name,
        cache_dir=cache_dir,
        sequence=model_type_to_export_sequence[model_type],
        rebuild=rebuild,
    )

    return gmodel


def benchit(
    model: Any,
    inputs: Dict[str, Any],
    build_name: Optional[str] = None,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    device: str = "groq",
    build_only: bool = False,
    lean_cache: bool = False,
    rebuild: str = MLAGILITY_DEFAULT_REBUILD_POLICY,
    groq_compiler_flags: Optional[List[str]] = None,
    groq_assembler_flags: Optional[List[str]] = None,
    groq_num_chips: Optional[int] = None,
    groqview: bool = False,
):
    """
    Benchmark a model against some inputs on target hardware
    """

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
        )
        perf = gmodel.benchmark()
    elif device == "nvidia":
        gmodel = exportit(
            model=model,
            inputs=inputs,
            build_name=build_name,
            cache_dir=cache_dir,
            rebuild=rebuild,
        )

        if build_only:
            return

        gpu_model = gpumodel.load(
            gmodel.state.config.build_name, cache_dir=gmodel.state.cache_dir
        )
        perf = gpu_model.benchmark()
    elif device == "x86":
        gmodel = exportit(
            model=model,
            inputs=inputs,
            build_name=build_name,
            cache_dir=cache_dir,
            rebuild=rebuild,
        )

        if build_only:
            return

        cpu_model = cpumodel.load(
            gmodel.state.config.build_name, cache_dir=gmodel.state.cache_dir
        )
        perf = cpu_model.benchmark()
    else:
        raise ValueError(
            f"Only groq, x86, or nvidia are allowed values for device type, but got {device}"
        )

    print(
        f"\nPerformance of build {gmodel.state.config.build_name} on {perf.device_type} device "
        f"{perf.device} is:"
    )
    print(f"latency: {perf.mean_latency:.3f} {perf.latency_units}")
    print(f"throughput: {perf.throughput:.1f} {perf.throughput_units}")

    # Add metadata and clean cache if needed
    output_dir = os.path.join(cache_dir, perf.build_name)
    if os.path.isdir(output_dir):
        # Delete all files except logs and other metadata
        if lean_cache:
            util.clean_output_dir(output_dir)
