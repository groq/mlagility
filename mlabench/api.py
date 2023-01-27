from typing import Any, Dict, Optional

from groqflow import groqit
import groqflow.common.build as build
import groqflow.justgroqit.stage as stage
from groqflow.justgroqit.ignition import identify_model_type
import groqflow.justgroqit.export as export
import groqflow.justgroqit.hummingbird as hummingbird
from mlabench.benchmark import gpumodel, cpumodel


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
    cache_dir: str = build.DEFAULT_CACHE_DIR,
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
    )

    return gmodel


def benchit(
    model: Any,
    inputs: Dict[str, Any],
    build_name: Optional[str] = None,
    cache_dir: str = build.DEFAULT_CACHE_DIR,
    device: str = "groq",
):
    """
    Benchmark a model against some inputs on target hardware
    """

    if device == "groq":
        gmodel = groqit(
            model=model, inputs=inputs, build_name=build_name, cache_dir=cache_dir
        )
        perf = gmodel.benchmark()
    elif device == "gpu":
        gmodel = exportit(
            model=model, inputs=inputs, build_name=build_name, cache_dir=cache_dir
        )
        gpu_model = gpumodel.load(
            gmodel.state.config.build_name, cache_dir=gmodel.state.cache_dir
        )
        perf = gpu_model.benchmark()

        latency_ms = float(perf.latency["mean "].split(" ")[1])
        throughput_ips = float(perf.throughput.split(" ")[0])
    elif device == "cpu":
        gmodel = exportit(
            model=model, inputs=inputs, build_name=build_name, cache_dir=cache_dir
        )
        cpu_model = cpumodel.load(
            gmodel.state.config.build_name, cache_dir=gmodel.state.cache_dir
        )
        perf = cpu_model.benchmark()

        latency_ms = float(perf.latency)
        throughput_ips = float(perf.throughput)
    else:
        raise ValueError("Only groq, cpu or gpu are allowed values for device")

    print(
        f"\nPerformance of build {gmodel.state.config.build_name} on device {device} is:"
    )
    print(f"latency: {latency_ms:.3f} ms")
    print(f"throughput: {throughput_ips:.1f} ips")
