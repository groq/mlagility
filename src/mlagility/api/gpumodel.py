import os
from dataclasses import dataclass
import json
import numpy as np
import torch
import groqflow.common.printing as printing
import groqflow.common.build as build
import mlagility.api.cloud as cloud


@dataclass
class GPUMeasuredPerformance:
    gpu_performance_file: str
    throughput_units: str = "inferences per second"

    @property
    def latency(self):
        if os.path.exists(self.gpu_performance_file):
            with open(self.gpu_performance_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance["Total Latency"]
        else:
            return "-"

    @property
    def throughput(self):
        if os.path.exists(self.gpu_performance_file):
            with open(self.gpu_performance_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance["Throughput"]
        else:
            return "-"


class GPUModel:
    def __init__(self, state: build.State, tensor_type=np.array):

        self.tensor_type = tensor_type
        self.state = state
        self.log_execute_path = os.path.join(
            build.output_dir(state.cache_dir, self.state.config.build_name),
            "log_gpu_execute.txt",
        )

    def benchmark(self, repetitions: int = 100, backend: str = "local") -> GPUMeasuredPerformance:

        printing.log_info(
            (
                "GPU is not used for accuracy comparisons it's only used for"
                " performance comparison. So inputs provided during model"
                " compilation is used.\n"
                " User is responsible for ensuring the remote GPU server is turned on and"
                " has python>=3.8, docker>=20.10 installed."
            )
        )

        benchmark_results = self._execute(repetitions=repetitions, backend=backend)
        self.state.info.gpu_measured_latency = benchmark_results.latency
        self.state.info.gpu_measured_throughput = benchmark_results.throughput
        return benchmark_results

    def gpu_performance_file(self):
        return os.path.join(
            self.state.cache_dir, self.state.config.build_name, "gpu_performance.json"
        )

    def gpu_error_file(self):
        return os.path.join(
            self.state.cache_dir, self.state.config.build_name, "gpu_error.npy"
        )

    def _execute(self, repetitions: int, backend: str) -> GPUMeasuredPerformance:
        """
        Execute model on GPU and return the performance
        """

        # Remove previously stored latency/outputs
        if os.path.isfile(self.gpu_performance_file()):
            os.remove(self.gpu_performance_file())
        if os.path.isfile(self.gpu_error_file()):
            os.remove(self.gpu_error_file())

        if (backend == "cloud"):
            cloud.execute_gpu_remotely(self.state, self.log_execute_path, repetitions)
        elif (backend == "local"):
            cloud.execute_gpu_locally(self.state, self.log_execute_path, repetitions)
        else:
            raise ValueError(f"Only 'cloud' and 'local' are supported, but received {backend}")

        return GPUMeasuredPerformance(self.gpu_performance_file())


class PytorchModelWrapper(GPUModel):
    def __init__(self, state):
        tensor_type = torch.tensor
        super(PytorchModelWrapper, self).__init__(state, tensor_type)

    # Pytorch models are callable
    def __call__(self):
        return self._execute(repetitions=100)


def load(build_name: str, cache_dir=build.DEFAULT_CACHE_DIR) -> GPUModel:
    state = build.load_state(cache_dir=cache_dir, build_name=build_name)

    if state.model_type == build.ModelType.PYTORCH:
        return PytorchModelWrapper(state)
    else:
        return GPUModel(state)
