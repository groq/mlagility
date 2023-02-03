import os
from dataclasses import dataclass
import json
import numpy as np
import torch
import groqflow.common.printing as printing
import groqflow.common.build as build
import mlagility.api.cloud as cloud


@dataclass
class CPUMeasuredPerformance:
    cpu_performance_file: str
    throughput_units: str = "inferences per second"

    @property
    def latency(self):
        if os.path.exists(self.cpu_performance_file):
            with open(self.cpu_performance_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance["Mean Latency(ms)"]
        else:
            return "-"

    @property
    def throughput(self):
        if os.path.exists(self.cpu_performance_file):
            with open(self.cpu_performance_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance["Throughput"]
        else:
            return "-"


class CPUModel:
    def __init__(self, state: build.State, tensor_type=np.array):

        self.tensor_type = tensor_type
        self.state = state
        self.log_execute_path = os.path.join(
            build.output_dir(state.cache_dir, self.state.config.build_name),
            "log_cpu_execute.txt",
        )

    def benchmark(self, repetitions: int = 100) -> CPUMeasuredPerformance:

        printing.log_info(
            (
                " CPU is not used for accuracy comparisons it's only used for"
                " performance comparison. So dummy inputs are used.\n"
                " User is responsible for ensuring the remote cpu server is turned on and"
                " has python>=3.8 installed."
            )
        )

        benchmark_results = self._execute(repetitions=repetitions)
        self.state.info.cpu_measured_latency = benchmark_results.latency
        self.state.info.cpu_measured_throughput = benchmark_results.throughput
        return benchmark_results

    def cpu_performance_file(self):
        return os.path.join(
            self.state.cache_dir, self.state.config.build_name, "cpu_performance.json"
        )

    def cpu_error_file(self):
        return os.path.join(
            self.state.cache_dir, self.state.config.build_name, "cpu_error.npy"
        )

    def _execute(self, repetitions: int) -> CPUMeasuredPerformance:
        """
        Execute model on cpu and return the performance
        """

        # Remove previously stored latency/outputs
        if os.path.isfile(self.cpu_performance_file()):
            os.remove(self.cpu_performance_file())
        if os.path.isfile(self.cpu_error_file()):
            os.remove(self.cpu_error_file())

        # Only cloud execution of the CPU is supported, local execution is not supported
        cloud.execute_cpu_remotely(self.state, self.log_execute_path, repetitions)

        return CPUMeasuredPerformance(self.cpu_performance_file())


class PytorchModelWrapper(CPUModel):
    def __init__(self, state):
        tensor_type = torch.tensor
        super(PytorchModelWrapper, self).__init__(state, tensor_type)

    # Pytorch models are callable
    def __call__(self):
        return self._execute(repetitions=100)


def load(build_name: str, cache_dir=build.DEFAULT_CACHE_DIR) -> CPUModel:
    state = build.load_state(cache_dir=cache_dir, build_name=build_name)

    if state.model_type == build.ModelType.PYTORCH:
        return PytorchModelWrapper(state)
    else:
        return CPUModel(state)
