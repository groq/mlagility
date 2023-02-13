import os
import json
import numpy as np
import torch
import groqflow.common.printing as printing
import groqflow.common.build as build
import mlagility.api.devices as devices
from mlagility.api.performance import MeasuredPerformance


class ORTModel:
    def __init__(self, state: build.State, tensor_type=np.array):

        self.tensor_type = tensor_type
        self.state = state
        self.log_execute_path = os.path.join(
            build.output_dir(state.cache_dir, self.state.config.build_name),
            "log_cpu_execute.txt",
        )

    def benchmark(
        self, repetitions: int = 100, backend: str = "local"
    ) -> MeasuredPerformance:

        printing.log_info(
            (
                " CPU is not used for accuracy comparisons, it's only used for"
                " performance comparison. So dummy inputs are used.\n"
                " User is responsible for ensuring the CPU is available and"
                " has miniconda3 and python>=3.8 installed."
            )
        )

        benchmark_results = self._execute(repetitions=repetitions, backend=backend)
        self.state.info.cpu_measured_latency = benchmark_results.mean_latency
        self.state.info.cpu_measured_throughput = benchmark_results.throughput
        return benchmark_results

    @property
    def _cpu_performance_file(self):
        return os.path.join(
            self.state.cache_dir, self.state.config.build_name, "cpu_performance.json"
        )

    def _get_stat(self, stat):
        if os.path.exists(self._cpu_performance_file):
            with open(self._cpu_performance_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance[stat]
        else:
            return "-"

    @property
    def _mean_latency(self):
        return float(self._get_stat("Mean Latency(ms)"))

    @property
    def _throughput(self):
        return float(self._get_stat("Throughput"))

    @property
    def _device(self):
        return self._get_stat("CPU Name")

    @property
    def _cpu_error_file(self):
        return os.path.join(
            self.state.cache_dir, self.state.config.build_name, "cpu_error.npy"
        )

    def _execute(self, repetitions: int, backend: str = "local") -> MeasuredPerformance:

        """
        Execute model on cpu and return the performance
        """

        # Remove previously stored latency/outputs
        if os.path.isfile(self._cpu_performance_file):
            os.remove(self._cpu_performance_file)
        if os.path.isfile(self._cpu_error_file):
            os.remove(self._cpu_error_file)

        if backend == "remote":
            devices.execute_cpu_remotely(self.state, self.log_execute_path, repetitions)
        elif backend == "local":
            devices.execute_cpu_locally(self.state, self.log_execute_path, repetitions)
        else:
            raise ValueError(
                f"Only 'remote' and 'local' are supported, but received {backend}"
            )

        return MeasuredPerformance(
            mean_latency=self._mean_latency,
            throughput=self._throughput,
            device=self._device,
            device_type="x86",
            build_name=self.state.config.build_name,
        )


class PytorchModelWrapper(ORTModel):
    def __init__(self, state):
        tensor_type = torch.tensor
        super(PytorchModelWrapper, self).__init__(state, tensor_type)

    # Pytorch models are callable
    def __call__(self):
        return self._execute(repetitions=100)


def load(build_name: str, cache_dir=build.DEFAULT_CACHE_DIR) -> ORTModel:
    state = build.load_state(cache_dir=cache_dir, build_name=build_name)

    if state.model_type == build.ModelType.PYTORCH:
        return PytorchModelWrapper(state)
    else:
        return ORTModel(state)
