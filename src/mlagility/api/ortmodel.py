import os
import json
import numpy as np
import groqflow.common.build as build
import mlagility.api.devices as devices
from mlagility.api.performance import MeasuredPerformance


class ORTModel:
    def __init__(self, state: build.State, tensor_type=np.array):

        self.tensor_type = tensor_type
        self.state = state
        self.device = "x86"

    def benchmark(
        self, repetitions: int = 100, backend: str = "local"
    ) -> MeasuredPerformance:
        benchmark_results = self._execute(repetitions=repetitions, backend=backend)
        return benchmark_results

    @property
    def _ort_performance_file(self):
        return devices.BenchmarkPaths(self.state, self.device, "local").outputs_file

    def _get_stat(self, stat):
        if os.path.exists(self._ort_performance_file):
            with open(self._ort_performance_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance[stat]
        else:
            raise devices.BenchmarkException(
                "No benchmarking outputs file found after benchmarking run."
                "Sorry we don't have more information."
            )

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
    def _ort_error_file(self):
        return devices.BenchmarkPaths(self.state, self.device, "local").errors_file

    def _execute(self, repetitions: int, backend: str = "local") -> MeasuredPerformance:

        """
        Execute model on ort and return the performance
        """

        # Remove previously stored latency/outputs
        if os.path.isfile(self._ort_performance_file):
            os.remove(self._ort_performance_file)
        if os.path.isfile(self._ort_error_file):
            os.remove(self._ort_error_file)

        if backend == "remote":
            devices.execute_ort_remotely(self.state, self.device, repetitions)
        elif backend == "local":
            devices.execute_ort_locally(self.state, self.device, repetitions)
        else:
            raise ValueError(
                f"Only 'remote' and 'local' are supported, but received {backend}"
            )

        return MeasuredPerformance(
            mean_latency=self._mean_latency,
            throughput=self._throughput,
            device=self._device,
            device_type=self.device,
            build_name=self.state.config.build_name,
        )


def load(build_name: str, cache_dir=build.DEFAULT_CACHE_DIR) -> ORTModel:
    state = build.load_state(cache_dir=cache_dir, build_name=build_name)
    return ORTModel(state)
