import os
import json
import numpy as np
import mlagility.api.devices as devices
from mlagility.api.performance import MeasuredPerformance
from mlagility.api.execute_trt import TRT_VERSION


class TRTModel:
    def __init__(self, cache_dir: str, build_name: str, tensor_type=np.array):

        self.tensor_type = tensor_type
        self.cache_dir = cache_dir
        self.build_name = build_name
        self.device_type = "nvidia"
        self.runtime = "trt"

    def benchmark(
        self, repetitions: int = 100, backend: str = "local"
    ) -> MeasuredPerformance:
        benchmark_results = self._execute(repetitions=repetitions, backend=backend)
        return benchmark_results

    @property
    def _trt_performance_file(self):
        return devices.BenchmarkPaths(
            self.cache_dir, self.build_name, self.device_type, "local"
        ).outputs_file

    def _get_stat(self, stat):
        if os.path.exists(self._trt_performance_file):
            with open(self._trt_performance_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance[stat]
        else:
            raise devices.BenchmarkException(
                "No benchmarking outputs file found after benchmarking run."
                "Sorry we don't have more information."
            )

    @property
    def mean_latency(self):
        return float(self._get_stat("Total Latency")["mean "].split(" ")[1])

    @property
    def throughput(self):
        return float(self._get_stat("Throughput").split(" ")[0])

    @property
    def device_name(self):
        return self._get_stat("Selected Device")

    @property
    def _trt_error_file(self):
        return devices.BenchmarkPaths(
            self.cache_dir, self.build_name, self.device_type, "local"
        ).errors_file

    def _execute(self, repetitions: int, backend: str = "local") -> MeasuredPerformance:
        """
        Execute model on TensorRT and return the performance
        """

        # Remove previously stored latency/outputs
        if os.path.isfile(self._trt_performance_file):
            os.remove(self._trt_performance_file)
        if os.path.isfile(self._trt_error_file):
            os.remove(self._trt_error_file)

        if backend == "remote":
            devices.execute_trt_remotely(
                self.cache_dir, self.build_name, self.device_type, repetitions
            )
        elif backend == "local":
            devices.execute_trt_locally(
                self.cache_dir, self.build_name, self.device_type, repetitions
            )
        else:
            raise ValueError(
                f"Only 'remote' and 'local' are supported, but received {backend}"
            )

        return MeasuredPerformance(
            mean_latency=self.mean_latency,
            throughput=self.throughput,
            device=self.device_name,
            device_type=self.device_type,
            runtime=self.runtime,
            runtime_version=TRT_VERSION,
            build_name=self.build_name,
        )
