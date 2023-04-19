from dataclasses import dataclass
import onnxflow.common.printing as printing


@dataclass
class MeasuredPerformance:
    throughput: float
    mean_latency: float
    device: str
    runtime: str
    runtime_version: str
    device_type: str
    build_name: str
    throughput_units: str = "inferences per second (IPS)"
    latency_units: str = "milliseconds (ms)"

    def print(self):
        printing.log_info(
            f"\nPerformance of build {self.build_name} on {self.device} "
            f"({self.runtime} v{self.runtime_version}) is:"
        )
        print(f"\tMean Latency: {self.mean_latency:.3f} {self.latency_units}")
        print(f"\tThroughput: {self.throughput:.1f} {self.throughput_units}")
        print()
