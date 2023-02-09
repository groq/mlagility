from dataclasses import dataclass


@dataclass
class MeasuredPerformance:
    throughput: float
    mean_latency: float
    device: str
    device_type: str
    build_name: str
    throughput_units: str = "inferences per second (IPS)"
    latency_units: str = "milliseconds (ms)"
