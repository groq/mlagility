import sys
import os
import time
import json
import torch
import numpy as np
from statistics import mean
from typing import Any, Dict, Optional, List
from onnxflow import build_model
from onnxflow.justbuildit.stage import Sequence
import onnxflow.common.printing as printing
from mlagility.api import trtmodel, ortmodel
from mlagility.api.setup_ort import get_cpu_specs
import mlagility.common.filesystem as filesystem
from mlagility.api.performance import MeasuredPerformance
from mlagility.api.devices import SUPPORTED_DEVICES

MLAGILITY_DEFAULT_REBUILD_POLICY = "if_needed"


def benchmark_model(
    model: Any,
    inputs: Dict[str, Any],
    build_name: str,
    script_name: str = None,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    device: str = "x86",
    runtime: str = "ort",
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
            # pylint: disable=import-error
            from groqflow import groqit

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
            omodel = build_model(
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
                    cache_dir=omodel.state.cache_dir,
                    build_name=omodel.state.config.build_name,
                )
                perf = gpu_model.benchmark(backend=backend)

        elif device == "x86" and runtime == "ort":
            omodel = build_model(
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
                    build_name=omodel.state.config.build_name,
                    cache_dir=omodel.state.cache_dir,
                )
                perf = cpu_model.benchmark(backend=backend)

        elif device == "x86" and runtime in ["torch", "torch_compiled"]:

            # Create cache folder with stats file
            # Although building with an empty sequence is possible, we don't want
            # the model to be rebuilt when other devices are used, causing the
            # results inside of the stats file to be lost.
            build_model(
                model=model,
                inputs=inputs,
                build_name=build_name,
                cache_dir=cache_dir,
                rebuild=rebuild,
                sequence=sequence,
            )

            if runtime == "torch_compiled":
                model = torch.compile(model)

            if not build_only:
                repetitions = 100
                total_time = [0] * repetitions
                for idx in range(repetitions):
                    start_time = time.process_time()
                    model(**inputs)
                    end_time = time.process_time()
                    total_time[idx] = end_time - start_time

                # Calculate perf from total_time
                mean_latency_ms = mean(total_time) * 1000
                throughput_ips = float(1 / (np.sum(total_time) / repetitions))
                torch_version = torch.__version__
                if runtime == "torch":
                    device_name = (
                        get_cpu_specs()["CPU Name"] + f" (Pytorch {torch_version})"
                    )
                else:
                    device_name = (
                        get_cpu_specs()["CPU Name"]
                        + f" (Pytorch {torch_version} Compiled)"
                    )

                return MeasuredPerformance(
                    mean_latency=mean_latency_ms,
                    throughput=throughput_ips,
                    device=device_name,
                    device_type=device,
                    build_name=build_name,
                )

        else:
            raise ValueError(
                (
                    f"Got device '{device}' and runtime '{runtime}'. "
                    f"However, only the following combinations are allowed: {SUPPORTED_DEVICES}"
                )
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
