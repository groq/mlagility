import os
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import groqflow.common.printing as printing
import groqflow.common.build as build
import groqflow.common.cache as cache
from groqflow.groqmodel import groqmodel
from mlagility.common import labels
from mlagility.api.ortmodel import ORTModel
from mlagility.api.trtmodel import TRTModel
from mlagility.api.devices import BenchmarkException
from mlagility.common.filesystem import check_cache_dir


def _update_numeric_attribute(
    new_val, current_val, default="-", build_name=None, parameter_name=None
):
    """
    Updates a numeric attribute if needed
    """
    if parameter_name == "groq_estimated_latency":
        print(new_val, current_val)
    if current_val == default:
        return new_val if new_val is not None else default
    else:
        if (
            build_name is not None
            and parameter_name is not None
            and new_val != current_val
        ):
            printing.log_warning(
                (
                    f"Got multiple values for {parameter_name} on build {build_name} "
                    f"(keeping {current_val}, discarding {new_val})"
                )
            )
        return current_val


def _get_estimated_groq_latency(model_folder, cache_folder):
    """
    Returns estimated Groq latency (io + compute - not including runtime) in ms
    """
    try:
        gmodel = groqmodel.load(model_folder, cache_folder)
        if (
            gmodel.state.info.assembler_success  # pylint: disable=singleton-comparison
            == True
        ):
            perf = gmodel.estimate_performance()
            if perf.latency_units != "seconds":
                raise BenchmarkException(
                    (
                        "Expected Groq latency_units to be in seconds, "
                        f"but got {perf.latency_units}"
                    )
                )
            return 1000 * perf.latency
        else:
            return "-"
    except FileNotFoundError:
        return "-"


def get_report_name() -> str:
    """
    Returns the name of the .csv report
    """
    day = datetime.now().day
    month = datetime.now().month
    year = datetime.now().year
    date_key = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    return f"{date_key}.csv"


@dataclass
class BuildResults:
    """
    Class for keeping track of  the results of a given build
    """

    # Description: Name of the model
    # Source: Specified as part of the .py file using the label "name"
    model_name: str = "-"

    # Description: Model's author
    # Source: Specified as part of the .py file using the label "author"
    author: str = "-"

    # Description: Model's class
    # Source: Extracted by benchit() during analysis
    model_class: str = "-"

    # Description: Number of trainable parameters in the model
    # Source: Extracted by benchit() during analysis
    params: str = "-"

    # Description: Hash of the model's architecture
    # Source: Computed by benchit() during analysis
    hash: str = "-"

    # Description: License of the model
    # Source: Specified as part of the .py file using the label "license"
    license: str = "-"

    # Description: Task being performed by the model
    # Source: Specified as part of the .py file using the label "task"
    task: str = "-"

    # Description: Number of chips needed to run the model when mapped to Groq
    # Source: Computed by GroqFlow
    groq_chips_used: str = "-"

    # Description: Number of chips needed to run the model when mapped to Groq
    # Source: Computed by GroqFlow
    groq_chips_used: str = "-"

    # Description: An estimation of latency, in ms, to run one invocation of the
    #              model on a GroqNode server (includes I/O).
    # Source: computed by GroqFlow
    groq_estimated_latency: str = "-"

    # Description: Mean total latency, in ms, to run one invocation using TensorRT.
    # Source: benchit.benchmark() on 100 runs
    nvidia_latency: str = "-"

    # Description: Mean total latency, in ms, to run one invocation using OnnxRuntime.
    # Source: benchit.benchmark() on 100 runs
    x86_latency: str = "-"


def summary_spreadsheet(args) -> None:

    # Input arguments from CLI
    cache_dirs = [os.path.expanduser(dir) for dir in args.cache_dirs]
    report_dir = os.path.expanduser(args.report_dir)

    # Name report file
    report_path = os.path.join(report_dir, get_report_name())

    # Create report dict
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report = {}

    # Add results from all user-provided cache folders
    for cache_dir in cache_dirs:

        # Check if this is a valid cache directory
        check_cache_dir(cache_dir)

        # List all yaml files available
        all_model_state_yamls = cache.get_all(path=cache_dir, file_type="state.yaml")
        all_model_state_yamls = sorted(all_model_state_yamls)

        # Add each model to report
        for model_state_yaml in all_model_state_yamls:

            # Load state
            state = build.load_state(state_path=model_state_yaml)

            # Models are identified my the build name
            build_name = state.config.build_name

            # Add model to report if it doesn't exist
            if build_name not in report:
                report[build_name] = BuildResults()

            # Get model hash from build name
            # TODO: Get hash_model from state once it is consistently reported
            # https://github.com/groq/mlagility/issues/149
            report[build_name].hash = build_name.split("_")[-1]

            # Get model hash from build name
            report[build_name].params = _update_numeric_attribute(
                state.info.num_parameters,
                report[build_name].params,
                build_name=build_name,
                parameter_name="params",
            )

            # Extract labels (if any)
            parsed_labels = labels.load_from_cache(cache_dir, build_name)
            expected_attr_from_label = {
                "name": "model_name",
                "author": "author",
                "class": "model_class",
                "task": "task",
            }
            for label in expected_attr_from_label.keys():
                results_attr = expected_attr_from_label[label]
                if label not in parsed_labels.keys():
                    report[build_name].__dict__[results_attr] = "-"
                else:
                    report[build_name].__dict__[results_attr] = parsed_labels[label][0]

            # Get Groq latency and number of chips
            groq_estimated_latency = _get_estimated_groq_latency(build_name, cache_dir)
            report[build_name].groq_estimated_latency = _update_numeric_attribute(
                groq_estimated_latency,
                report[build_name].groq_estimated_latency,
                build_name=build_name,
                parameter_name="groq_estimated_latency",
            )
            report[build_name].groq_chips_used = _update_numeric_attribute(
                state.num_chips_used,
                report[build_name].groq_chips_used,
                build_name=build_name,
                parameter_name="groq_chips_used",
            )

            # Reloading state after estimating latency
            state = build.load_state(state_path=model_state_yaml)

            # Get CPU latency
            try:
                omodel = ORTModel(cache_dir, build_name)
                report[build_name].x86_latency = omodel.mean_latency
            except BenchmarkException:
                pass

            # Get GPU latency
            try:
                tmodel = TRTModel(cache_dir, build_name)
                report[build_name].nvidia_latency = tmodel.mean_latency
            except (BenchmarkException, KeyError):
                pass

    # Populate results spreadsheet
    with open(report_path, "w", newline="", encoding="utf8") as spreadsheet:
        writer = csv.writer(spreadsheet)
        cols = BuildResults().__dict__.keys()
        writer.writerow(cols)
        for model in report.keys():
            writer.writerow([vars(report[model])[col] for col in cols])

    # Print message with the output file path
    printing.log("Summary spreadsheet saved at ")
    printing.logn(str(report_path), printing.Colors.OKGREEN)
