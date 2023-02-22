import os
import csv
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import groqflow.common.printing as printing
import groqflow.common.build as build
import groqflow.common.cache as cache
from groqflow.groqmodel import groqmodel


def _numericCleanup(new_val, current_val, default="-"):
    if current_val == default:
        return new_val if new_val is not None else default
    else:
        return current_val


def parse_labels(key, label_list):
    for label in label_list:
        if key + "::" in label:
            parsed_label = label[len(key) + 2 :].replace("_", " ")
            return "-" if parsed_label == "unknown" else parsed_label
    return "-"


def get_estimated_e2e_latency(model_folder, cache_folder):
    """
    Returns estimated e2e latency (io + compute - not including runtime) in ms
    """
    try:
        gmodel = groqmodel.load(model_folder, cache_folder)
        if (
            gmodel.state.info.assembler_success  # pylint: disable=singleton-comparison
            == True
        ):
            return 1000 * gmodel.estimate_performance().latency
        else:
            return "-"
    except:  # pylint: disable=bare-except
        return "-"


def get_report_name():
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


def summary_spreadsheet(args) -> str:

    # Input arguments from CLI
    cache_dirs = [os.path.expanduser(dir) for dir in args.cache_dirs]
    report_dir = os.path.expanduser(args.report_dir)

    # Name report file
    report_path = os.path.join(report_dir, get_report_name())

    # Create report dict
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report = {}

    # Add results from all cache folders
    for cache_dir in cache_dirs:

        # List all yaml files available
        all_model_state_yamls = cache.get_all(path=cache_dir, file_type="state.yaml")
        all_model_state_yamls = sorted(all_model_state_yamls)

        # Add each model to report
        for model_state_yaml in all_model_state_yamls:

            # Models are identified my the build name
            build_name = model_state_yaml.split("/")[-2]

            # Load state
            state = build.load_state(state_path=model_state_yaml)

            # Add model to report if it doesn't exist
            if build_name not in report:
                report[build_name] = BuildResults()

            # Get model hash from build name
            report[build_name].model_hash = state.config.build_name.split("_")[-1]

            # Get model hash from build name
            report[build_name].params = _numericCleanup(
                state.info.num_parameters, report[build_name].params
            )

            # Extract labels (if any)
            build_name = model_state_yaml.split("/")[-2]
            labels_file = f"{cache_dir}/labels/{build_name}.txt"
            with open(labels_file, encoding="utf-8") as f:
                labels = f.readline().split(" ")
            report[build_name].model_name = parse_labels("name", labels)
            report[build_name].author = parse_labels("author", labels)
            report[build_name].model_class = parse_labels("class", labels)
            report[build_name].task = parse_labels("task", labels)

            # Get Groq latency and number of chips
            groq_estimated_latency = get_estimated_e2e_latency(build_name, cache_dir)
            report[build_name].groq_estimated_latency = _numericCleanup(
                groq_estimated_latency, report[build_name].groq_estimated_latency
            )
            report[build_name].groq_chips_used = _numericCleanup(
                state.num_chips_used, report[build_name].groq_chips_used
            )

            # Reloading state after estimating latency
            state = build.load_state(state_path=model_state_yaml)

            # Get CPU latency
            x86_output_dir = os.path.join(cache_dir, build_name, "x86_benchmark")
            x86_stats_file = os.path.join(x86_output_dir, "outputs.json")
            if os.path.isfile(x86_stats_file):
                with open(x86_stats_file, encoding="utf-8") as f:
                    g = json.load(f)
                report[build_name].x86_latency = g.get("Mean Latency(ms)", {})

            # Get GPU latency
            nvidia_output_dir = os.path.join(cache_dir, build_name, "nvidia_benchmark")
            nvidia_stats_file = os.path.join(nvidia_output_dir, "outputs.json")
            if os.path.isfile(nvidia_stats_file):
                with open(nvidia_stats_file, encoding="utf-8") as f:
                    g = json.load(f)
                report[build_name].nvidia_latency = (
                    g.get("Total Latency", {}).get("mean ", "-").split()[0]
                )

    # Populate spreadsheet
    with open(report_path, "w", newline="", encoding="utf8") as spreadsheet:
        writer = csv.writer(spreadsheet)
        cols = BuildResults().__dict__.keys()
        writer.writerow(cols)
        for model in report.keys():
            writer.writerow([report[model].__dict__[col] for col in cols])

    # Print message with the output file path
    printing.log("Summary spreadsheet saved at ")
    printing.logn(str(report_path), printing.Colors.OKGREEN)
    return report_path
