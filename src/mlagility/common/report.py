import os
import csv
import json
from datetime import datetime
from pathlib import Path
import groqflow.common.printing as printing
import groqflow.common.build as build
import groqflow.common.cache as cache
from groqflow.groqmodel import groqmodel


def _numericCleanup(item, default="-"):
    return item if item is not None else default


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


class ModelResults:
    def __init__(self):
        self.model_name = "-"
        self.author = "-"
        self.model_class = "-"
        self.downloads = "-"
        self.params = "-"
        self.chips_used = "-"
        self.hash = "-"
        self.license = "-"
        self.task = "-"
        self.model_type = "-"
        self.groq_estimated_latency = "-"
        self.gpu_latency = "-"
        self.x86_latency = "-"


def summary_spreadsheet(args) -> str:

    # Input arguments from CLI
    cache_dirs = [os.path.expanduser(dir) for dir in args.cache_dirs]
    report_dir = os.path.expanduser(args.report_dir)

    # Name report file
    day = datetime.now().day
    month = datetime.now().month
    year = datetime.now().year
    date_key = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    summary_filename = f"{date_key}.csv"
    spreadsheet_file = os.path.join(report_dir, summary_filename)

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

            # Models are identified my the state.yaml path
            model_key = os.path.basename(model_state_yaml)

            # Add model to report if it doesn't exist
            if model_key not in report:
                report[model_key] = ModelResults()

            # Load state
            state = build.load_state(state_path=model_state_yaml)

            # Extract labels (if any)
            build_name = model_state_yaml.split("/")[-2]
            labels_file = f"{cache_dir}/labels/{build_name}.txt"
            with open(labels_file, encoding="utf-8") as f:
                labels = f.readline().split(" ")
            report[model_key].script_name = parse_labels("name", labels)
            report[model_key].author = parse_labels("author", labels)
            try:
                report[model_key].downloads = int(
                    parse_labels("downloads", labels).replace(",", "")
                )
            except ValueError:
                report[model_key].downloads = 0
            report[model_key].model_class = parse_labels("class", labels)
            report[model_key].task = parse_labels("task", labels)

            # Get Groq latency
            report[model_key].groq_estimated_latency = get_estimated_e2e_latency(
                build_name, cache_dir
            )

            # Reloading state after estimating latency
            state = build.load_state(state_path=model_state_yaml)

            # Get CPU latency
            cpu_output_dir = os.path.join(cache_dir, build_name, "x86_benchmark")
            cpu_stats_file = os.path.join(cpu_output_dir, "outputs.json")
            if os.path.isfile(cpu_stats_file):
                with open(cpu_stats_file, encoding="utf-8") as f:
                    g = json.load(f)
                report[model_key].x86_latency = g.get("Mean Latency(ms)", {})

            # Get GPU latency
            gpu_output_dir = os.path.join(cache_dir, build_name, "nvidia_benchmark")
            gpu_stats_file = os.path.join(gpu_output_dir, "outputs.json")
            if os.path.isfile(gpu_stats_file):
                with open(gpu_stats_file, encoding="utf-8") as f:
                    g = json.load(f)
                report[model_key].gpu_latency = (
                    g.get("Total Latency", {}).get("mean ", "-").split()[0]
                )

            # Get model hash from build name
            report[model_key].model_hash = state.config.build_name.split("_")[-1]

    # Populate spreadsheet
    with open(spreadsheet_file, "w", newline="", encoding="utf8") as spreadsheet:
        writer = csv.writer(spreadsheet)
        cols = ModelResults().__dict__.keys()
        writer.writerow(cols)
        for model in report.keys():
            writer.writerow([report[model].__dict__[col] for col in cols])

    # Print message with the output file path
    printing.log("Summary spreadsheet saved at ")
    printing.logn(str(spreadsheet_file), printing.Colors.OKGREEN)
    return spreadsheet_file
