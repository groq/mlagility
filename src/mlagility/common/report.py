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

    # Add first row to report
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(spreadsheet_file, "w", newline="", encoding="utf8") as spreadsheet:
        writer = csv.writer(spreadsheet)
        writer.writerow(
            [
                "model_name",
                "author",
                "class",
                "downloads",
                "params",
                "chips_used",
                "hash",
                "license",
                "task",
                "model_type",
                "groq_estimated_latency",
                "gpu_latency",
                "x86_latency",
            ]
        )

    # Add results from all cache folders
    for cache_dir in cache_dirs:
        cache_dir = os.path.expanduser(cache_dir)

        with open(spreadsheet_file, "w", newline="", encoding="utf8") as spreadsheet:
            writer = csv.writer(spreadsheet)

            all_model_state_yamls = cache.get_all(
                path=cache_dir, file_type="state.yaml"
            )
            all_model_state_yamls = sorted(all_model_state_yamls)

            # Print stats for each model
            print(len(all_model_state_yamls), "yaml files found")
            for model_state_yaml in all_model_state_yamls:

                state = build.load_state(state_path=model_state_yaml)

                # Extract labels (if any)
                build_name = model_state_yaml.split("/")[-2]
                labels_file = f"{cache_dir}/labels/{build_name}.txt"
                with open(labels_file, encoding="utf-8") as f:
                    labels = f.readline().split(" ")
                script_name = parse_labels("name", labels)
                author = parse_labels("author", labels)
                try:
                    downloads = int(parse_labels("downloads", labels).replace(",", ""))
                except ValueError:
                    downloads = 0
                model_class = parse_labels("class", labels)
                task = parse_labels("task", labels)

                # GroqChip estimated e2e latency
                groq_estimated_latency = get_estimated_e2e_latency(
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
                    x86_latency = g.get("Mean Latency(ms)", {})
                else:
                    x86_latency = "-"

                # Get GPU latency
                gpu_output_dir = os.path.join(cache_dir, build_name, "nvidia_benchmark")
                gpu_stats_file = os.path.join(gpu_output_dir, "outputs.json")
                if os.path.isfile(gpu_stats_file):
                    with open(gpu_stats_file, encoding="utf-8") as f:
                        g = json.load(f)
                    print(gpu_stats_file)
                    gpu_latency = (
                        g.get("Total Latency", {}).get("mean ", "-").split()[0]
                    )
                else:
                    gpu_latency = "-"

                name = state.config.build_name
                name = name.split("_")
                model_hash = name[-1]

                writer.writerow(
                    [
                        script_name,
                        author,
                        model_class,
                        downloads,
                        _numericCleanup(state.info.num_parameters),
                        _numericCleanup(state.num_chips_used),
                        model_hash,
                        license,
                        task,
                        groq_estimated_latency,
                        gpu_latency,
                        x86_latency,
                    ]
                )

    # Print message with the output file path
    printing.log("Summary spreadsheet saved at ")
    printing.logn(str(spreadsheet_file), printing.Colors.OKGREEN)
    return spreadsheet_file
