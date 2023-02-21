import os
import csv
import json
from datetime import datetime
from pathlib import Path
import groqflow.common.printing as printing
import groqflow.common.build as build
import groqflow.common.cache as cache
from groqflow.groqmodel import groqmodel


def _successCleanup(item):
    return item if item is not None else False


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


def report(cache_dir: str) -> str:
    return summary_spreadsheet(
        groq_cache_dir=cache_dir, gpu_cache_dir=cache_dir, cpu_cache_dir=cache_dir
    )


def summary_spreadsheet(
    groq_cache_dir: str,
    gpu_cache_dir: str,
    cpu_cache_dir: str,
    report_folder: str = os.getcwd(),
) -> str:

    # Declare file paths
    groq_cache_dir = os.path.expanduser(groq_cache_dir)
    day = datetime.now().day
    month = datetime.now().month
    year = datetime.now().year
    Path(report_folder).mkdir(parents=True, exist_ok=True)
    date_key = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    summary_filename = f"{report_folder}/{date_key}.csv"
    spreadsheet_file = os.path.join(groq_cache_dir, summary_filename)

    with open(spreadsheet_file, "w", newline="", encoding="utf8") as spreadsheet:
        writer = csv.writer(spreadsheet)

        all_model_state_yamls = cache.get_all(
            path=groq_cache_dir, file_type="state.yaml"
        )
        all_model_state_yamls = sorted(all_model_state_yamls)

        writer.writerow(
            [
                "model_name",
                "author",
                "class",
                "downloads",
                "assembles",
                "params",
                "chips_used",
                "hash",
                "license",
                "task",
                "model_type",
                "cycles",
                "tsp_compute_latency",  # Groqchip compute latency (ms)
                "gpu_compute_latency",  # NVIDIA A100 compute latency (ms)
                "tsp_gpu_compute_ratio",  # Compute Only GroqChip/ NVIDIA A100
                "tsp_estimated_e2e_latency",  # Groqchip estimated E2E latency (ms)
                "gpu_e2e_latency",  # NVIDIA A100 E2E latency (ms)
                "tsp_gpu_e2e_ratio",  # E2E GroqChip/ NVIDIA A100
                "x86_latency",
            ]
        )

        # Print stats for each model
        print(len(all_model_state_yamls), "yaml files found")
        for model_state_yaml in all_model_state_yamls:

            state = build.load_state(state_path=model_state_yaml)

            # Extract labels (if any)
            build_name = model_state_yaml.split("/")[-2]
            labels_file = f"{groq_cache_dir}/labels/{build_name}.txt"
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

            # Get number of cycles
            output_dir = os.path.join(groq_cache_dir, build_name)
            stats_file = os.path.join(output_dir, "compile", "stats.json")
            if os.path.isfile(stats_file):
                with open(stats_file, encoding="utf-8") as f:
                    d = json.load(f)
                total_cycles = d["total_cycles"]
            else:
                total_cycles = "-"

            # GroqChip estimated e2e latency
            groqchip_estimated_e2e_latency = get_estimated_e2e_latency(
                build_name, groq_cache_dir
            )

            # Reloading state after estimating latency
            state = build.load_state(state_path=model_state_yaml)

            # Get deterministic latency and convert from s to ms
            if state.info.deterministic_compute_latency is not None:
                groqchip_compute_latency = (
                    1000 * state.info.deterministic_compute_latency
                )
            else:
                groqchip_compute_latency = "-"

            # Get CPU latency
            cpu_output_dir = os.path.join(cpu_cache_dir, build_name, "x86_benchmark")
            cpu_stats_file = os.path.join(cpu_output_dir, "outputs.json")
            if os.path.isfile(cpu_stats_file):
                with open(cpu_stats_file, encoding="utf-8") as f:
                    g = json.load(f)
                x86_latency = g.get("Mean Latency(ms)", {})
            else:
                x86_latency = "-"

            # Get GPU latency
            gpu_output_dir = os.path.join(gpu_cache_dir, build_name, "nvidia_benchmark")
            gpu_stats_file = os.path.join(gpu_output_dir, "outputs.json")
            if os.path.isfile(gpu_stats_file):
                with open(gpu_stats_file, encoding="utf-8") as f:
                    g = json.load(f)
                print(gpu_stats_file)
                gpu_e2e_latency = (
                    g.get("Total Latency", {}).get("mean ", "-").split()[0]
                )
            else:
                gpu_compute_latency = "-"
                gpu_e2e_latency = "-"

            if gpu_compute_latency != "-" and groqchip_compute_latency != "-":
                compute_groqchip_vs_gpu = (
                    float(gpu_compute_latency) / groqchip_compute_latency
                )
            else:
                compute_groqchip_vs_gpu = "-"

            if gpu_e2e_latency != "-" and groqchip_estimated_e2e_latency != "-":
                e2e_groqchip_vs_gpu = (
                    float(gpu_e2e_latency) / groqchip_estimated_e2e_latency
                )
            else:
                e2e_groqchip_vs_gpu = "-"

            model_type = state.model_type.value
            name = state.config.build_name
            name = name.split("_")
            model_hash = name[-1]

            writer.writerow(
                [
                    script_name,  # Model Name
                    author,  # Model Name
                    model_class,  # Model Name
                    downloads,  # monthly downloads
                    _successCleanup(state.info.assembler_success)
                    and _successCleanup(state.info.compiler_success),  # Generates iop
                    _numericCleanup(state.info.num_parameters),  # Params
                    _numericCleanup(state.num_chips_used),  # chips (used)
                    model_hash,  # hash
                    license,  # license
                    task,  # task
                    model_type,  # model_type
                    total_cycles,
                    groqchip_compute_latency,  # Latency of Groqchip
                    gpu_compute_latency,  # Compute only latency of GPU
                    compute_groqchip_vs_gpu,  # Compute only comparison
                    groqchip_estimated_e2e_latency,  # Groqchip estimated end to end latency
                    gpu_e2e_latency,  # GPU end to end latency
                    e2e_groqchip_vs_gpu,  # End to end comparison
                    x86_latency,
                ]
            )

    # Print message with the output file path
    printing.log("Summary spreadsheet saved at ")
    printing.logn(str(spreadsheet_file), printing.Colors.OKGREEN)
    return spreadsheet_file
