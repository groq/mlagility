import os
from typing import Dict, Union, List
from onnxflow.common import printing
import onnxflow.common.build as build
from mlagility.analysis.util import ModelInfo


def update(models_found: Dict[str, ModelInfo]) -> None:
    """
    Prints all models and submodels found
    """
    if os.environ.get("MLAGILITY_DEBUG") != "True":
        os.system("clear")

    printing.logn(
        "\nModels discovered during profiling:\n",
        c=printing.Colors.BOLD,
    )
    recursive_print(models_found, None, None, [])


def recursive_print(
    models_found: Dict[str, ModelInfo],
    parent_model_hash: Union[str, None] = None,
    parent_workload_hash: Union[str, None] = None,
    script_names_visited: List[str] = False,
) -> None:
    script_names_visited = []

    for model_hash in models_found.keys():
        model_visited = False
        model_info = models_found[model_hash]
        workload_idx = 0
        for workload_hash in model_info.workloads.keys():
            workload = model_info.workloads[workload_hash]

            if (
                parent_model_hash == model_info.parent_hash
                and workload.executed > 0
                and (
                    model_info.workloads[workload_hash].parent_hash
                    == parent_workload_hash
                )
            ):
                print_file_name = False
                if model_info.script_name not in script_names_visited:
                    script_names_visited.append(model_info.script_name)
                    if model_info.depth == 0:
                        print_file_name = True

                print_workload(
                    model_info,
                    workload_hash,
                    print_file_name,
                    workload_idx=workload_idx,
                    model_visited=model_visited,
                )
                model_visited = True
                workload_idx += 1

                if print_file_name:
                    script_names_visited.append(model_info.script_name)

                recursive_print(
                    models_found,
                    parent_model_hash=model_hash,
                    parent_workload_hash=workload_hash,
                    script_names_visited=script_names_visited,
                )


def print_workload(
    model_info: ModelInfo,
    workload_hash: Union[str, None],
    print_file_name: bool = False,
    workload_idx: int = 0,
    model_visited: bool = False,
) -> None:
    """
    Print information about a given model or submodel
    """
    workload = model_info.workloads[workload_hash]
    ident = "\t" * (2 * model_info.depth + 1)
    if print_file_name:
        print(f"{model_info.script_name}.py:")

    if workload.exec_time == 0 or model_info.build_model:
        exec_time = ""
    else:
        exec_time = f" - {workload.exec_time:.2f}s"

    if model_info.depth == 0 and len(model_info.workloads) > 1:
        if not model_visited:
            printing.logn(f"{ident}{model_info.name}")
    else:
        printing.log(f"{ident}{model_info.name}")
        printing.logn(
            f" (executed {workload.executed}x{exec_time})",
            c=printing.Colors.OKGREEN,
        )

    if (model_info.depth == 0 and not model_visited) or (model_info.depth != 0):
        if model_info.depth == 0:
            if model_info.model_type == build.ModelType.PYTORCH:
                print(f"{ident}\tModel Type:\tPytorch (torch.nn.Module)")
            elif model_info.model_type == build.ModelType.KERAS:
                print(f"{ident}\tModel Type:\tKeras (tf.keras.Model)")

        # Display class of found model and the file where it was found
        model_class = type(model_info.model)
        print(f"{ident}\tClass:\t\t{model_class.__name__} ({model_class})")
        if model_info.depth == 0:
            print(f"{ident}\tLocation:\t{model_info.file}, line {model_info.line}")

        # Converting number of parameters to MB assuming 2 bytes per parameter
        model_size = model_info.params * 2 / (1024 * 1024)
        model_size = "{:.1f}".format(model_size) if model_size > 0.1 else "<0.1"
        print(
            f"{ident}\tParameters:\t{'{:,}'.format(model_info.params)} ({model_size} MB)"
        )

    if model_info.depth == 0 and len(model_info.workloads) > 1:
        printing.logn(
            f"\n{ident}\twith input shape {workload_idx+1} (executed {workload.executed}x{exec_time})",
            c=printing.Colors.OKGREEN,
        )

    # Prepare input shape to be printed
    input_shape = dict(model_info.workloads[workload_hash].input_shapes)
    input_shape = {key: value for key, value in input_shape.items() if value != ()}
    input_shape = str(input_shape).replace("{", "").replace("}", "")

    print(f"{ident}\tInput Shape:\t{input_shape}")
    print(f"{ident}\tHash:\t\t" + workload_hash)

    # Print benchit results if benchit was run
    if workload.performance:
        printing.log(f"{ident}\tStatus:\t\t")
        printing.logn(
            f"Successfully benchmarked on {workload.performance.device} ({workload.performance.runtime} v{workload.performance.runtime_version})",
            c=workload.status_message_color,
        )
        printing.logn(
            f"{ident}\t\t\tMean Latency:\t{workload.performance.mean_latency:.3f}"
            f"\t{workload.performance.latency_units}"
        )
        printing.logn(
            f"{ident}\t\t\tThroughput:\t{workload.performance.throughput:.1f}"
            f"\t{workload.performance.throughput_units}"
        )
        print()
    else:
        if workload.is_target and model_info.build_model:
            printing.log(f"{ident}\tStatus:\t\t")
            printing.logn(f"{workload.status_message}", c=workload.status_message_color)

            if workload.traceback is not None:
                if os.environ.get("MLAGILITY_TRACEBACK") != "False":
                    for line in workload.traceback:
                        for subline in line.split("\n")[:-1]:
                            print(f"{ident}\t{subline}")

                else:
                    printing.logn(
                        f"{ident}\t\t\tTo see the full stack trace, "
                        "rerun with `export MLAGILITY_TRACEBACK=True`.\n",
                        c=model_info.status_message_color,
                    )
            else:
                print()
        print()
