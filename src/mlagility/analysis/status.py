import os
from typing import Dict, Union, List
from groqflow.common import printing
import groqflow.common.build as build
from mlagility.analysis.util import ModelInfo


def update(models_found: Dict[str, ModelInfo]) -> None:
    """
    Prints all models and submodels found
    """
    if os.environ.get("MLAGILITY_DEBUG")!="True":
        os.system("clear")

    printing.logn(
        "\nModels discovered during profiling:\n",
        c=printing.Colors.BOLD,
    )
    recursive_print(models_found, None, [])


def recursive_print(
    models_found: Dict[str, ModelInfo],
    parent_hash: Union[str, None] = None,
    script_names_visited: List[str] = False,
) -> None:
    script_names_visited = []

    for h in models_found.keys():
        if parent_hash == models_found[h].parent_hash and models_found[h].executed > 0:
            print_file_name = models_found[h].script_name not in script_names_visited

            print_model(models_found[h], h, print_file_name)

            if print_file_name:
                script_names_visited.append(models_found[h].script_name)

            recursive_print(
                models_found, parent_hash=h, script_names_visited=script_names_visited
            )


def print_model(
    model_info: ModelInfo, model_hash: Union[str, None], print_file_name: bool = False
) -> None:
    """
    Print information about a given model or submodel
    """
    ident = "\t" * (2 * model_info.depth + 1)
    if print_file_name:
        print(f"{model_info.script_name}.py:")
    printing.log(f"{ident}{model_info.name} ")

    # Show the number of times the model has been executed
    # Only show the execution time if we are not running benchit() as this
    # impacts time measurement.
    if model_info.exec_time == 0 or model_info.build_model:
        exec_time = ""
    else:
        exec_time = f" - {model_info.exec_time:.2f}s"
    printing.logn(
        f"(executed {model_info.executed}x{exec_time})",
        c=printing.Colors.OKGREEN,
    )

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
    print(f"{ident}\tParameters:\t{'{:,}'.format(model_info.params)} ({model_size} MB)")
    print(f"{ident}\tHash:\t\t" + model_hash)

    # Print benchit results if benchit was run
    if model_info.performance:
        printing.log(f"{ident}\tStatus:\t\t")
        printing.logn(
            f"Model successfully benchmarked on {model_info.performance.device}",
            c=model_info.status_message_color,
        )
        printing.logn(
            f"{ident}\t\t\tMean Latency:\t{model_info.performance.mean_latency:.3f}"
            f"\t{model_info.performance.latency_units}"
        )
        printing.logn(
            f"{ident}\t\t\tThroughput:\t{model_info.performance.throughput:.1f}"
            f"\t{model_info.performance.throughput_units}"
        )
        print()
    else:
        if model_info.is_target and model_info.build_model:
            printing.log(f"{ident}\tStatus:\t\t")
            printing.logn(
                f"{model_info.status_message}\n", c=model_info.status_message_color
            )
        else:
            print("")
