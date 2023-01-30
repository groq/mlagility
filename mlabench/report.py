import os
import csv
import dataclasses
from typing import List
import yaml
import groqflow.common.printing as printing
import groqflow.common.build as build
import groqflow.common.cache as cache
from groqflow import groqmodel
from mlabench.benchmark import gpumodel


def _successCleanup(item):
    return item if item is not None else False


def _numericCleanup(item):
    return item if item is not None else "-"


def _cleanOpList(input):
    if input is None:
        return "3P_failure"
    elif input == []:
        return "-"
    else:
        return "; ".join(map(str, input))


def _getLog(model: build.State, logname, cache_dir):
    try:
        with open(
            os.path.join(
                build.output_dir(cache_dir, model.config.build_name),
                logname,
            ),
            "r",
            encoding="utf8",
        ) as log:
            return log.read()
    except FileNotFoundError:
        return "No log file"


summary_filename = "summary.csv"
op_support_filename = "op_support.csv"


def _buildResult(model: build.State, cache_dir):

    if not model.info.base_onnx_exported:
        # FIXME: we don't have a useful separation of
        # the different "export" flows yet, which confuses
        # this reporting. Come back and update this after
        # we separate ONNX file input vs. torch model input
        # vs. etc. better
        result = "3P_torch_onnx_export_failure"
        log_export_file = ""
        if model.model_type is build.ModelType.PYTORCH:
            log_export_file = "log_export_pytorch.txt"
        elif model.model_type is build.ModelType.KERAS:
            log_export_file = "log_export_keras.txt"
        elif model.model_type is build.ModelType.ONNX_FILE:
            log_export_file = "log_export_onnx.txt"
        else:
            raise ValueError(f"Model type {model.model_type} not supported")
        log = _getLog(model, log_export_file, cache_dir)
    elif not model.info.opt_onnx_exported:
        result = "3P_ort_failure"
        log = _getLog(model, "log_optimize.txt", cache_dir)
    elif not model.info.opt_onnx_all_ops_supported:
        result = "op_support_failure"
        log = _getLog(model, "log_check_compatibility.txt", cache_dir)
    elif not model.info.converted_onnx_exported:
        result = "3P_onnxmltools_failure"
        log = _getLog(model, "log_fp16_conversion.txt", cache_dir)
    elif not model.info.compiler_success:
        result = "compiler_failure"
        log = _getLog(model, "log_compile.txt", cache_dir)
    elif not model.info.assembler_success:
        result = "assembler_failure"
        log = _getLog(model, "log_assemble.txt", cache_dir)
    else:
        result = "success"
        log = None

    return result, log


@dataclasses.dataclass
class ResultsColumn:
    title: str
    description: str
    values: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Results:
    model_name: ResultsColumn = ResultsColumn(
        "Model Name",
        "Name of the model specified with the build_name arg. "
        "Default name is the filename of the Python script "
        "that ran the model.",
    )

    model_type: ResultsColumn = ResultsColumn(
        "Model Type",
        "Name of the model framework used to instantiate the "
        "model (e.g., PyTorch, TensorFlow, etc.)",
    )

    sequence: ResultsColumn = ResultsColumn(
        "Sequence",
        "List of build stages applied by groqit() to transform "
        "the model into IOP(s)",
    )

    corpus: ResultsColumn = ResultsColumn(
        "Corpus",
        "Name of the corpus that the model is a member of. "
        "Default corpus name is the name of the folder that "
        "holds the Python script that ran the model.",
    )

    unsupported_ops: ResultsColumn = ResultsColumn(
        "Unsupported Ops List",
        "Indicates whether all ONNX ops in the optimized ONNX model "
        "are supported by Groq Compiler. Reports “None” when all ops are "
        "supported, or provides a list of unsupported ops.",
    )

    parameters: ResultsColumn = ResultsColumn(
        "Parameters",
        "Number of constant elements (ie, trained parameters) in the model.",
    )

    inputs: ResultsColumn = ResultsColumn(
        "Inputs",
        "List of shapes (dimensions) and data type of each argument "
        "to the model’s forward function.",
    )

    outputs: ResultsColumn = ResultsColumn(
        "Outputs",
        "List of shapes (dimensions) and data type of each output "
        "from  the model's forward function.",
    )

    chips_estimated: ResultsColumn = ResultsColumn(
        "# Chips Estimated",
        "Minimum number of GroqChip processors that could be used to "
        "instantiate the model (note: GroqFlow may round up to a higher number "
        "to fit an available topology).",
    )

    chips_used: ResultsColumn = ResultsColumn(
        "# Chips Used",
        "Actual number of GroqChip processors targeted by the compilation process.",
    )

    estimated_groq_latency: ResultsColumn = ResultsColumn(
        "Estimated Groq Latency (seconds)",
        "An estimation of latency, in seconds, to run one invocation of the "
        "model on a GroqNode server (includes I/O).",
    )

    estimated_groq_throughput: ResultsColumn = ResultsColumn(
        "Estimated Groq Throughput (inferences per second)",
        "An estimation of throughput, in inferences per second, "
        "to run one invocation of the model on a GroqNode server (includes I/O).",
    )

    measured_groq_latency: ResultsColumn = ResultsColumn(
        "Measured Groq Latency (seconds)",
        "An measurement of latency, in seconds, to run one invocation of the "
        "model on a GroqNode server (includes I/O).",
    )

    measured_groq_throughput: ResultsColumn = ResultsColumn(
        "Measured Groq Throughput (inferences per second)",
        "A measurement of throughput, in inferences per second, for "
        "running many invocations of the model on a GroqNode server (includes I/O).",
    )

    measured_groq_power: ResultsColumn = ResultsColumn(
        "Measured Groq Power (Watts)",
        "An measurement of power draw, in Watts, incurred while running "
        "many invocations of the model on a GroqNode server.",
    )

    measured_gpu_latency: ResultsColumn = ResultsColumn(
        "Measured Groq Latency (seconds)",
        "An measurement of latency, in seconds, to run one invocation of the "
        "model on a GPU server in GCP (includes I/O).",
    )

    measured_gpu_throughput: ResultsColumn = ResultsColumn(
        "Measured Groq Throughput (inferences per second)",
        "An measurement of throughput, in inferences per second, for "
        "running many invocations of the model on a GPU server in GCP "
        "(includes I/O).",
    )

    result: ResultsColumn = ResultsColumn(
        "Result",
        "Result status of building the model with GroqFlow See "
        "this document for the meaning of the result codes: "
        # pylint: disable=line-too-long
        "https://docs.google.com/document/d/1dQ5fxjFHAwEcAF-hxj69OPefACmv1Q0wBBSFFQtb56Q/edit?usp=sharing.",
    )

    build_duration: ResultsColumn = ResultsColumn(
        "Build Duration (seconds)",
        "Number of seconds spent running GroqFlow on the model",
    )

    compiler_duration: ResultsColumn = ResultsColumn(
        "Compiler Duration (seconds)",
        "Number of seconds spent running Groq Compiler on the model",
    )

    assembler_duration: ResultsColumn = ResultsColumn(
        "Assembler Duration (seconds)",
        "Number of seconds spent running Groq Assembler on the model",
    )

    def add_build(
        self,
        model_name,
        model_type,
        sequence,
        corpus,
        unsupported_ops,
        parameters,
        inputs,
        outputs,
        chips_estimated,
        chips_used,
        estimated_groq_latency,
        estimated_groq_throughput,
        measured_groq_latency,
        measured_groq_throughput,
        measured_groq_power,
        measured_gpu_latency,
        measured_gpu_throughput,
        result,
        build_duration,
        compiler_duration,
        assembler_duration,
    ):
        self.model_name.values.append(model_name)
        self.model_type.values.append(model_type)
        self.sequence.values.append(sequence)
        self.corpus.values.append(corpus)
        self.unsupported_ops.values.append(unsupported_ops)
        self.parameters.values.append(parameters)
        self.inputs.values.append(inputs)
        self.outputs.values.append(outputs)
        self.chips_estimated.values.append(chips_estimated)
        self.chips_used.values.append(chips_used)
        self.estimated_groq_latency.values.append(estimated_groq_latency)
        self.estimated_groq_throughput.values.append(estimated_groq_throughput)
        self.measured_groq_latency.values.append(measured_groq_latency)
        self.measured_groq_throughput.values.append(measured_groq_throughput)
        self.measured_groq_power.values.append(measured_groq_power)
        self.measured_gpu_latency.values.append(measured_gpu_latency)
        self.measured_gpu_throughput.values.append(measured_gpu_throughput)
        self.result.values.append(result)
        self.build_duration.values.append(build_duration)
        self.compiler_duration.values.append(compiler_duration)
        self.assembler_duration.values.append(assembler_duration)

    def _write_legend(self, writer):
        # Git commit
        writer.writerow(
            [
                "Git Commit: TODO",
            ]
        )

        # GroqFlow version
        writer.writerow(
            [
                "GroqFlow Version: TODO",
            ]
        )

        # GroqWare SDK version
        writer.writerow(
            [
                "GroqWare SDK Version: TODO",
            ]
        )

        # GPU device
        writer.writerow(
            [
                "GPU device (TensorRT version): TODO",
            ]
        )

        # Legend
        writer.writerow(
            [
                "Legend",
            ]
        )

        # Write the title and the description for each column
        for column in vars(self).values():
            writer.writerow([column.title, column.description])

    def _write_headers(self, writer):
        headers = [column.title for column in vars(self).values()]
        writer.writerow(headers)

    def _write_rows(self, writer):
        row_count = len(vars(self)["model_name"].values)
        rows = [
            [column.values[row_idx] for column in vars(self).values()]
            for row_idx in range(row_count)
        ]
        for row in rows:
            writer.writerow(row)

    def write(self, filename):
        with open(filename, "w", newline="", encoding="utf8") as spreadsheet:
            writer = csv.writer(spreadsheet)

            self._write_legend(writer)
            self._write_headers(writer)
            self._write_rows(writer)


# class CsvFile:
#     def __init__(self, writer, columns):
#         self.writer = writer
#         self.columns = columns

#     def write_legend(self):
#         # Git commit
#         self.writer.writerow(
#             [
#                 "Git Commit: TODO",
#             ]
#         )

#         # GroqFlow version
#         self.writer.writerow(
#             [
#                 "GroqFlow Version: TODO",
#             ]
#         )

#         # GroqWare SDK version
#         self.writer.writerow(
#             [
#                 "GroqWare SDK Version: TODO",
#             ]
#         )

#         # GPU device
#         self.writer.writerow(
#             [
#                 "GPU device (TensorRT version): TODO",
#             ]
#         )

#         # Legend
#         self.writer.writerow(
#             [
#                 "Legend",
#             ]
#         )

#         for title in self.columns:
#             # Write the title and the description
#             self.writer.writerow([title, self.columns[title]])

#     def write_headers(self):
#         headers = [title for title in self.columns]
#         self.writer.writerow(headers)

#     def write_rows(self):
#         for title in self.columns:
#             self.write_rows(title[VALUES])

#     def add_value(self, column, value):
#         columns[column][VALUES].append(value)


def _get_gpu_measured_latency(state: build.State) -> str:
    return gpumodel.GPUMeasuredPerformance(
        os.path.join(
            state.cache_dir,
            state.config.build_name,
            "gpu_performance.json",
        )
    ).latency


def _get_gpu_measured_throughput(state: build.State) -> str:
    return gpumodel.GPUMeasuredPerformance(
        os.path.join(
            state.cache_dir,
            state.config.build_name,
            "gpu_performance.json",
        )
    ).throughput


def summary_spreadsheet(args):

    # Declare file paths
    spreadsheet_file = os.path.join(args.cache_dir, summary_filename)
    results_file = os.path.join(args.cache_dir, "groqflow_results.yaml")

    # results_csv = CsvFile(writer=writer, columns=columns)
    results_csv = Results()

    # with open(spreadsheet_file, "w", newline="", encoding="utf8") as spreadsheet:
    #     writer = csv.writer(spreadsheet)

    all_model_state_yamls = cache.get_all(path=args.cache_dir, file_type="state.yaml")
    all_model_state_yamls = sorted(all_model_state_yamls)

    results_dict = {}

    # Print stats for each model
    for model_state_yaml in all_model_state_yamls:
        state = build.load_state(state_path=model_state_yaml)

        # Get estimated number of chips needed
        estimated_num_chips = build.calculate_num_chips(
            state.info.num_parameters, estimate=True
        )

        # Get the performance estimate for the build
        if (
            state.build_status == build.Status.SUCCESSFUL_BUILD
            and "assemble" in state.info.completed_build_stages
        ):
            gmodel = groqmodel.load(state.config.build_name, args.cache_dir)
            estimated_perf = gmodel.estimate_performance()
            estimated_latency = estimated_perf.latency
            estimated_throughput = estimated_perf.throughput
        else:
            estimated_latency = "-"
            estimated_throughput = "-"

        # Get the result of the build and put the log
        # from the failed stage (if any) into a dict
        result, log = _buildResult(state, args.cache_dir)
        repro_command = f"groqit-util run_all -m {state.config.build_name}"
        results_dict[state.config.build_name] = {
            "repro_command": repro_command,
            "build_result": result,
            "log": log,
            "state": vars(state),
        }

        results_csv.add_build(
            model_name=state.config.build_name,
            model_type="TODO",
            sequence="TODO",
            corpus=state.corpus,
            unsupported_ops="TODO",
            parameters=state.info.num_parameters,
            inputs="TODO",
            outputs="TODO",
            chips_estimated=estimated_num_chips,
            chips_used=state.num_chips_used,
            estimated_groq_latency=estimated_latency,
            estimated_groq_throughput=estimated_throughput,
            measured_groq_latency=state.info.measured_latency,
            measured_groq_throughput=state.info.measured_throughput,
            measured_groq_power="TODO",
            measured_gpu_latency="TODO",
            measured_gpu_throughput="TODO",
            result=result,
            build_duration="TODO",
            compiler_duration="TODO",
            assembler_duration="TODO",
        )

    # results_csv.write_legend()
    # results_csv.write_headers()
    results_csv.write(spreadsheet_file)

    # Print message with the output file path
    printing.log("Summary spreadsheet saved at ")
    printing.logn(str(spreadsheet_file), printing.Colors.OKGREEN)

    # Make the logs.yaml file
    with open(results_file, "w", encoding="utf8") as outfile:
        yaml.dump(results_dict, outfile)

    # Print message with the output file path
    printing.log("Build results across all models collected at ")
    printing.logn(str(results_file), printing.Colors.OKGREEN)


def op_spreadsheet(cache_dir):

    # Declare file paths
    spreadsheet_file = os.path.join(cache_dir, op_support_filename)

    with open(spreadsheet_file, "w", newline="", encoding="utf8") as spreadsheet:
        writer = csv.writer(spreadsheet)

        # Git commit placeholder
        writer.writerow(
            [
                "Git Commit: ",
            ]
        )

        writer.writerow(
            [
                "Model Name",
                "Unsupported ops (Optimized ONNX)",
            ]
        )

        all_model_state_yamls = cache.get_all(path=cache_dir, file_type="state.yaml")
        all_model_state_yamls = sorted(all_model_state_yamls)

        for model_state_yaml in all_model_state_yamls:
            state = build.load_state(state_path=model_state_yaml)

            writer.writerow(
                [
                    state.config.build_name,  # Model Name
                    _cleanOpList(
                        state.info.opt_onnx_unsupported_ops
                    ),  # Unsupported ops (Optimized ONNX)
                ]
            )

    # Print message with the output file path
    printing.log("Op support spreadsheet saved at ")
    printing.logn(str(spreadsheet_file), printing.Colors.OKGREEN)
