import argparse
import os
import sys
from difflib import get_close_matches
import onnxflow.common.build as build
import onnxflow.common.exceptions as exceptions
import mlagility.common.filesystem as filesystem
import mlagility.cli.report as report
from mlagility.api.script_api import benchmark_files
from mlagility.version import __version__ as mlagility_version
from mlagility.api.devices import SUPPORTED_DEVICES, DEFAULT_RUNTIME


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)

    def print_cache_help(self):
        print("Error: a cache command is required")
        self.print_help()
        sys.exit(2)


def print_version(_):
    """
    Print the package version number
    """
    print(mlagility_version)


def print_stats(args):
    state_path = build.state_file(args.cache_dir, args.build_name)
    filesystem.print_yaml_file(state_path, "GroqFlow build state")

    filesystem.print_yaml_file(
        filesystem.stats_file(args.cache_dir, args.build_name), "MLAgility stats"
    )


def benchmark_command(args):
    """
    Map the argparse args into benchmark_files() arguments
    """

    benchmark_files(
        input_files=args.input_files,
        use_slurm=args.use_slurm,
        process_isolation=args.process_isolation,
        lean_cache=args.lean_cache,
        cache_dir=args.cache_dir,
        labels=args.labels,
        rebuild=args.rebuild,
        device=args.device,
        backend=args.backend,
        runtimes=args.runtimes,
        analyze_only=args.analyze_only,
        build_only=args.build_only,
        export_only=args.export_only,
        resume=args.resume,
        script_args=args.script_args,
        max_depth=args.max_depth,
        onnx_opset=args.onnx_opset,
        sequence=args.sequence_file,
        groq_compiler_flags=args.groq_compiler_flags,
        groq_assembler_flags=args.groq_assembler_flags,
        groq_num_chips=args.groq_num_chips,
        groqview=args.groqview,
    )


def main():
    """
    Parses arguments passed by user and forwards them into a
    command function
    """

    parser = MyParser(
        description="MLAgility benchmarking command line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # We use sub-parsers to keep the help info neatly organized for each command
    # Sub-parses also allow us to set command-specific help on options like --cache-dir
    # that are used in multiple commands

    subparsers = parser.add_subparsers(
        title="command",
        help="Choose one of the following commands:",
        metavar="COMMAND",
        required=True,
    )

    #######################################
    # Parser for the "benchmark" command
    #######################################

    def check_extension(choices, file_name):
        _, extension = os.path.splitext(file_name.split("::")[0])
        if extension[1:].lower() not in choices:
            raise exceptions.ArgError(
                f"input_files must end with .py or .onnx (got '{file_name}')"
            )
        return file_name

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark the performance of one or more models"
    )
    benchmark_parser.set_defaults(func=benchmark_command)

    benchmark_parser.add_argument(
        "input_files",
        nargs="+",
        help="One or more script (.py) or ONNX (.onnx) files to be benchmarked",
        type=lambda file: check_extension(("py", "onnx"), file),
    )

    slurm_or_processes_group = benchmark_parser.add_mutually_exclusive_group()

    slurm_or_processes_group.add_argument(
        "--use-slurm",
        dest="use_slurm",
        help="Execute on Slurm instead of using local compute resources",
        action="store_true",
    )

    slurm_or_processes_group.add_argument(
        "--process-isolation",
        dest="process_isolation",
        help="Isolate evaluating each input into a separate process",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--lean-cache",
        dest="lean_cache",
        help="Delete all build artifacts except for log files when the command completes",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="Build cache directory where the resulting build directories will "
        f"be stored (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    benchmark_parser.add_argument(
        "--labels",
        dest="labels",
        help="Only benchmark the scripts that have the provided labels",
        nargs="*",
        default=[],
    )

    benchmark_parser.add_argument(
        "--sequence-file",
        dest="sequence_file",
        help="Path to a python script that implements the GroqFlow sequence.py template, "
        "which returns an instance of a custom build Sequence that will be passed to the "
        "groqit(sequence=...) arg (default behavior is to use the default groqit() "
        "build sequence)",
        required=False,
        default=None,
    )

    benchmark_parser.add_argument(
        "--rebuild",
        dest="rebuild",
        help=f"Sets the cache rebuild policy (defaults to {build.DEFAULT_REBUILD_POLICY})",
        required=False,
        default=build.DEFAULT_REBUILD_POLICY,
    )

    benchmark_default_backend = "local"
    argparse_backend = benchmark_parser.add_argument(
        "--backend",
        choices=[
            "local",
            "remote",
        ],
        dest="backend",
        help="Indicates whether the device is installed on the local machine or a remote machine "
        f'(defaults to "{benchmark_default_backend}")',
        required=False,
        default=benchmark_default_backend,
    )

    benchmark_default_device = "x86"
    benchmark_parser.add_argument(
        "--device",
        choices=SUPPORTED_DEVICES,
        nargs="?",
        dest="device",
        help="Type of hardware device to be used for the benchmark "
        f'(defaults to "{benchmark_default_device}")',
        required=False,
        default=benchmark_default_device,
    )

    argparse_runtimes = benchmark_parser.add_argument(
        "--runtimes",
        choices=sum(SUPPORTED_DEVICES.values(), []),
        nargs="+",
        dest="runtimes",
        help="Runtime(s) for the selected device",
        required=False,
        default=[],
    )

    benchmark_parser.add_argument(
        "--analyze-only",
        dest="analyze_only",
        help="Stop this command after the analysis phase",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--build-only",
        dest="build_only",
        help="Stop this command after the build phase",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--export-only",
        dest="export_only",
        help="Stop this command after the ONNX export in the build phase",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--resume",
        dest="resume",
        help="Resume a benchit run by skipping any input scripts that have already been visited",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--script-args",
        dest="script_args",
        type=str,
        nargs=1,
        help="Arguments to pass into the target script(s)",
    )

    benchmark_parser.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=0,
        help="Maximum depth to analyze within the model structure of the target script(s)",
    )

    benchmark_parser.add_argument(
        "--onnx-opset",
        dest="onnx_opset",
        type=int,
        default=build.DEFAULT_ONNX_OPSET,
        help=f"ONNX opset used when creating ONNX files (default={build.DEFAULT_ONNX_OPSET})",
    )

    benchmark_parser.add_argument(
        "--groq-compiler-flags",
        nargs="+",
        dest="groq_compiler_flags",
        help="Sets the groqit(compiler_flags=...) arg (default behavior is to use groqit()'s "
        "default compiler flags)",
        required=False,
        default=None,
    )

    benchmark_parser.add_argument(
        "--groq-assembler-flags",
        nargs="+",
        dest="groq_assembler_flags",
        help="Sets the groqit(assembler_flags=...) arg (default behavior is to use groqit()'s "
        "default assembler flags)",
        required=False,
        default=None,
    )

    benchmark_parser.add_argument(
        "--groq-num-chips",
        dest="groq_num_chips",
        help="Sets the groqit(num_chips=...) arg (default behavior is to let groqit() "
        "automatically select the number of chips)",
        required=False,
        default=None,
        type=int,
    )

    benchmark_parser.add_argument(
        "--groqview",
        dest="groqview",
        help="Enables GroqView for the build(s)",
        action="store_true",
    )

    #######################################
    # Subparser for the "cache" command
    #######################################

    cache_parser = subparsers.add_parser(
        "cache",
        help="Commands for managing the build cache",
    )

    cache_subparsers = cache_parser.add_subparsers(
        title="cache",
        help="Commands for managing the build cache",
        required=True,
        dest="cache_cmd",
    )

    #######################################
    # Parser for the "cache report" command
    #######################################

    report_parser = cache_subparsers.add_parser(
        "report", help="Generate reports in CSV format"
    )
    report_parser.set_defaults(func=report.summary_spreadsheet)

    report_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dirs",
        help=(
            "One or more build cache directories to generate the report "
            f"(defaults to {filesystem.DEFAULT_CACHE_DIR})"
        ),
        default=[filesystem.DEFAULT_CACHE_DIR],
        nargs="*",
    )

    report_parser.add_argument(
        "-r",
        "--report-dir",
        dest="report_dir",
        help="Path to folder where report will be saved (defaults to current working directory)",
        required=False,
        default=os.getcwd(),
    )

    #######################################
    # Parser for the "cache list" command
    #######################################

    list_parser = cache_subparsers.add_parser(
        "list", help="List all builds in a target cache"
    )
    list_parser.set_defaults(func=filesystem.print_available_builds)

    list_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="The builds in this build cache directory will printed to the terminal "
        f" (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    #######################################
    # Parser for the "cache stats" command
    #######################################

    stats_parser = cache_subparsers.add_parser(
        "stats", help="Print stats about a build in a target cache"
    )
    stats_parser.set_defaults(func=print_stats)

    stats_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="The stats of a build in this build cache directory will printed to the terminal "
        f" (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    stats_parser.add_argument(
        "build_name",
        help="Name of the specific build whose stats are to be printed, within the cache directory",
    )

    #######################################
    # Parser for the "cache delete" command
    #######################################

    delete_parser = cache_subparsers.add_parser(
        "delete", help="Delete one or more builds in a build cache"
    )
    delete_parser.set_defaults(func=filesystem.delete_builds)

    delete_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="Search path for builds " f"(defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    delete_group = delete_parser.add_mutually_exclusive_group(required=True)

    delete_group.add_argument(
        "build_name",
        nargs="?",
        help="Name of the specific build to be deleted, within the cache directory",
    )

    delete_group.add_argument(
        "--all",
        dest="delete_all",
        help="Delete all builds in the cache directory",
        action="store_true",
    )

    #######################################
    # Parser for the "cache clean" command
    #######################################

    clean_parser = cache_subparsers.add_parser(
        "clean",
        help="Remove the build artifacts from one or more builds in a build cache",
    )
    clean_parser.set_defaults(func=filesystem.clean_builds)

    clean_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="Search path for builds " f"(defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    clean_group = clean_parser.add_mutually_exclusive_group(required=True)

    clean_group.add_argument(
        "build_name",
        nargs="?",
        help="Name of the specific build to be cleaned, within the cache directory",
    )

    clean_group.add_argument(
        "--all",
        dest="clean_all",
        help="Clean all builds in the cache directory",
        action="store_true",
    )

    #######################################
    # Parser for the "cache location" command
    #######################################

    cache_location_parser = cache_subparsers.add_parser(
        "location",
        help="Print the location of the default build cache directory",
    )
    cache_location_parser.set_defaults(func=filesystem.print_cache_dir)

    #######################################
    # Subparser for the "models" command
    #######################################

    models_parser = subparsers.add_parser(
        "models",
        help="Commands for managing the MLAgility benchmark's models",
    )

    """
    Design note: the `models` command is simple right now, however some additional ideas
        are documented in https://github.com/groq/mlagility/issues/247
    """

    models_subparsers = models_parser.add_subparsers(
        title="models",
        help="Commands for managing the MLAgility benchmark's models",
        required=True,
        dest="models_cmd",
    )

    models_location_parser = models_subparsers.add_parser(
        "location",
        help="Print the location of the MLAgility models directory",
    )
    models_location_parser.set_defaults(func=filesystem.print_models_dir)

    models_location_parser.add_argument(
        "--quiet",
        dest="verbose",
        help="Command output will only include the directory path",
        required=False,
        action="store_false",
    )

    #######################################
    # Parser for the "version" command
    #######################################

    version_parser = subparsers.add_parser(
        "version",
        help="Print the groqflow package version number",
    )
    version_parser.set_defaults(func=print_version)

    #######################################
    # Execute the command
    #######################################

    # The default behavior of this CLI is to run the build command
    # on a target script. If the user doesn't provide a command,
    # we alter argv to insert the command for them.

    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg not in subparsers.choices.keys() and "-h" not in first_arg:
            if "." in first_arg:
                sys.argv.insert(1, "benchmark")
            else:
                # Check how close we are from each of the valid options
                valid_options = list(subparsers.choices.keys())
                close_matches = get_close_matches(first_arg, valid_options)

                error_msg = f"Unexpected positional argument `benchit {first_arg}`. "
                if close_matches:
                    error_msg += f"Did you mean `benchit {close_matches[0]}`?"
                else:
                    error_msg += (
                        "The first positional argument must either be "
                        "an input file with the .py or .onnx file extension or "
                        f"one of the following commands: {valid_options}."
                    )
                raise exceptions.ArgError(error_msg)

    args = parser.parse_args()
    if args.func == benchmark_command:
        # Validate runtimes arg
        if args.runtimes:
            for runtime in args.runtimes:
                if runtime not in SUPPORTED_DEVICES[args.device]:
                    raise argparse.ArgumentError(
                        argparse_runtimes,
                        (
                            f"Runtime '{runtime}' is not valid for device '{args.device}'. "
                            f"Expected one of the following: {SUPPORTED_DEVICES[args.device]}."
                        ),
                    )
        # Assign default runtimes
        else:
            args.runtimes = [SUPPORTED_DEVICES[args.device][DEFAULT_RUNTIME]]

        # Ensure that the selected runtimes are supported by the backend
        if (
            "torch-eager" in args.runtimes or "torch-compiled" in args.runtimes
        ) and args.backend == "remote":
            raise argparse.ArgumentError(
                argparse_backend,
                (
                    "Remote backend is not available for 'torch-eager' and 'torch-compiled' runtimes."
                ),
            )
    args.func(args)


if __name__ == "__main__":
    main()
