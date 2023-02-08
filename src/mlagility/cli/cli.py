import argparse
import os
import sys
import groqflow.common.build as build
import groqflow.common.printing as printing
import mlagility.cli.report as report
import mlagility.common.filesystem as filesystem
import mlagility.cli.benchmark as benchmark_command
from mlagility.version import __version__ as mlagility_version


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


def print_state(args):
    printing.log_info(
        f"The state of build {args.build_name} in cache {args.cache_dir} is:"
    )

    state_path = build.state_file(args.cache_dir, args.build_name)
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as file:
            print(file.read())
    else:
        printing.log_error(
            f"No build found with name: {build}. "
            "Try running `benchit cache list` to see the builds in your build cache."
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

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark the performance of one or more models"
    )
    benchmark_parser.set_defaults(func=benchmark_command.main)

    benchmark_parser.add_argument(
        "-s",
        "--search-dir",
        dest="search_dir",
        help="Path to a directory (defaults to the command line location), "
        "which serves as the search path for input scripts",
        required=False,
        default=os.getcwd(),
    )

    benchmark_parser.add_argument(
        "input_script",
        nargs="?",
        help="Name of the script (.py) file, within the search directory, "
        "to be benchmarked",
    )

    benchmark_parser.add_argument(
        "--all",
        dest="benchmark_all",
        help="Benchmark all models within all scripts in the search directory",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--use-slurm",
        dest="use_slurm",
        help="Execute on Slurm instead of using local compute resources",
        action="store_true",
    )

    benchmark_parser.add_argument(
        "--lean-cache",
        dest="lean_cache",
        help="Delete all build artifacts except for log files when the command completes",
        action="store_true",
    )

    # TODO: Implement this feature
    # slurm_ram_default = 64000
    # benchmark_parser.add_argument(
    #     "--slurm-ram",
    #     dest="slurm_ram",
    #     help="Amount of RAM, in MB, to allocate to each Slurm worker "
    #     f"(defaults to {slurm_ram_default})",
    #     required=False,
    #     default=slurm_ram_default,
    # )

    # TODO: Implement this feature
    # timeout_default = 60
    # benchmark_parser.add_argument(
    #     "--timeout",
    #     dest="timeout",
    #     help="Number of minutes to allow each build to run for, before it is canceled "
    #     f"(defaults to {timeout_default})",
    #     required=False,
    #     default=timeout_default,
    # )

    benchmark_parser.add_argument(
        "-d",
        "--cache-dir",
        dest="cache_dir",
        help="Build cache directory where the resulting build directories will "
        f"be stored (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
    )

    # TODO: Implement this feature
    # benchmark_parser.add_argument(
    #     "--sequence-file",
    #     dest="sequence_file",
    #     help="Path to a python script that implements the GroqFlow sequence.py template, "
    #     "which returns an instance of a custom build Sequence that will be passed to the "
    #     "groqit(sequence=...) arg (default behavior is to use the default groqit() "
    #     "build sequence)",
    #     required=False,
    #     default=None,
    # )

    benchmark_parser.add_argument(
        "--rebuild",
        dest="rebuild",
        help=f"Sets the cache rebuild policy (defaults to {build.DEFAULT_REBUILD_POLICY})",
        required=False,
        default=build.DEFAULT_REBUILD_POLICY,
    )

    benchmark_parser.add_argument(
        "--compiler-flags",
        nargs="+",
        dest="compiler_flags",
        help="Sets the groqit(compiler_flags=...) arg (default behavior is to use groqit()'s "
        "default compiler flags)",
        required=False,
        default=None,
    )

    benchmark_parser.add_argument(
        "--assembler-flags",
        nargs="+",
        dest="assembler_flags",
        help="Sets the groqit(assembler_flags=...) arg (default behavior is to use groqit()'s "
        "default assembler flags)",
        required=False,
        default=None,
    )

    benchmark_parser.add_argument(
        "--num-chips",
        dest="num_chips",
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

    benchmark_parser.add_argument(
        "--ip",
        dest="ip",
        help="IP address where the device is located (defaults to localhost)",
        required=False,
        default="localhost",
    )

    benchmark_default_device = "x86"
    benchmark_parser.add_argument(
        "--devices",
        choices=[
            "x86",
            "nvidia",
            "groq",
        ],
        nargs="+",
        dest="devices",
        help="Types(s) of hardware devices to be used for the benchmark "
        f'(defaults to ["{benchmark_default_device}"])',
        required=False,
        default=[benchmark_default_device],
    )

    benchmark_default_runtime = "ort"
    benchmark_parser.add_argument(
        "--runtime",
        choices=[
            "ort",
            "trt",
            "groq",
        ],
        nargs="+",
        dest="runtimes",
        help="Name(s) of the software runtimes to be used for the benchmark "
        f'(defaults to ["{benchmark_default_runtime}"])',
        required=False,
        default=["benchmark_default_runtime"],
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

    # TODO: Implement this feature
    # benchmark_parser.add_argument(
    #     "--sweep-file",
    #     dest="sweep_file",
    #     help="Path to a .yaml file that implements the GroqFlow sweep.yaml template, "
    #     "which defines the parameter sweeping behavior for this set of builds",
    #     required=False,
    #     default=None,
    # )

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
        dest="cache_dir",
        help="The reports will be generated based on the builds in this GroqFlow "
        f"build cache directory (defaults to {filesystem.DEFAULT_CACHE_DIR})",
        required=False,
        default=filesystem.DEFAULT_CACHE_DIR,
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
    stats_parser.set_defaults(func=print_state)

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
        script_name, _ = benchmark_command.decode_script_name(sys.argv[1])
        if sys.argv[1] not in subparsers.choices.keys() and script_name.endswith(".py"):
            sys.argv.insert(1, "benchmark")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
