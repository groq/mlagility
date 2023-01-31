import time
import os
import json
import pathlib
import groqflow.common.printing as printing
import groqflow.common.exceptions as exceptions
import groqflow.common.build
import mlagility.slurm as slurm
import mlagility.filesystem as filesystem
from mlagility.analysis.analysis import evaluate_script, TracerArgs, Action


def main(args):

    # Force the user to specify a legal cache dir in NFS if they are using slurm
    if not os.path.expanduser(args.cache_dir).startswith("/net") and args.use_slurm:
        raise ValueError(
            "You must specify a --cache-dir in `/net` when using groqit-util with Slurm, "
            f"however your current --cache-dir is set to {os.path.expanduser(args.cache_dir)}"
        )

    # Get a specific list of models to process
    available_scripts = filesystem.get_available_scripts(args.search_dir)

    # Filter based on the model names provided by the user
    if args.input_scripts == ["all"]:
        scripts = [
            os.path.join(args.search_dir, script) for script in available_scripts
        ]
    else:
        scripts = []
        for user_script in args.input_scripts:
            user_script_path = os.path.join(args.search_dir, user_script)
            if os.path.exists(user_script_path):
                scripts.append(user_script_path)
            else:
                raise exceptions.GroqitArgError(
                    f"Script could not be found: {user_script_path}"
                )

    # Decode benchit args into TracerArgs flags
    if args.analyze_only:
        actions = [
            Action.ANALYZE,
        ]
    elif args.build_only:
        actions = [
            Action.ANALYZE,
            Action.BUILD,
        ]
    else:
        actions = [
            Action.ANALYZE,
            Action.BUILD,
            Action.BENCHMARK,
        ]

    if args.use_slurm:
        jobs = slurm.jobs_in_queue()
        if len(jobs) > 0:
            printing.log_warning(f"There are already slurm jobs in your queue: {jobs}")
            printing.log_info(
                "Suggest quitting benchit, running 'scancel -u $USER' and trying again."
            )

    for script in scripts:
        for device in args.devices:
            if args.use_slurm:
                slurm.run_autogroq(
                    op="benchmark",
                    script=script,
                    search_dir=args.search_dir,
                    cache_dir=args.cache_dir,
                    rebuild=args.rebuild,
                    compiler_flags=args.compiler_flags,
                    assembler_flags=args.assembler_flags,
                    num_chips=args.num_chips,
                    groqview=args.groqview,
                    devices=device,
                    runtimes=args.runtimes,
                    ip=args.ip,
                    max_depth=args.max_depth,
                    analyze_only=args.analyze_only,
                    build_only=args.build_only,
                )

            else:

                tracer_args = TracerArgs(
                    input=script,
                    labels=None,
                    lean_cache=args.lean_cache,
                    targets=[],
                    max_depth=args.max_depth,
                    cache_dir=args.cache_dir,
                    rebuild=args.rebuild,
                    compiler_flags=args.compiler_flags,
                    assembler_flags=args.assembler_flags,
                    num_chips=args.num_chips,
                    groqview=args.groqview,
                    device=device,
                    actions=actions,
                )

                evaluate_script(tracer_args, args.script_args)

                # Print performance info
                if args.devices and Action.BENCHMARK in actions:
                    builds = filesystem.get_builds_from_script(
                        args.cache_dir, pathlib.Path(script).stem
                    )
                    for build in builds:
                        if "x86" in args.devices:
                            perf_file = os.path.join(
                                groqflow.common.build.output_dir(args.cache_dir, build),
                                "cpu_performance.json",
                            )
                            with open(perf_file, "r", encoding="utf8") as stream:
                                perf_data = json.loads(stream.read())
                                printing.log_info(
                                    f"Performance of device {perf_data['CPU Name']} is:"
                                )
                                print(f"Latency: {perf_data['Mean Latency(ms)']} ms")
                                print(f"Throughput: {perf_data['Throughput']} IPS")

    # Wait until all the Slurm jobs are done
    if args.use_slurm:
        while len(slurm.jobs_in_queue()) != 0:
            print(
                f"Waiting: {len(slurm.jobs_in_queue())} "
                f"jobs left in queue: {slurm.jobs_in_queue()}"
            )
            time.sleep(5)

    printing.log_success(
        "The 'benchmark' command is complete. Use the 'report' command to get a .csv "
        "file that summarizes results across all builds in the cache."
    )
