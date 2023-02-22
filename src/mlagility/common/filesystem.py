import os
import shutil
import glob
import pathlib
import groqflow.common.printing as printing
import groqflow.common.cache as cache
import groqflow.common.build as build
import groqflow.common.exceptions as exc

# Allow an environment variable to override the default
# location for the build cache
if os.environ.get("MLAGILITY_CACHE_DIR"):
    DEFAULT_CACHE_DIR = os.environ.get("MLAGILITY_CACHE_DIR")
else:
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/mlagility")

CACHE_MARKER = ".mlacache"
BUILD_MARKER = ".mlabuild"


class CacheError(exc.GroqFlowError):
    """
    Raise this exception when the cache is being accessed incorrectly
    """


def make_build_dir(cache_dir: str, build_name: str):
    # Create the build and cache directories, and put hidden files in them
    # to mark them as such.
    build_dir = build.output_dir(cache_dir, build_name)
    os.makedirs(build_dir, exist_ok=True)

    # File that indicates that the directory is an MLAgility cache directory
    cache_file_path = os.path.join(cache_dir, CACHE_MARKER)
    open(cache_file_path, mode="w", encoding="utf").close()

    # File that indicates that the directory is an MLAgility build directory
    build_file_path = os.path.join(build_dir, BUILD_MARKER)
    open(build_file_path, mode="w", encoding="utf").close()


def check_cache_dir(cache_dir: str):
    cache_file_path = os.path.join(cache_dir, CACHE_MARKER)
    if not os.path.isfile(cache_file_path):
        raise CacheError(f"{cache_dir} is not a valid MLAgility cache")


def is_build_dir(cache_dir: str, build_name: str):
    build_dir = build.output_dir(cache_dir, build_name)
    build_file_path = os.path.join(build_dir, BUILD_MARKER)
    return os.path.isfile(build_file_path)


def clean_output_dir(cache_dir: str, build_name: str) -> None:
    """
    Delete all elements of the output directory that are not human readable
    """
    output_dir = os.path.join(cache_dir, build_name)
    if os.path.isdir(output_dir) and is_build_dir(cache_dir, build_name):
        output_dir = os.path.expanduser(output_dir)
    else:
        raise CacheError(f"No build found at {output_dir}")

    # Remove files that do not have an allowed extension
    allowed_extensions = (".txt", ".out", ".yaml", ".json")
    all_paths = glob.glob(f"{output_dir}/**/*", recursive=True)
    for path in all_paths:
        if os.path.isfile(path) and not path.endswith(allowed_extensions):
            os.remove(path)

    # Remove all empty folders
    for path in all_paths:
        if os.path.isdir(path):
            if len(os.listdir(path)) == 0:
                shutil.rmtree(path)


def get_available_scripts(search_dir: str):
    scripts = [
        f
        for f in os.listdir(search_dir)
        if os.path.isfile(os.path.join(search_dir, f)) and ".py" in f
    ]

    return scripts


def get_available_builds(cache_dir):
    """
    Get all of the build directories within the GroqFlow build cache
    located at `cache_dir`
    """

    check_cache_dir(cache_dir)

    builds = [
        pathlib.PurePath(build).name
        for build in os.listdir(os.path.abspath(cache_dir))
        if os.path.isdir(os.path.join(cache_dir, build))
        and is_build_dir(cache_dir, build)
    ]
    builds.sort()

    return builds


def print_available_builds(args):
    printing.log_info(f"Builds available in cache {args.cache_dir}:")
    builds = get_available_builds(args.cache_dir)
    printing.list_table(builds)
    print()


def delete_builds(args):

    check_cache_dir(args.cache_dir)

    if args.delete_all:
        builds = get_available_builds(args.cache_dir)
    else:
        builds = [args.build_name]

    for build in builds:
        build_path = os.path.join(args.cache_dir, build)
        if is_build_dir(args.cache_dir, build):
            cache.rmdir(build_path)
            printing.log_info(f"Deleted build: {build}")
        else:
            raise CacheError(
                f"No build found with name: {build}. "
                "Try running `benchit cache list` to see the builds in your build cache."
            )


def clean_builds(args):

    check_cache_dir(args.cache_dir)

    if args.clean_all:
        builds = get_available_builds(args.cache_dir)
    else:
        builds = [args.build_name]

    for build in builds:
        if is_build_dir(args.cache_dir, build):
            clean_output_dir(args.cache_dir, build)
            printing.log_info(f"Removed the build artifacts from: {build}")
        else:
            raise CacheError(
                f"No build found with name: {build}. "
                "Try running `benchit cache list` to see the builds in your build cache."
            )


def get_builds_from_script(cache_dir, script_name):
    all_builds_in_cache = get_available_builds(cache_dir)
    script_builds = [x for x in all_builds_in_cache if script_name in x]

    return script_builds
