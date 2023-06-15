import os
import shutil
import glob
import pathlib
import datetime
from typing import Dict, List
import importlib.util
import yaml
from fasteners import InterProcessLock
import onnxflow.common.printing as printing
import onnxflow.common.cache as cache
import onnxflow.common.build as build
import onnxflow.common.exceptions as exc
from mlagility.common import labels


# Allow an environment variable to override the default
# location for the build cache
if os.environ.get("MLAGILITY_CACHE_DIR"):
    DEFAULT_CACHE_DIR = os.environ.get("MLAGILITY_CACHE_DIR")
else:
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/mlagility")

CACHE_MARKER = ".mlacache"
BUILD_MARKER = ".mlabuild"

# Locate the models directory
MODELS_DIR = importlib.util.find_spec("mlagility_models").submodule_search_locations[0]


def clean_script_name(script_path: str) -> str:
    # Trim the ".py"
    return pathlib.Path(script_path).stem


class CacheError(exc.Error):
    """
    Raise this exception when the cache is being accessed incorrectly
    """


def _load_yaml(file) -> Dict:
    if os.path.isfile(file):
        with open(file, "r", encoding="utf8") as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)
    else:
        return {}


def _save_yaml(dict: Dict, file):
    with open(file, "w", encoding="utf8") as outfile:
        yaml.dump(dict, outfile)


def print_yaml_file(file_path, description):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            printing.log_info(f"The {description} for {file_path} are:")
            print(file.read())
    else:
        raise CacheError(
            f"No {description} found at {file_path}. "
            "Try running `benchit cache list` to see the builds in your build cache."
        )


class CacheDatabase:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    @property
    def _database_file(self) -> str:
        return os.path.join(self.cache_dir, ".cache_database.yaml")

    @property
    def _database(self) -> Dict[str, Dict[str, str]]:
        return _load_yaml(self._database_file)

    def dbtransaction(func):
        def safe_transaction(self, *args, **kwargs):
            with InterProcessLock(os.path.join(self.cache_dir, ".cache_database.lock")):
                return func(self, *args, **kwargs)

        return safe_transaction

    @dbtransaction
    def script_in_database(self, script_name) -> bool:
        return script_name in self._database.keys()

    def _validate_script_in_database(self, script_name: str, func_name: str):
        if not self.script_in_database(script_name):
            raise CacheError(
                f"This is a bug. {func_name}() was called with a script_name "
                "that has not been added to the database yet."
            )

    @dbtransaction
    def exists(self) -> bool:
        return len(self._database) > 0

    @dbtransaction
    def add_script(self, script_name: str):
        database_dict = self._database

        if script_name not in database_dict.keys():
            database_dict[script_name] = {}

        _save_yaml(database_dict, self._database_file)

    @dbtransaction
    def add_build(self, script_name, build_name):
        self.add_script(script_name)

        database_dict = self._database

        database_dict[script_name][build_name] = datetime.datetime.now()

        _save_yaml(database_dict, self._database_file)

    @dbtransaction
    def remove_script(self, script_name: str) -> Dict[str, Dict[str, str]]:
        self._validate_script_in_database(script_name, "remove_script")

        database_dict = self._database

        database_dict.pop(script_name)

        _save_yaml(database_dict, self._database_file)

        return database_dict

    @dbtransaction
    def remove_build(self, build_name: str):
        database_dict = self._database

        for script_name, script_builds in database_dict.items():
            if build_name in script_builds:

                script_builds.pop(build_name)

                if len(script_builds) == 0:
                    database_dict = self.remove_script(script_name)

        _save_yaml(database_dict, self._database_file)


def make_cache_dir(cache_dir: str):
    """
    Create the build and cache directories, and put hidden files in them
    to mark them as such.
    """

    os.makedirs(cache_dir, exist_ok=True)

    # File that indicates that the directory is an MLAgility cache directory
    cache_file_path = os.path.join(cache_dir, CACHE_MARKER)
    open(cache_file_path, mode="w", encoding="utf").close()


def make_build_dir(cache_dir: str, build_name: str):
    """
    Create the build and cache directories, and put hidden files in them
    to mark them as such.
    """
    make_cache_dir(cache_dir)

    build_dir = build.output_dir(cache_dir, build_name)
    os.makedirs(build_dir, exist_ok=True)

    # File that indicates that the directory is an MLAgility build directory
    build_file_path = os.path.join(build_dir, BUILD_MARKER)
    open(build_file_path, mode="w", encoding="utf").close()


def check_cache_dir(cache_dir: str):
    cache_file_path = os.path.join(cache_dir, CACHE_MARKER)
    if not os.path.isfile(cache_file_path):
        raise CacheError(
            f"{cache_dir} is not a cache directory generated by MLAgility. "
            "You can only clean, delete and generate reports for directories that "
            "have been generated by MLAgility. Set a different --cache-dir before "
            "trying again."
        )


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
    printing.list_table(builds, num_cols=1)
    print()


def delete_builds(args):

    check_cache_dir(args.cache_dir)

    if args.delete_all:
        builds = get_available_builds(args.cache_dir)
    else:
        builds = [args.build_name]

    for build in builds:
        db = CacheDatabase(args.cache_dir)
        db.remove_build(build)

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


def clean_build_name(build_name: str) -> str:
    """
    Remove hash from build name
    Build names have the format: <script_name>_<author>_hash
    """

    # Get everything except the trailing _<hash>
    return "_".join(build_name.split("_")[:-1])


def get_build_name(
    script_name: str, script_labels: Dict[str, List], model_hash: str = None
):
    """
    Create build name from script_name, labels and model hash
    """
    build_name = script_name
    if "author" in script_labels:
        build_name += f"_{script_labels['author'][0]}"
    if model_hash:
        build_name += f"_{model_hash}"
    return build_name


def get_builds_from_script(cache_dir, script):
    script_name = clean_script_name(script)
    script_labels = labels.load_from_file(script)
    all_builds_in_cache = get_available_builds(cache_dir)

    script_builds = [
        x
        for x in all_builds_in_cache
        if get_build_name(script_name, script_labels) == clean_build_name(x)
    ]

    return script_builds, script_name


def stats_file(cache_dir: str, build_name: str):
    return os.path.join(build.output_dir(cache_dir, build_name), "mlagility_stats.yaml")


def stats_file_exists():
    return os.path.isfile(stats_file)


def get_stats(cache_dir: str, build_name: str):
    stats_path = stats_file(cache_dir, build_name)
    return _load_yaml(stats_path)


def _save_stats(cache_dir: str, build_name: str, stats_dict: Dict):
    stats_path = stats_file(cache_dir, build_name)
    _save_yaml(stats_dict, stats_path)


def save_stat(cache_dir: str, build_name: str, key: str, value):
    """
    Save statistics to an yaml file in the build directory
    """

    stats_dict = get_stats(cache_dir, build_name)

    stats_dict[key] = value

    _save_stats(cache_dir, build_name, stats_dict)


def add_sub_stat(cache_dir: str, build_name: str, parent_key: str, key: str, value):
    """
    Save statistics to an yaml file in the build directory
    """

    stats_dict = get_stats(cache_dir, build_name)

    if parent_key in stats_dict.keys():
        dict_to_update = stats_dict[parent_key]
    else:
        dict_to_update = {}

    dict_to_update[key] = value
    stats_dict[parent_key] = dict_to_update

    _save_stats(cache_dir, build_name, stats_dict)


def print_cache_dir(_=None):
    printing.log_info(f"The default cache directory is: {DEFAULT_CACHE_DIR}")


def print_models_dir(args=None):
    if args.verbose:
        printing.log_info(f"The MLAgility models directory is: {MODELS_DIR}")
    else:
        print(MODELS_DIR)
