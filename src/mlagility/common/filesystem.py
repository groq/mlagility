import os
import pathlib
import groqflow.common.printing as printing
import groqflow.common.cache as cache

# Allow an environment variable to override the default
# location for the build cache
if os.environ.get("MLAGILITY_CACHE_DIR"):
    DEFAULT_CACHE_DIR = os.environ.get("MLAGILITY_CACHE_DIR")
else:
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/mlagility")


def full_model_path(corpora_dir, model_name):
    """
    Returns the absolute path of a model within a corpora
    """
    # FIXME: seems redundant with most of get_available_models()

    file_name = f"{model_name}.py"
    corpora = [
        f
        for f in os.listdir(corpora_dir)
        if os.path.isdir(os.path.join(corpora_dir, f))
    ]
    for corpus in corpora:
        full_path = f"{corpora_dir}/{corpus}/{file_name}"
        if os.path.isfile(full_path):
            return full_path
    raise ValueError(f"Model {model_name} not found in tree {corpora_dir}")


def get_available_models(corpora_dir):
    """
    Get all of the model.py files within the corpora located at `corpora_dir`

    def Corpus (noun): a collection of models
    def Corpora (noun): plural of "corpus"

    corpora_dir/
      corpus_1/
        model_1.py
        model_2.py
      corpus_2/
        model_3.py
        model_4.py
    """

    available_models = {}
    corpora = [
        pathlib.PurePath(corpus).name
        for corpus in os.listdir(os.path.abspath(corpora_dir))
        if os.path.isdir(os.path.join(corpora_dir, corpus))
    ]
    corpora.sort()

    # Loop over all corpora
    for corpus in corpora:
        # Get model names
        corpus_path = os.path.join(corpora_dir, corpus)
        model_names = [
            f.replace(".py", "")
            for f in os.listdir(corpus_path)
            if os.path.isfile(os.path.join(corpus_path, f)) and ".py" in f
        ]
        model_names.sort()
        available_models[corpus] = model_names

    return available_models


def print_available_models(args):
    available_models = get_available_models(args.corpora_dir)
    corpora = available_models.keys()

    # Loop over all corpora
    for corpus in corpora:
        if corpus in args.corpus_names:
            # Print in a nice table
            printing.log(
                f"{corpus} ({len(available_models[corpus])} models)\n\t",
                c=printing.Colors.BOLD,
            )
            printing.list_table(available_models[corpus])
            print()


def get_available_builds(cache_dir):
    """
    Get all of the build directories within the GroqFlow build cache
    located at `cache_dir`
    """

    builds = [
        pathlib.PurePath(build).name
        for build in os.listdir(os.path.abspath(cache_dir))
        if os.path.isdir(os.path.join(cache_dir, build))
    ]
    builds.sort()

    return builds


def print_available_builds(args):
    printing.log_info(f"Builds available in cache {args.cache_dir}:")
    builds = get_available_builds(args.cache_dir)
    printing.list_table(builds)
    print()


def delete_builds(args):

    if args.delete_all:
        builds = get_available_builds(args.cache_dir)
    else:
        builds = [args.build_name]

    for build in builds:
        build_path = os.path.join(args.cache_dir, build)
        if cache.rmdir(build_path):
            printing.log_info(f"Deleted build: {build}")
        else:
            printing.log_error(
                f"No build found with name: {build}. "
                "Try running `benchit cache list` to see the builds in your build cache."
            )


def get_builds_from_script(cache_dir, script_name):
    all_builds_in_cache = get_available_builds(cache_dir)
    script_builds = [x for x in all_builds_in_cache if script_name in x]

    return script_builds
