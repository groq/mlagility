import sys
import argparse
import pkg_resources


def check_requirements(requirements_file: str, expect_failure: bool = False):
    """
    Receives the path to a requirements.txt file and raises an exception if the
    current environment does not comply with that requirements file.
    """
    # Read requirements file
    with open(requirements_file, encoding="utf8") as f:
        deps = f.read().splitlines()

    # Check for dependencies and print all issues instead of raising exceptions
    all_deps_ok = True
    for dep in deps:
        try:
            pkg_resources.require(dep)
        except pkg_resources.DistributionNotFound as e:
            if not expect_failure:
                print("DistributionNotFound: ", e, file=sys.stderr)
            all_deps_ok = False
        except pkg_resources.VersionConflict as e:
            if not expect_failure:
                print("VersionConflict: ", e, file=sys.stderr)
            all_deps_ok = False

    # Raise exceptions if needed
    if expect_failure:
        if all_deps_ok:
            raise Exception(
                "Expected failure when checking for dependencies, but all dependencies are fine."
            )
    else:
        if not all_deps_ok:
            raise pkg_resources.VersionConflict(
                "At least one package has a version conflict or was not found."
            )


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "requirements_file",
        help="Path to requirements.py file",
        action="store",
        type=str,
    )
    parser.add_argument(
        "--expect_failure",
        help="Raise an exception if all requirements are correctly installed",
        action="store_true",
        required=False,
    )
    parser.set_defaults(expect_failure=False)
    args = parser.parse_args()

    # Call check requirements functions
    check_requirements(args.requirements_file, args.expect_failure)
