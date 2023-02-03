"""
Tests focused on the command-level functionality of benchit CLI
"""

import os
from typing import List
import unittest
from unittest.mock import patch
import sys
import io
from contextlib import redirect_stdout
from mlagility.cli import main as benchitcli
import mlagility.report as report
from mlagility import filesystem
import groqflow.common.build as build
import groqflow.common.cache as cache

# We generate a corpus on to the filesystem during the test
# to get around how weird bake tests are when it comes to
# filesystem access

test_scripts_dot_py = {
    "linear.py": """
import torch

torch.manual_seed(0)


class LinearTestModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearTestModel, self).__init__()
        self.fc = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        output = self.fc(x)
        return output


input_features = 10
output_features = 10

# Model and input configurations
model = LinearTestModel(input_features, output_features)
inputs = {"x": torch.rand(input_features)}

output = model(**inputs)

""",
    "linear2.py": """
import torch

torch.manual_seed(0)

# Define model class
class TwoLayerModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(TwoLayerModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        output = self.fc2(output)
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5
pytorch_model = TwoLayerModel(input_size, output_size)
inputs = {"x": torch.rand(input_size)}

pytorch_outputs = pytorch_model(**inputs)

# Print results
print(f"Pytorch_outputs: {pytorch_outputs}")
""",
}

corpus_dir = "test_corpus"
os.makedirs(corpus_dir, exist_ok=True)

for key, value in test_scripts_dot_py.items():
    model_path = os.path.join(corpus_dir, key)

    with open(model_path, "w", encoding="utf") as f:
        f.write(value)


def strip_dot_py(test_script_file: str) -> str:
    return test_script_file.split(".")[0]


def assert_success_of_builds(test_script_files: List[str]):
    # Figure out the build name by surveying the build cache
    # for a build that includes test_script_name in the name
    # TODO: simplify this code when
    # https://git.groq.io/code/Groq/-/issues/16110
    # is done
    builds = cache.get_all(filesystem.DEFAULT_CACHE_DIR)

    for test_script in test_script_files:
        test_script_name = strip_dot_py(test_script)
        script_build_found = False

        for build_state_file in builds:
            if test_script_name in build_state_file:
                build_state = build.load_state(state_path=build_state_file)
                assert build_state.build_status == build.Status.SUCCESSFUL_BUILD
                script_build_found = True

        assert script_build_found


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        cache.rmdir(filesystem.DEFAULT_CACHE_DIR)

        return super().setUp()

    def test_cli_single(self):

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            "-i",
            os.path.join(corpus_dir, test_script),
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script])

    def test_cli_search_dir(self):

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            "-s",
            corpus_dir,
            "-i",
            test_script,
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script])

    def test_cli_build_dir(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        test_scripts = test_scripts_dot_py.keys()

        testargs = [
            "benchit",
            "benchmark",
            "-s",
            corpus_dir,
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds(test_scripts)

    def test_cli_report(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        test_scripts = test_scripts_dot_py.keys()

        # Build the test corpus so we have builds to report
        testargs = [
            "benchit",
            "benchmark",
            "-s",
            corpus_dir,
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        testargs = [
            "benchit",
            "report",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure our test models are mentioned in
        # the summary csv

        summary_csv_path = os.path.join(
            filesystem.DEFAULT_CACHE_DIR, report.summary_filename
        )
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary_csv_contents = summary_csv.read()
            for test_script in test_scripts:
                script_name = strip_dot_py(test_script)
                assert script_name in summary_csv_contents

    def test_cli_list(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to list
        testargs = [
            "benchit",
            "benchmark",
            "-s",
            corpus_dir,
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "list",
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name in f.getvalue()

    def test_cli_delete(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus
        # - test_cli_list

        # Build the test corpus so we have builds to delete
        testargs = [
            "benchit",
            "benchmark",
            "-s",
            corpus_dir,
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "list",
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name in f.getvalue()

        # Delete the builds
        testargs = [
            "benchit",
            "delete",
            "--all",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure the builds are gone
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "list",
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name not in f.getvalue()

    def test_cli_version(self):

        # Get the version number
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "version",
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        # Make sure we get back a 3-digit number
        assert len(f.getvalue().split(".")) == 3

    def test_cli_benchit_args(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        # Set as many benchit args as possible
        testargs = [
            "benchit",
            "benchmark",
            "-s",
            corpus_dir,
            "-i",
            test_script,
            "--rebuild",
            "always",
            "--num-chips",
            "1",
            "--groqview",
            "--compiler-flags=--large-program",
            "--assembler-flags=--no-metrics",
            "--build-only",
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script])


if __name__ == "__main__":
    unittest.main()
