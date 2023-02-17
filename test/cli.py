"""
Tests focused on the command-level functionality of benchit CLI
"""

import os
from typing import List, Tuple, Any
import unittest
from unittest.mock import patch
import sys
import io
from pathlib import Path
import shutil
from contextlib import redirect_stdout
from mlagility.cli.cli import main as benchitcli
import mlagility.cli.report as report
from mlagility.common import filesystem
import mlagility.api.ortmodel as ortmodel
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

example_sequence_file = "example_sequence.py"

extras_dot_py = {
    example_sequence_file: """
from groqflow.justgroqit.stage import Sequence
import groqflow.justgroqit.export as export
from mlagility.common.groqflow_helpers import SuccessStage

def get_sequence():
    return Sequence(
        unique_name="example_sequence",
        monitor_message="Example pytorch sequence that only exports ONNX",
        stages=[
            export.ExportPytorchModel(),
            SuccessStage(),
        ],
        enable_model_validation=True,
    )
    """
}

# Create a test directory and make it the CWD
test_dir = "cli_test_dir"
cache_dir = "cache-dir"
dirpath = Path(test_dir)
if dirpath.is_dir():
    shutil.rmtree(dirpath)
os.makedirs(test_dir)
os.chdir(test_dir)

corpus_dir = "test_corpus"
extras_dir = os.path.join(corpus_dir, "extras")
os.makedirs(extras_dir, exist_ok=True)

for key, value in test_scripts_dot_py.items():
    model_path = os.path.join(corpus_dir, key)

    with open(model_path, "w", encoding="utf") as f:
        f.write(value)

for key, value in extras_dot_py.items():
    file_path = os.path.join(extras_dir, key)

    with open(file_path, "w", encoding="utf") as f:
        f.write(value)


def strip_dot_py(test_script_file: str) -> str:
    return test_script_file.split(".")[0]


def assert_success_of_builds(
    test_script_files: List[str],
    info_property: Tuple[str, Any] = None,
    check_perf: bool = False,
):
    # Figure out the build name by surveying the build cache
    # for a build that includes test_script_name in the name
    # TODO: simplify this code when
    # https://github.com/groq/mlagility/issues/16
    # is done
    builds = cache.get_all(cache_dir)

    for test_script in test_script_files:
        test_script_name = strip_dot_py(test_script)
        script_build_found = False

        for build_state_file in builds:
            if test_script_name in build_state_file:
                build_state = build.load_state(state_path=build_state_file)
                assert build_state.build_status == build.Status.SUCCESSFUL_BUILD
                script_build_found = True

                if info_property is not None:
                    assert (
                        build_state.info.__dict__[info_property[0]] == info_property[1]
                    ), f"{build_state.info.__dict__[info_property[0]]} == {info_property[1]}"

                if check_perf:
                    cpu_model = ortmodel.load(
                        build_state.config.build_name, cache_dir=build_state.cache_dir
                    )
                    assert cpu_model._mean_latency > 0
                    assert cpu_model._throughput > 0

        assert script_build_found


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        cache.rmdir(cache_dir)

        return super().setUp()

    def test_001_cli_single(self):

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script])

    def test_002_cli_search_dir(self):

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            test_script,
            "-s",
            corpus_dir,
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script])

    def test_003_cli_build_dir(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        test_scripts = test_scripts_dot_py.keys()

        testargs = [
            "benchit",
            "benchmark",
            "--all",
            "-s",
            corpus_dir,
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds(test_scripts)

    @unittest.skip(
        "Will be implemented in https://github.com/groq/mlagility/issues/142"
    )
    def test_004_cli_report(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        test_scripts = test_scripts_dot_py.keys()

        # Build the test corpus so we have builds to report
        testargs = [
            "benchit",
            "benchmark",
            "--all",
            "-s",
            corpus_dir,
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        testargs = [
            "benchit",
            "cache",
            "report",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure our test models are mentioned in
        # the summary csv

        summary_csv_path = os.path.join(cache_dir, report.summary_filename)
        with open(summary_csv_path, "r", encoding="utf8") as summary_csv:
            summary_csv_contents = summary_csv.read()
            for test_script in test_scripts:
                script_name = strip_dot_py(test_script)
                assert script_name in summary_csv_contents

    def test_005_cli_list(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to list
        testargs = [
            "benchit",
            "benchmark",
            "--all",
            "-s",
            corpus_dir,
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name in f.getvalue()

    def test_006_cli_delete(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus
        # - test_cli_list

        # Build the test corpus so we have builds to delete
        testargs = [
            "benchit",
            "benchmark",
            "--all",
            "-s",
            corpus_dir,
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure we can list the builds in the cache
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name in f.getvalue()

        # Delete the builds
        testargs = [
            "benchit",
            "cache",
            "delete",
            "--all",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure the builds are gone
        with redirect_stdout(io.StringIO()) as f:
            testargs = [
                "benchit",
                "cache",
                "list",
                "--cache-dir",
                cache_dir,
            ]
            with patch.object(sys, "argv", testargs):
                benchitcli()

        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            assert script_name not in f.getvalue()

    def test_007_cli_stats(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_corpus

        # Build the test corpus so we have builds to print
        testargs = [
            "benchit",
            "benchmark",
            "--all",
            "-s",
            corpus_dir,
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        # Make sure we can print the builds in the cache
        for test_script in test_scripts_dot_py.keys():
            script_name = strip_dot_py(test_script)
            builds = filesystem.get_builds_from_script(cache_dir, script_name)

            for build_name in builds:
                with redirect_stdout(io.StringIO()) as f:
                    testargs = [
                        "benchit",
                        "cache",
                        "stats",
                        build_name,
                        "--cache-dir",
                        cache_dir,
                    ]
                    with patch.object(sys, "argv", testargs):
                        benchitcli()

                    assert script_name in f.getvalue()

    def test_008_cli_version(self):

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

    def test_009_cli_benchit_args(self):

        # NOTE: this is not a unit test, it relies on other command
        # If this test is failing, make sure the following tests are passing:
        # - test_cli_single

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        # Set as many benchit args as possible
        testargs = [
            "benchit",
            "benchmark",
            test_script,
            "-s",
            corpus_dir,
            "--rebuild",
            "always",
            "--groq-num-chips",
            "1",
            "--groqview",
            "--groq-compiler-flags=--large-program",
            "--groq-assembler-flags=--no-metrics",
            "--build-only",
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script])

    def test_cli_sequence(self):

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--build-only",
            "--sequence-file",
            os.path.join(extras_dir, example_sequence_file),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds(
            [test_script], ("all_build_stages", ["export_pytorch", "set_success"])
        )

    def test_009_cli_benchmark(self):

        # Test the first model in the corpus
        test_script = list(test_scripts_dot_py.keys())[0]

        testargs = [
            "benchit",
            "benchmark",
            os.path.join(corpus_dir, test_script),
            "--cache-dir",
            cache_dir,
        ]
        with patch.object(sys, "argv", testargs):
            benchitcli()

        assert_success_of_builds([test_script], None, check_perf=True)


if __name__ == "__main__":
    unittest.main()
